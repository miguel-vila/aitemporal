from youtube_transcript_api import YouTubeTranscriptApi, FetchedTranscript
from youtube_transcript_api.proxies import WebshareProxyConfig
import yt_dlp
from pydantic import BaseModel
import os
from pinecone import PineconeAsyncio
from pinecone.db_data import IndexAsyncio
from sentence_transformers import SentenceTransformer
import torch
from typing import Any, List, Dict
import re
import asyncio
import functools

import psutil, tracemalloc
from pympler import muppy, summary
import csv
import threading
from pathlib import Path
from transcript_db import TranscriptDB, VideoRecord

debug = True
debug_memory = False

proc = psutil.Process(os.getpid())
tracemalloc.start(25)  # keep 25 frames of stack for later

model_semaphore = asyncio.Semaphore(1)

def report_mem(tag: str):
    global debug_memory
    if debug_memory:
        rss = proc.memory_info().rss / (1024**2)
        objs = muppy.get_objects()
        sum1 = summary.summarize(objs) # type: ignore
        # top = summary.format_(summary.sort(sum1, 'size'))[:10]
        print(f"\n=== {tag} ===")
        print(f"RSS: {rss:.1f} MiB  (pid={proc.pid})")
        summary.print_(sum1) # type: ignore
        # for line in top:
        #     print(line)
        
async def get_embedding(model: SentenceTransformer, transcripts: List[str]) -> List[List[float]]:
    global model_semaphore
    # model.encode is not thread-safe, so we limit concurrency to 1
    async with model_semaphore:
        loop = asyncio.get_running_loop()
        # Run encode in a thread to avoid blocking event loop
        encode_partial = functools.partial(
            model.encode, 
            sentences = transcripts,
            normalize_embeddings = True, # important for cosine similarity
            batch_size = 512 # good for CPU with 8-16GB RAM
        )
        embeddings = await loop.run_in_executor( # type: ignore
            None, 
            encode_partial
        )
        return embeddings.tolist() # type: ignore

class Video(BaseModel):
    id: str
    title: str
    url: str
    description: str
    transcript: str | None

ydl_opts: dict[str, Any] = {
    "quiet": True,
    "extract_flat": True,   # don't download, just get metadata
    "skip_download": True,
    "no_warnings": True,
}

# Global database instance
transcript_db: TranscriptDB = TranscriptDB()

def clean_description(description: str) -> str:
    segments = [
        'Capítulos',
        'Atemporal en instagram',
        'Escucha 13% Pasión por el trabajo en',
        'Apoyar Atemporal en Patreon' ,
        'Mentalidad 13%',
        'Recibe mi newsletter',
        'Escucha Atemporal en'
    ]
    for segment in segments:
        if segment in description:
            description = description.split(segment)[0]
    return description.strip()

def print_debug(msg: str):
    global debug
    if debug:
        print(msg)

async def get_full_description(video_id: str, video_url: str, title: str) -> str:
    """Get video description with SQLite caching support"""
    # Check cache first
    cached_description = await transcript_db.get_description(video_id)
    if cached_description:
        print_debug(f"Cache hit for video {video_id}")
        return cached_description
    
    print_debug(f"Cache miss for video {video_id}, fetching from API")
    # a new client to avoid synchronization locks. YoutubeDL is not thread-safe
    with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
        video_info = await asyncio.to_thread(ydl.extract_info, url=video_url, download=False)
        description = video_info.get('description', '')
        
        return description

async def channel_entry_to_video(entry: Dict[str, Any]) -> Video:
    video_id = entry['id']
    print_debug(f'processing {video_id}')
    cached_video = await transcript_db.get_video(video_id)
    if cached_video:
        return Video(
            id=cached_video.id,
            title=cached_video.title,
            url=cached_video.url,
            description=cached_video.description or '',
            transcript=cached_video.transcript
        )
    title = entry['title']
    url = entry['url']
    description = clean_description(await get_full_description(video_id, url, title))
    video = Video(id=video_id, title=title, url=url, description=description, transcript=None)
    await transcript_db.upsert_video(VideoRecord(
        id=video.id,
        title=video.title,
        url=video.url,
        description=video.description,
        processed=False,
        transcript=None
    ))
    print_debug(f'processed {video_id}')
    return video

async def download_videos_info(channel_url: str) -> list[Video]:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
        channel_info = ydl.extract_info(channel_url, download=False)
        return await asyncio.gather(*[channel_entry_to_video(entry) for entry in channel_info['entries']]) # type: ignore

def split_spanish_sentences(text: str) -> List[str]:
    # Lightweight Spanish sentence splitter (works well enough for transcripts).
    # For higher accuracy use spaCy: es_core_news_sm
    text = re.sub(r'\s+', ' ', text).strip()
    # Split on ., !, ? followed by space and a capital letter or end of text
    parts = re.split(r'(?<=[\.\!\?])\s+(?=[A-ZÁÉÍÓÚÑÜ]|$)', text)
    # Clean tiny leftovers
    return [p.strip() for p in parts if p.strip()]

def chunk_by_sentences(
    text: str,
    max_chars_log: int = 1500,
    overlap_sents: int = 2,
    # max_chars: int = 1400,   # soft cap to avoid giant chunks when sentences are long
) -> List[str]:
    sents = split_spanish_sentences(text)
    print(f'split in {len(sents)} sentences')
    chunks : List[str] = []

    for i in range(overlap_sents, len(sents)):
        chunk = ' '.join(sents[i-overlap_sents:i])
        if len(chunk) > max_chars_log:
            print(f'chunk with {len(chunk)} chars')
        chunks.append(chunk)            

    return chunks

def wrap_video_with_embedding(i: int, chunk_text: str, embedding: List[float], video: Video) -> tuple[str, List[float], Dict[str, Any]]:
    return (
        f'{video.id}_chunk{i}',
        embedding,
        {
            'id': video.id,
            'description': video.description,
            'text': chunk_text,
            'chunk_id': i,
            'title': video.title,
            'url': video.url
        }
    )

transcript_fetch_semaphore = asyncio.Semaphore(2)

async def get_transcript_text(video_id: str, ytt_api: YouTubeTranscriptApi, cached_video: VideoRecord | None) -> str:
    global transcript_fetch_semaphore
    if cached_video and cached_video.transcript:
        print_debug(f"Transcript cache hit for video {video_id}")
        return cached_video.transcript
    async with transcript_fetch_semaphore:
        loop = asyncio.get_running_loop()
        transcript: FetchedTranscript = await loop.run_in_executor( # type: ignore
            None,
            ytt_api.fetch,
            video_id, # type: ignore
            ["es"] # type: ignore
        )
    
    transcript_text = ' '.join([snippet.text for snippet in transcript.snippets]) # type: ignore
    # Cache the transcript
    await transcript_db.cache_transcript(video_id, transcript_text)
    return transcript_text

async def insert_video(video: Video, ytt_api: YouTubeTranscriptApi, model: SentenceTransformer, index: IndexAsyncio):    
    # Check if video is already processed
    cached_video = await transcript_db.get_video(video.id)
    if cached_video and cached_video.processed:
        print_debug(f"Video {video.id} already processed, skipping")
        return
    
    transcript_text = await get_transcript_text(video.id, ytt_api, cached_video)
            
    chunks = chunk_by_sentences(transcript_text)
    print(f'Split video {video.id} in {len(chunks)} chunks')
    if(len(chunks) == 0):
        print(f'Video {video.id} has no chunks. length={len(transcript_text)}.')
        print(f'Transcript text: {transcript_text[:200]}...{transcript_text[-200:]}')
    embeddings = await get_embedding(model, chunks)
    videos_with_embeddings = [
        wrap_video_with_embedding(i, chunk, embedding, video)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    await index.upsert(videos_with_embeddings, batch_size=100) # type: ignore
    
    # Mark as processed
    await transcript_db.mark_processed(video.id)

async def insert_video_and_report(video: Video, ytt_api: YouTubeTranscriptApi, model: SentenceTransformer, index: IndexAsyncio):
    await insert_video(video, ytt_api, model, index)
    report_mem(f'after {video.id}')

async def main():
    report_mem('baseline')
    
    # Initialize database
    await transcript_db.initialize()
    stats = await transcript_db.get_stats()
    print(f"Database stats: {stats}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer(embedding_model, device=device)
    index_host = 'atemporal-transcripts-f09myss.svc.aped-4627-b74a.pinecone.io'
    print("using webshare proxy username:", os.getenv("WEBSHARE_PROXY_USERNAME"))
    print("using webshare proxy password:", os.getenv("WEBSHARE_PROXY_PASSWORD"))
    ytt_api = YouTubeTranscriptApi(
        # proxy_config=WebshareProxyConfig(
        #     proxy_username=os.getenv("WEBSHARE_PROXY_USERNAME"), # type: ignore
        #     proxy_password=os.getenv("WEBSHARE_PROXY_PASSWORD"), # type: ignore
        # )
    )
    channel_url='https://www.youtube.com/@atemporalpodcast/videos'
    report_mem('initialized')

    videos = await download_videos_info(channel_url)
    print_debug(f"Found {len(videos)} videos.")
    async with PineconeAsyncio(api_key=os.getenv("PINECONE_API_KEY")) as pc:
        index: IndexAsyncio = pc.IndexAsyncio(host=index_host) # type: ignore
        report_mem('baseline before videos')
        await asyncio.gather(
            *[insert_video_and_report(video, ytt_api, model, index) for video in videos]
        )
        print_debug('Done upserting!')

if __name__ == "__main__":
    asyncio.run(main())
