from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from pydantic import BaseModel
import os
from pinecone import Pinecone
from pinecone.db_data import Index
from sentence_transformers import SentenceTransformer
import torch
from typing import Any, List, Dict
import re
import asyncio

import psutil, tracemalloc
from pympler import muppy, summary
import csv
import threading
from pathlib import Path

debug = False
debug_memory = True

proc = psutil.Process(os.getpid())
tracemalloc.start(25)  # keep 25 frames of stack for later

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
        
def get_embedding(model: SentenceTransformer, transcript: str) -> List[float]:
    return model.encode(transcript).tolist() # type: ignore

class Video(BaseModel):
    id: str
    title: str
    url: str
    description: str

ydl_opts: dict[str, Any] = {
    "quiet": True,
    "extract_flat": True,   # don't download, just get metadata
    "skip_download": True,
    "no_warnings": True,
}

# Global cache and lock for thread-safe operations
description_cache: Dict[str, str] = {}
cache_lock = threading.Lock()
CACHE_FILE = "video_descriptions_cache.csv"

def load_description_cache() -> None:
    """Load existing descriptions from CSV cache file"""
    global description_cache
    cache_path = Path(CACHE_FILE)
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                description_cache[row['id']] = row['description']
        print_debug(f"Loaded {len(description_cache)} cached descriptions")
    else:
        print_debug("No existing cache file found, starting fresh")

def save_description_to_cache(video_id: str, description: str) -> None:
    """Thread-safely append a new description to the CSV cache"""
    with cache_lock:
        # Add to in-memory cache
        description_cache[video_id] = description
        
        # Append to file
        cache_path = Path(CACHE_FILE)
        file_exists = cache_path.exists()
        
        with open(cache_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'description'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({'id': video_id, 'description': description})

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

async def get_full_description(video_id: str, video_url: str) -> str:
    """Get video description with caching support"""
    # Check cache first
    if video_id in description_cache:
        print_debug(f"Cache hit for video {video_id}")
        return description_cache[video_id]
    
    print_debug(f"Cache miss for video {video_id}, fetching from API")
    # a new client to avoid synchronization locks. YoutubeDL is not thread-safe
    with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
        video_info = await asyncio.to_thread(ydl.extract_info, url=video_url, download=False)
        description = video_info.get('description', '')
        
        # Save to cache
        save_description_to_cache(video_id, description)
        return description

async def channel_entry_to_video(entry: Dict[str, Any]) -> Video:
    video_id = entry['id']
    print_debug(f'processing {video_id}')
    title = entry['title']
    url = entry['url']
    description = clean_description(await get_full_description(video_id, url))
    video = Video(id=video_id, title=title, url=url, description=description)
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
    target_chars: int = 1000,
    overlap_sents: int = 2,
    max_chars: int = 1400,   # soft cap to avoid giant chunks when sentences are long
) -> List[str]:
    sents = split_spanish_sentences(text)
    chunks : List[str] = []
    buf: List[str] = []
    buf_len = 0

    i = 0
    while i < len(sents):
        s = sents[i]
        add_len = len(s) + (1 if buf else 0)  # +1 for space
        if buf and (buf_len + add_len > max_chars) and buf_len >= target_chars:
            # flush chunk
            chunk_text = " ".join(buf)
            chunks.append(chunk_text)
            # prepare overlap
            overlap = buf[-overlap_sents:] if overlap_sents > 0 else []
            buf = overlap[:]  # start next chunk with overlap
            buf_len = len(" ".join(buf)) if buf else 0
            # do not advance i here; we’ll attempt to add s again
        else:
            # add sentence and advance
            if buf:
                buf.append(s)
                buf_len += add_len
            else:
                buf = [s]
                buf_len = len(s)
            i += 1

    # flush remainder
    if buf:
        chunks.append(" ".join(buf))

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

def insert_video(video: Video, ytt_api: YouTubeTranscriptApi, model: SentenceTransformer, index: Index):
    transcript = ytt_api.fetch(video.id, languages = ["es"])
    transcript_text = ' '.join([snippet.text for snippet in transcript.snippets])
    chunks = chunk_by_sentences(transcript_text)
    print_debug(f'Split video {video.id} in {len(chunks)} chunks')
    index.upsert([ wrap_video_with_embedding(i, chunk, get_embedding(model, chunk), video) for i, chunk in enumerate(chunks) ]) # type: ignore

async def main():
    report_mem('baseline')
    
    # Load description cache before processing
    load_description_cache()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer(embedding_model, device=device)
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = 'atemporal-transcripts'
    ytt_api = YouTubeTranscriptApi()
    channel_url='https://www.youtube.com/@atemporalpodcast/videos'
    report_mem('initialized')

    videos = await download_videos_info(channel_url)
    print_debug(f"Found {len(videos)} videos.")
    index: Index = pc.Index(index_name) # type: ignore
    report_mem('baseline before videos')
    for video in videos:
        insert_video(video, ytt_api, model, index)
        report_mem(f'after {video.id}')
    print_debug('Done upserting!')

if __name__ == "__main__":
    asyncio.run(main())
