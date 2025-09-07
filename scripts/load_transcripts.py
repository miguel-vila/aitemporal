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

proc = psutil.Process(os.getpid())
tracemalloc.start(25)  # keep 25 frames of stack for later

def report(tag: str):
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

async def get_full_description(ydl: yt_dlp.YoutubeDL, video_url: str) -> str:
    video_info = await asyncio.to_thread(ydl.extract_info, url= video_url, download=False)
    return video_info.get('description', '')

async def channel_entry_to_video(ydl: yt_dlp.YoutubeDL, entry: Dict[str, Any]) -> Video:
    video_id = entry['id']
    print(f'processing {video_id}')
    title = entry['title']
    url = entry['url']
    description = clean_description(await get_full_description(ydl, url))
    video = Video(id=video_id, title=title, url=url, description=description)
    print(f'processed {video_id}')
    return video

async def download_videos_info(channel_url: str) -> list[Video]:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
        channel_info = ydl.extract_info(channel_url, download=False)
        return await asyncio.gather(*[channel_entry_to_video(ydl, entry) for entry in channel_info['entries']]) # type: ignore

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
    print(f'Split video {video.id} in {len(chunks)} chunks')
    index.upsert([ wrap_video_with_embedding(i, chunk, get_embedding(model, chunk), video) for i, chunk in enumerate(chunks) ]) # type: ignore

async def main():
    report('baseline')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer(embedding_model, device=device)
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), )
    index_name = 'atemporal-transcripts'
    ytt_api = YouTubeTranscriptApi()
    channel_url='https://www.youtube.com/@atemporalpodcast/videos'
    report('initialized')

    videos = await download_videos_info(channel_url)
    print(f"Found {len(videos)} videos.")
    index: Index = pc.Index(index_name) # type: ignore
    report('baseline before videos')
    for video in videos:
        insert_video(video, ytt_api, model, index)
        report(f'after {video.id}')
    print('Done upserting!')

if __name__ == "__main__":
    asyncio.run(main())
