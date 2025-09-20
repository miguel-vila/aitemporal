from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
import yt_dlp
from pydantic import BaseModel
import os
from pinecone import PineconeAsyncio
from pinecone.db_data import IndexAsyncio
from sentence_transformers import SentenceTransformer
import torch
from typing import Any, List, Dict
import asyncio
import functools
import whisper
from pyannote.audio import Pipeline
import ffmpeg
from pathlib import Path

import psutil
import tracemalloc
from pympler import muppy, summary
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

async def get_full_description(video_id: str, video_url: str) -> str:
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
    description = clean_description(await get_full_description(video_id, url))
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

def chunk_diarized_text(
    diarized_text: str,
    target_chunk_size: int = 1500,
    overlap_turns: int = 1
) -> List[str]:
    """
    Chunk diarized text by speaker turns while maintaining conversation context.

    Args:
        diarized_text: Text in format "[SPEAKER_XX]: content\n[SPEAKER_YY]: content\n..."
        target_chunk_size: Target size in characters for each chunk
        overlap_turns: Number of speaker turns to overlap between chunks for context

    Returns:
        List of text chunks that preserve speaker turn structure
    """
    if not diarized_text.strip():
        return []

    # Split by lines and parse speaker turns
    lines = diarized_text.strip().split('\n')
    speaker_turns = []

    for line in lines:
        line = line.strip()
        if line and line.startswith('[') and ']:' in line:
            # Parse speaker turn: [SPEAKER_XX]: content
            speaker_end = line.find(']:')
            if speaker_end > 0:
                speaker = line[1:speaker_end]
                content = line[speaker_end + 2:].strip()
                if content:  # Only add non-empty turns
                    speaker_turns.append((speaker, content))

    if not speaker_turns:
        return [diarized_text]  # Fallback to original text if parsing fails

    print(f'Found {len(speaker_turns)} speaker turns')

    chunks = []
    current_chunk_turns = []
    current_chunk_size = 0

    for i, (speaker, content) in enumerate(speaker_turns):
        turn_text = f"[{speaker}]: {content}"
        turn_size = len(turn_text)

        # If adding this turn would exceed target size and we have content, finalize chunk
        if current_chunk_size + turn_size > target_chunk_size and current_chunk_turns:
            # Create chunk from current turns
            chunk_text = '\n'.join(current_chunk_turns)
            chunks.append(chunk_text)

            # Start new chunk with overlap from previous chunk
            if overlap_turns > 0 and len(current_chunk_turns) > overlap_turns:
                # Keep last N turns for context
                overlap_turns_text = current_chunk_turns[-overlap_turns:]
                current_chunk_turns = overlap_turns_text.copy()
                current_chunk_size = sum(len(turn) for turn in overlap_turns_text)
            else:
                current_chunk_turns = []
                current_chunk_size = 0

        # Add current turn
        current_chunk_turns.append(turn_text)
        current_chunk_size += turn_size + 1  # +1 for newline

        # Log large turns
        if turn_size > target_chunk_size:
            print(f'Large speaker turn: {turn_size} chars from {speaker}')

    # Add final chunk if there are remaining turns
    if current_chunk_turns:
        chunk_text = '\n'.join(current_chunk_turns)
        chunks.append(chunk_text)

    print(f'Created {len(chunks)} chunks from diarized text')
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
diarization_semaphore = asyncio.Semaphore(1)  # Limit diarization to 1 concurrent process

async def download_audio(video_id: str, video_url: str) -> str:
    """Download audio from YouTube video and cache it locally"""
    audio_path = f"./audio/{video_id}.wav"

    # Check if audio file already exists
    if Path(audio_path).exists():
        print_debug(f"Audio cache hit for video {video_id}")
        return audio_path

    print_debug(f"Downloading audio for video {video_id}")

    # Download audio using yt-dlp
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'./audio/{video_id}.%(ext)s',
        'quiet': True,
        'no_warnings': True,
    }

    loop = asyncio.get_running_loop()

    def download_and_convert():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            downloaded_file = ydl.prepare_filename(info)

            # Convert to WAV using ffmpeg
            try:
                (
                    ffmpeg
                    .input(downloaded_file)
                    .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
                    .overwrite_output()
                    .run(quiet=True)
                )
                # Remove the original downloaded file
                os.remove(downloaded_file)
            except Exception as e:
                print(f"Error converting audio: {e}")
                # If conversion fails, rename the original file
                os.rename(downloaded_file, audio_path)

    await loop.run_in_executor(None, download_and_convert)
    return audio_path

async def get_diarized_transcript(video_id: str, video_url: str, cached_video: VideoRecord | None) -> str:
    """Get diarized transcript using Whisper + pyannote.audio"""
    global diarization_semaphore

    if cached_video and cached_video.transcript:
        print_debug(f"Transcript cache hit for video {video_id}")
        return cached_video.transcript

    async with diarization_semaphore:
        # Download audio
        audio_path = await download_audio(video_id, video_url)

        loop = asyncio.get_running_loop()

        def process_audio():
            # Load Whisper model
            whisper_model = whisper.load_model("base")

            # Transcribe with Whisper
            result = whisper_model.transcribe(audio_path, language="es")

            # Load diarization pipeline
            # Note: You need to accept user agreement and set HF_TOKEN for pyannote
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=os.getenv("HF_DIARIZATION_TOKEN")
            )

            # Perform diarization
            diarization = diarization_pipeline(audio_path)

            # Align transcription with diarization
            segments = result["segments"]
            diarized_text = []

            for segment in segments:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"].strip()

                # Find the speaker for this time period
                speaker = "SPEAKER_UNKNOWN"
                for turn, _, spk in diarization.itertracks(yield_label=True):
                    if turn.start <= start_time <= turn.end or turn.start <= end_time <= turn.end:
                        speaker = spk
                        break

                diarized_text.append(f"[{speaker}]: {text}")

            return "\n".join(diarized_text)

        transcript_text = await loop.run_in_executor(None, process_audio)

        # Cache the transcript
        await transcript_db.cache_transcript(video_id, transcript_text)
        return transcript_text

async def insert_video(video: Video, ytt_api: YouTubeTranscriptApi, model: SentenceTransformer, index: IndexAsyncio):
    # Check if video is already processed
    cached_video = await transcript_db.get_video(video.id)
    if cached_video and cached_video.processed:
        print_debug(f"Video {video.id} already processed, skipping")
        return

    transcript_text = await get_diarized_transcript(video.id, video.url, cached_video)
            
    chunks = chunk_diarized_text(transcript_text)
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
        proxy_config=WebshareProxyConfig(
            proxy_username=os.getenv("WEBSHARE_PROXY_USERNAME"), # type: ignore
            proxy_password=os.getenv("WEBSHARE_PROXY_PASSWORD"), # type: ignore
        )
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
