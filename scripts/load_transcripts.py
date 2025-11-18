import json
import yt_dlp
from pydantic import BaseModel
import os
from pinecone import PineconeAsyncio
from pinecone.db_data import IndexAsyncio
from sentence_transformers import SentenceTransformer
import torch
from audio_processing import transcribe_and_diarize_audio
from audio_models import AudioLine, AudioLineEncoder
from typing import Any, List, Dict
import asyncio
import functools
import ffmpeg
from pathlib import Path
import sys

import psutil
import tracemalloc
from pympler import muppy, summary
from transcript_db import TranscriptDB, VideoRecord
from name_recognition import extract_names_for_text, extract_names_for_title, normalize_name

debug = True
debug_memory = False

proc = psutil.Process(os.getpid())
tracemalloc.start(25)  # keep 25 frames of stack for later

model_semaphore = asyncio.Semaphore(1)

def report_mem(tag: str):
    global debug_memory
    if debug_memory:
        rss = proc.memory_info().rss / (1024**2)
        objs = muppy.get_objects() # type: ignore
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
        encode_partial = functools.partial( # type: ignore
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

ydl_opts: dict[str, Any] = {
    "quiet": True,
    "extract_flat": True,   # don't download, just get metadata
    "skip_download": True,
    "no_warnings": True,
    "ignore_no_formats_error": True,
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
    
    # td: unify proxy config setting
    webshare_username = os.getenv("WEBSHARE_PROXY_USERNAME")
    webshare_password = os.getenv("WEBSHARE_PROXY_PASSWORD")
    if webshare_username and webshare_password:
        ydl_opts['proxy'] = f'http://{webshare_username}-4:{webshare_password}@p.webshare.io:80'
    
    # a new client to avoid synchronization locks. YoutubeDL is not thread-safe
    with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
        video_info = await asyncio.to_thread(ydl.extract_info, url=video_url, download=False)
        description = video_info.get('description', '')
        
        return description

async def channel_entry_to_video(entry: Dict[str, Any]) -> Video:
    video_id = entry['id']
    print_debug(f'retrieving video info for {video_id}')
    cached_video = await transcript_db.get_video(video_id)
    if cached_video:
        return Video(
            id=cached_video.id,
            title=cached_video.title,
            url=cached_video.url,
            description=cached_video.description or ''
        )
    title = entry['title']
    url = entry['url']
    description = clean_description(await get_full_description(video_id, url))
    video = Video(id=video_id, title=title, url=url, description=description)
    await transcript_db.upsert_video(VideoRecord(
        id=video.id,
        title=video.title,
        url=video.url,
        description=video.description,
        processed=False
    ))
    print_debug(f'downloaded info and saved for {video_id}')
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
    print(f'Chunking diarized text: {diarized_text}')
    if not diarized_text.strip():
        return []

    # Split by lines and parse speaker turns
    lines = diarized_text.strip().split('\n')
    speaker_turns: List[tuple[str, str]] = []

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

    chunks: List[str] = []
    current_chunk_turns: List[str] = []
    current_chunk_size = 0

    for (speaker, content) in speaker_turns:
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

async def download_audio(video_id: str, video_url: str) -> str:
    """Download audio from YouTube video and cache it locally"""
    audio_path = f"./audio/{video_id}.wav"

    # Check if audio file already exists
    if Path(audio_path).exists():
        print_debug(f"Audio cache hit for video {video_id}")
        return audio_path

    print_debug(f"Downloading audio for video {video_id}")

    # Download audio using yt-dlp
    ydl_opts: dict[str, Any] = {
        'format': 'bestaudio/best',
        'outtmpl': f'./audio/{video_id}.%(ext)s',
        'quiet': False,
        'no_warnings': False,
        'socket_timeout': 60, # 1 minute timeout
        'http_chunk_size': 10485760,  # 10MB chunks
    }

    # Add proxy configuration if webshare env vars are available
    webshare_username = os.getenv("WEBSHARE_PROXY_USERNAME")
    webshare_password = os.getenv("WEBSHARE_PROXY_PASSWORD")
    if webshare_username and webshare_password:
        ydl_opts['proxy'] = f'http://{webshare_username}-4:{webshare_password}@p.webshare.io:80'

    loop = asyncio.get_running_loop()

    def download_and_convert():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            print_debug(f"Downloaded audio for video {video_id}")
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

    try:
      await asyncio.wait_for(
          loop.run_in_executor(None, download_and_convert),
          timeout=600  # 10 minutes
      )
    except asyncio.TimeoutError:
        print(f"Download timeout for video {video_id}")
        raise
    return audio_path

def condense_replicate_diarization(output: List[AudioLine]) -> List[AudioLine]:
    first_segment = output[0] # type: ignore
    last_speaker: str = first_segment.speaker # type: ignore
    buffer: list[AudioLine] = [AudioLine(speaker=first_segment.speaker, text=first_segment.text, start=first_segment.start, end=first_segment.end)] # type: ignore
    segments_output: list[AudioLine] = []

    for segment in output[1:]: # type: ignore
        speaker: str = segment.speaker
        text: str = segment.text
        if speaker != last_speaker:
            # close current buffer
            if buffer:
                combined_text = ' '.join(s.text for s in buffer)
                segments_output.append(
                    AudioLine(
                        speaker=last_speaker,
                        text=combined_text,
                        start=buffer[0].start,
                        end=buffer[-1].end
                    )
                )
                buffer = [
                    AudioLine(speaker=speaker, text=text, start=segment.start, end=segment.end)
                ]
            else:
                pass
        else:
            buffer.append(AudioLine(speaker=speaker, text=text, start=segment.start, end=segment.end))
        last_speaker = speaker
    # flush buffer
    if buffer:
        combined_text = ' '.join(s.text for s in buffer)
        segments_output.append(AudioLine(
            speaker=last_speaker,
            text=combined_text,
            start=buffer[0].start,
            end=buffer[-1].end
        ))
    return segments_output

async def get_diarized_transcript(video_id: str, video_url: str, interviewer_name: str, interviewee_name: str) -> str:
    """Get diarized transcript using Whisper + pyannote.audio"""
    audio_path = await download_audio(video_id, video_url)
    cached_diarized_transcript = await transcript_db.get_diarized_transcript(video_id)
    if cached_diarized_transcript:
        diarized_transcript_base = AudioLine.from_db(cached_diarized_transcript)
    else:
        diarized_transcript_base = await transcribe_and_diarize_audio(audio_path, video_id)
    diarized_transcript = condense_replicate_diarization(diarized_transcript_base)
    with open(f'diarized_{video_id}.json', 'w') as f:
        json.dump([line.model_dump() for line in diarized_transcript_base], f, cls=AudioLineEncoder, indent=2, ensure_ascii=False)
    with open(f'diarized_{video_id}_condensed.json', 'w') as f:
        json.dump([line.model_dump() for line in diarized_transcript], f, cls=AudioLineEncoder, indent=2, ensure_ascii=False)
    diarized_transcript = map_speakers(video_id, diarized_transcript, interviewer_name, interviewee_name)
    with open(f'diarized_{video_id}_mapped.json', 'w') as f:
        json.dump([line.model_dump() for line in diarized_transcript], f, cls=AudioLineEncoder, indent=2, ensure_ascii=False)
    # print_debug(f'First 5 lines of diarized transcript for video {video_id}:')
    # for audio_line in diarized_transcript[:5]:
    #     print_debug(f'{audio_line}')
    
    transcript_text = "\n".join([str(x) for x in diarized_transcript])
    return transcript_text

def map_speakers(video_id: str, audio_lines: List[AudioLine], interviewer_name: str, interviewee_name: str) -> List[AudioLine]:
    interviewer_speaker = None
    print(f'Interviewer name: {interviewer_name}, interviewee name: {interviewee_name}')
    interviewee_name_norm = normalize_name(interviewee_name)
    for line in audio_lines:
        line_names_norm = [normalize_name(n) for n in extract_names_for_text(line.text)]
        if interviewee_name_norm in line_names_norm:
            interviewer_speaker = line.speaker
            break
    if not interviewer_speaker:
        raise Exception(f"Couldn't identify interviewer speaker in transcript for video {video_id}")
    mapped_lines: List[AudioLine] = []
    for line in audio_lines:
        if line.speaker == interviewer_speaker:
            mapped_lines.append(AudioLine(speaker=interviewer_name, text=line.text, start=line.start, end=line.end))
        else:
            mapped_lines.append(AudioLine(speaker=interviewee_name, text=line.text, start=line.start, end=line.end))
    # Sanity check: ensure both speakers are present
    speakers = set(line.speaker for line in mapped_lines)
    print(f'speakers found after mapping: {speakers}')
    if interviewer_name not in speakers or interviewee_name not in speakers:
        raise Exception(f"Speaker mapping failed for video {video_id}: found speakers {speakers}, expected {interviewer_name} and {interviewee_name}")
    # Sanity check: ensure no unknown speakers
    if len(speakers) > 2:
        raise Exception(f"Too many speakers after mapping for video {video_id}: found speakers {speakers}, expected only {interviewer_name} and {interviewee_name}")
    return mapped_lines

video_titles_to_skip = [
    "M.U.T", # not an interview episode
    "#105 - Mateo Castaño y Sebastián Salazar - Antioquia emergente" # 2 interviewees, need to figure out
]

async def process_video(video: Video, model: SentenceTransformer, index: IndexAsyncio):
    if video.title in video_titles_to_skip:
        print_debug(f"Skipping video {video.id}")
        return
    # Check if video is already processed
    cached_video = await transcript_db.get_video(video.id)
    if cached_video and cached_video.processed:
        print_debug(f"Video {video.id} already processed, skipping")
        return
    
    recognized_names = extract_names_for_title(video.title)
    if len(recognized_names) == 0:
        raise Exception("Didn't recognize any name in title:", video.title)
    if len(recognized_names) > 1:
        raise Exception("Too many names recognized in title:", video.title, recognized_names)
    
    interviewer_name = "Andrés Acevedo"
    interviewee_name = recognized_names[0]

    transcript_text = await get_diarized_transcript(video.id, video.url, interviewer_name, interviewee_name)
    print(transcript_text)
            
    chunks = chunk_diarized_text(transcript_text)
    print(f'Split video {video.id} in {len(chunks)} chunks')
    # if(len(chunks) == 0):
    #     print(f'Video {video.id} has no chunks. length={len(transcript_text)}.')
    #     print(f'Transcript text: {transcript_text[:200]}...{transcript_text[-200:]}')
    embeddings = await get_embedding(model, chunks)
    videos_with_embeddings = [
        wrap_video_with_embedding(i, chunk, embedding, video)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    # await index.upsert(videos_with_embeddings, batch_size=100) # type: ignore
    
    # Mark as processed
    # await transcript_db.mark_processed(video.id)

async def process_video_and_report(video: Video, model: SentenceTransformer, index: IndexAsyncio):
    await process_video(video, model, index)
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
    channel_url='https://www.youtube.com/@atemporalpodcast/videos'
    report_mem('initialized')

    async with PineconeAsyncio(api_key=os.getenv("PINECONE_API_KEY")) as pc:
        index: IndexAsyncio = pc.IndexAsyncio(host=index_host) # type: ignore
        # if we receive a video ID as argument, process just that video
        if len(sys.argv) > 1:
            video_id = sys.argv[1]
            print_debug(f"Processing single video {video_id}")
            # Fetch only the specific video's metadata
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            cached_video = await transcript_db.get_video(video_id)
            if cached_video:
                video = Video(
                    id=cached_video.id,
                    title=cached_video.title,
                    url=cached_video.url,
                    description=cached_video.description or ''
                )
            else:
                # Fetch single video metadata directly
                description = clean_description(await get_full_description(video_id, video_url))
                with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
                    video_info = await asyncio.to_thread(ydl.extract_info, url=video_url, download=False)
                    title = video_info.get('title', '')
                video = Video(id=video_id, title=title, url=video_url, description=description)
                await transcript_db.upsert_video(VideoRecord(
                    id=video.id,
                    title=video.title,
                    url=video.url,
                    description=video.description,
                    processed=False
                ))
            await process_video_and_report(video, model, index)
            print_debug('Done upserting single video!')
        else:
            videos = await download_videos_info(channel_url)
            print_debug(f"Found {len(videos)} videos.")
            report_mem('baseline before videos')
            await asyncio.gather(
                *[process_video_and_report(video, model, index) for video in videos]
            )
            print_debug('Done upserting!')

if __name__ == "__main__":
    asyncio.run(main())
