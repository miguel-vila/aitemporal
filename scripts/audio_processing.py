from pydantic import BaseModel
from S3AudioFile import S3AudioFile
import replicate
import json
from typing import List, Dict, Any
from transcript_db import TranscriptDB
from audio_models import AudioLine

# Global database instance
transcript_db = TranscriptDB()

s3_bucket_name = "atemporal-audios"

def replicate_ai_diarization(audio_url: str):
    """Perform diarization using Replicate AI's Whisper Diarization model."""
    input = {
        "file_url": audio_url,
        "language": "es"
    }

    return replicate.run(
        "thomasmol/whisper-diarization:1495a9cddc83b2203b0d8d3516e38b80fd1572ebc4bc5700ac1da56a9b3ed886",
        input=input
    ) # type: ignore

async def transcribe_and_diarize_audio(audio_path: str, video_id: str) -> List[AudioLine]:
    cached_diarized_transcript = await transcript_db.get_diarized_transcript(video_id)
    if cached_diarized_transcript:
        print(f"Loading cached diarized transcript from database for model {whisper_transcription_model}")
        return AudioLine.from_db(cached_diarized_transcript)

    with S3AudioFile(aws_profile='miguel-exps', video_id=video_id, local_path=audio_path, bucket_name=s3_bucket_name) as s3_audio:
        audio_url = s3_audio.public_url()
        print(f"Performing diarization using Replicate AI for video {video_id}...")
        replicate_output = replicate_ai_diarization(audio_url)
    await transcript_db.cache_diarized_transcript(video_id, json.dumps(replicate_output))
    return replicate_output
