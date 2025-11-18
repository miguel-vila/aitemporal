import warnings

from pydantic import BaseModel

from S3AudioFile import S3AudioFile
import replicate
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

import json
import torch
import whisper
import os
import asyncio
from typing import List, Dict, Any
from pyannote.audio import Pipeline
from transcript_db import TranscriptDB

os.environ['PYANNOTE_AUDIO_PROGRESS'] = '1'
whisper_transcription_model = "large"
whisper_diarization_model = "3.1"
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Global database instance
transcript_db = TranscriptDB()

async def compare_models_segments(video_id: str) -> None:
    n = 10
    out = {}
    models = ["tiny", "base", "small", "medium", "large", "turbo"]
    await transcript_db.initialize()

    for model in models:
        cached_segments = await transcript_db.get_transcription(video_id, model)
        if not cached_segments:
            print(f"Segments for model {model} not found, skipping...")
            continue
        segments = json.loads(cached_segments)
        out[model] = ' '.join([seg['text'] for seg in segments[:n]])

    for model in models:
        if model not in out:
            continue
        print(f"Model: {model}")
        print(out[model])
        print("\n")
        
class AudioLine(BaseModel):
    speaker: str
    text: str
    start: float
    end: float

    def toJSON(self) -> str:
        return json.dumps({
            'speaker': self.speaker,
            'text': self.text,
            'start': self.start,
            'end': self.end
        }, ensure_ascii=False)
        
    @staticmethod
    def fromDict(data: dict[str,Any]) -> 'AudioLine':
        return AudioLine(
            speaker=data['speaker'],
            text=data['text'],
            start=data.get('start', 0.0),
            end=data.get('end', 0.0)
        )
    @staticmethod
    def from_db(data: str) -> list['AudioLine']:
        segments_json = json.loads(data)['segments']
        if type(segments_json) is list:
            return [AudioLine.fromDict(line) for line in segments_json]
        else:
            raise Exception(f"Invalid cached diarized transcript format. Expected list, got {type(segments_json)}")

class AudioLineEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, AudioLine):
            return o.model_dump()
        return super().default(o)

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

async def transcribe_with_model(audio_path: str, video_id: str) -> List[Dict[str, Any]]:
    # Override device for 'mps' to use 'cpu' because sparse not implemented for mps: https://github.com/pytorch/pytorch/issues/129842
    transcription_device = torch.device("cpu") if device.type == 'mps' else device

    # Check database cache first
    cached_segments = await transcript_db.get_transcription(video_id, whisper_transcription_model)
    if cached_segments:
        print(f"Loading cached segments from database for model {whisper_transcription_model}")
        return json.loads(cached_segments)

    print(f"Transcribing with Whisper model '{whisper_transcription_model}' on device: {transcription_device}")

    # Load Whisper model
    transcription_model = whisper.load_model(whisper_transcription_model, device=transcription_device)

    # Transcribe with Whisper
    result = transcription_model.transcribe(audio_path, language="es")
    segments = result["segments"]

    print(f"Transcription completed for {video_id} with model {whisper_transcription_model}")

    # Cache segments in database
    segments_json = json.dumps(segments)
    await transcript_db.cache_transcription(video_id, whisper_transcription_model, segments_json)

    return segments
