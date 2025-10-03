import warnings
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
segments_model = "tiny"
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
        
class AudioLine:
    def __init__(self, speaker: str, text: str):
        self.speaker = speaker
        self.text = text
    @staticmethod
    def from_str(lines: str) -> List['AudioLine']:
        result: List['AudioLine'] = []
        for line in lines.splitlines():
            if line.startswith('[') and ']' in line:
                speaker, text = line[1:].split(']', 1)
                result.append(AudioLine(speaker.strip(), text.strip()))
            else:
                raise ValueError(f"Line does not start with [SPEAKER]: {line}")
        return result
    
    def __str__(self) -> str:
        return f"[{self.speaker}] {self.text}"

async def transcribe_and_diarize_audio(audio_path: str, video_id: str) -> List[AudioLine]:
    cached_diarized_transcript = await transcript_db.get_diarized_transcript(video_id, segments_model)
    if cached_diarized_transcript:
        print(f"Loading cached diarized transcript from database for model {segments_model}")
        return cached_diarized_transcript

    segments = await transcribe_with_model(audio_path, video_id, segments_model)

    # Load diarization pipeline
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=os.getenv("HF_DIARIZATION_TOKEN"),
    )
    diarization_pipeline.to(device)

    # Perform diarization
    diarization = diarization_pipeline(audio_path)
    print("Diarization completed for:", audio_path)

    # Align transcription with diarization
    diarized_text: List[AudioLine] = []

    for segment in segments:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"].strip()

        # Find the speaker for this time period
        speaker = "SPEAKER_UNKNOWN"
        for turn, _, spk in diarization.itertracks(yield_label=True):
            if (
                turn.start <= start_time <= turn.end
                or turn.start <= end_time <= turn.end
            ):
                speaker = spk
                break

        diarized_text.append(AudioLine(speaker, text))

    return diarized_text

async def transcribe_with_model(audio_path: str, video_id: str, model_name: str) -> List[Dict[str, Any]]:
    """
    Transcribe audio with a specific Whisper model and cache results in database.

    Args:
        audio_path: Path to the audio file
        video_id: Video ID for caching
        model_name: Whisper model name ("tiny", "base", "small", "medium", "large", "turbo")

    Returns:
        List of transcript segments
    """
    # Override device for 'mps' to use 'cpu' because sparse not implemented for mps: https://github.com/pytorch/pytorch/issues/129842
    transcription_device = torch.device("cpu") if device.type == 'mps' else device

    # Check database cache first
    cached_segments = await transcript_db.get_transcription(video_id, model_name)
    if cached_segments:
        print(f"Loading cached segments from database for model {model_name}")
        return json.loads(cached_segments)

    print(f"Transcribing with Whisper model '{model_name}' on device: {transcription_device}")

    # Load Whisper model
    whisper_model = whisper.load_model(model_name, device=transcription_device)

    # Transcribe with Whisper
    result = whisper_model.transcribe(audio_path, language="es")
    segments = result["segments"]

    print(f"Transcription completed for {video_id} with model {model_name}")

    # Cache segments in database
    segments_json = json.dumps(segments)
    await transcript_db.cache_transcription(video_id, model_name, segments_json)

    return segments
