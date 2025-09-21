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

async def transcribe_and_diarize_audio(audio_path: str, video_id: str) -> str:
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
    diarized_text = []

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

        diarized_text.append(f"[{speaker}]: {text}")

    return "\n".join(diarized_text)

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

    # Check database cache first
    cached_segments = await transcript_db.get_transcription(video_id, model_name)
    if cached_segments:
        print(f"Loading cached segments from database for model {model_name}")
        return json.loads(cached_segments)

    print(f"Transcribing with Whisper model '{model_name}' on device: {device}")

    # Load Whisper model
    whisper_model = whisper.load_model(model_name, device=device)

    # Transcribe with Whisper
    result = whisper_model.transcribe(audio_path, language="es")
    segments = result["segments"]

    print(f"Transcription completed for {video_id} with model {model_name}")

    # Cache segments in database
    segments_json = json.dumps(segments)
    await transcript_db.cache_transcription(video_id, model_name, segments_json)

    return segments
