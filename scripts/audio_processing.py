import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

import json
import torch
import whisper
import os
from pyannote.audio import Pipeline

os.environ['PYANNOTE_AUDIO_PROGRESS'] = '1'
segments_model = "tiny"
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def compare_models_segments(video_id: str) -> None:
    n = 10
    out = {}
    models = ["tiny", "base", "small", "medium", "large", "turbo"]
    for model in models:
        if not os.path.exists(f'./audio/{video_id}.wav-segments-{model}.json'):
            print(f"Segments file for model {model} not found, skipping...")
            continue
        with open(f'./audio/{video_id}.wav-segments-{model}.json', 'r') as f:
            segments = json.load(f)
            out[model] = ' '.join([seg['text'] for seg in segments[:n]])
    for model in models:
        if model not in out:
            continue
        print(f"Model: {model}")
        print(out[model])
        print("\n")

def transcribe_and_diarize_audio(audio_path: str) -> str:
    print(f"Processing audio file: {audio_path}. Running on device: {device}")
    segments_filename = f'{audio_path}-segments-{segments_model}.json'
    if os.path.exists(segments_filename):
        print(f"Loading cached segments from {segments_filename}")
        with open(segments_filename, 'r') as f:
            segments = json.load(f)
    else:
        
        # Load Whisper model
        whisper_model = whisper.load_model(segments_model, device=device)
        print(f"Whisper model '{segments_model}' loaded on device: {device}")

        # Transcribe with Whisper
        result = whisper_model.transcribe(audio_path, language="es")
        print("Transcription completed for:", audio_path)
        with open(segments_filename, 'w') as f:
            json.dump(result["segments"], f)
        segments = result["segments"]


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
