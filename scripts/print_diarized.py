#!/usr/bin/env python3
"""
Print diarized transcript for a video given a video ID, transcription and diarization models.

Usage:
    python print_diarized.py <video_id> <transcription_model> <diarization_model>

Example:
    python print_diarized.py go6PLP91MwY large 3.1
"""

import asyncio
import sys
from transcript_db import TranscriptDB

valid_transcription_models = ["tiny", "base", "small", "medium", "large", "turbo"]
valid_diarization_models = ["2.1", "3.1"]

async def main():
    if len(sys.argv) < 4:
        print("Usage: python print_diarized.py <video_id> <transcription_model> <diarization_model>", file=sys.stderr)
        print("", file=sys.stderr)
        print(f"Transcription models: {', '.join(valid_transcription_models)}", file=sys.stderr)
        print(f"Diarization models: {', '.join(valid_diarization_models)}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Example: python print_diarized.py go6PLP91MwY large 3.1", file=sys.stderr)
        sys.exit(1)

    video_id = sys.argv[1]
    transcription_model = sys.argv[2]
    diarization_model = sys.argv[3]

    if transcription_model not in valid_transcription_models:
        print(f"Error: Invalid model '{transcription_model}'. Must be one of: {', '.join(valid_transcription_models)}", file=sys.stderr)
        sys.exit(1)
    if diarization_model not in valid_diarization_models:
        print(f"Error: Invalid model '{diarization_model}'. Must be one of: {', '.join(valid_diarization_models)}", file=sys.stderr)
        sys.exit(1)

    db = TranscriptDB()
    await db.initialize()

    # Get diarized transcript
    diarized_text = await db.get_diarized_transcript(video_id, transcription_model, diarization_model)

    if not diarized_text:
        print(f"Error: No diarized transcript found for video '{video_id}' with transcription model '{transcription_model}' and diarization model '{diarization_model}'", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"Make sure the video has been processed with diarization.", file=sys.stderr)
        sys.exit(1)

    # Print to stdout
    print(diarized_text)


if __name__ == "__main__":
    asyncio.run(main())
