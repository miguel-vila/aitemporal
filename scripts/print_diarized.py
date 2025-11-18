#!/usr/bin/env python3
"""
Print diarized transcript for a video given a video ID.

Usage:
    python print_diarized.py <video_id>

Example:
    python print_diarized.py go6PLP91MwY
"""

import asyncio
import sys
from transcript_db import TranscriptDB

async def main():
    if len(sys.argv) < 2:
        print("Usage: python print_diarized.py <video_id>", file=sys.stderr)
        print("", file=sys.stderr)
        print("Example: python print_diarized.py go6PLP91MwY", file=sys.stderr)
        sys.exit(1)

    video_id = sys.argv[1]

    db = TranscriptDB()
    await db.initialize()

    # Get diarized transcript
    diarized_text = await db.get_diarized_transcript(video_id)

    if not diarized_text:
        print(f"Error: No diarized transcript found for video '{video_id}'", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"Make sure the video has been processed with diarization.", file=sys.stderr)
        sys.exit(1)

    # Print to stdout
    print(diarized_text)


if __name__ == "__main__":
    asyncio.run(main())
