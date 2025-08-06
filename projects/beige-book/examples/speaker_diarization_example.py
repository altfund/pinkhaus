#!/usr/bin/env python3
"""
Example of using speaker diarization with the transcription pipeline.

This demonstrates how to enhance podcast transcriptions with speaker identification.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from beige_book.transcriber import AudioTranscriber
from beige_book.speaker_diarizer import (
    create_speaker_aware_transcription,
    SpeakerDiarizer,
)


def example_with_mock_diarization():
    """Example using mock diarization (no pyannote required)."""
    print("=== Example with Mock Diarization ===\n")

    # Initialize transcriber
    transcriber = AudioTranscriber(model_name="tiny")

    # Example audio file
    audio_file = "../../../resources/audio/harvard.wav"

    if not os.path.exists(audio_file):
        print(f"Please provide a valid audio file path. Current path: {audio_file}")
        print("You can update the 'audio_file' variable in this script.")
        return

    # Perform standard transcription
    print("Transcribing audio...")
    result = transcriber.transcribe_file(audio_file)

    # Enhance with speaker diarization (using mock for demo)
    print("Adding speaker diarization (mock mode)...")
    enhanced_result = create_speaker_aware_transcription(
        audio_file, result, use_mock=True
    )

    # Display results
    print(f"\nDetected {enhanced_result['num_speakers']} speakers")
    print("\nFirst 5 segments with speaker labels:")

    for i, segment in enumerate(enhanced_result["segments"][:5]):
        print(f"\n[{segment['speaker']}] {segment['start']} - {segment['end']}")
        print(f"  {segment['text']}")

    # Save enhanced result
    output_file = "transcription_with_speakers.json"
    with open(output_file, "w") as f:
        json.dump(enhanced_result, f, indent=2)
    print(f"\nFull result saved to: {output_file}")


def example_with_real_diarization():
    """Example using real pyannote diarization (requires installation and HF token)."""
    print("\n=== Example with Real Diarization ===\n")

    # Require Hugging Face token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is required for real speaker diarization.\n"
            "Please set: export HF_TOKEN='hf_...'\n"
            "And accept conditions at: https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    # Initialize components
    transcriber = AudioTranscriber(model_name="tiny")
    diarizer = SpeakerDiarizer(auth_token=hf_token)

    # Example audio file
    audio_file = "../../../resources/audio/harvard.wav"

    if not os.path.exists(audio_file):
        print(f"Please provide a valid audio file path. Current path: {audio_file}")
        return

    try:
        # Perform transcription
        print("Transcribing audio...")
        result = transcriber.transcribe_file(audio_file)

        # Perform diarization
        print("Performing speaker diarization...")
        diarization = diarizer.diarize_file(audio_file)

        # Align results
        segments = result.to_dict()["segments"]
        enhanced_segments = diarizer.align_with_transcription(diarization, segments)

        # Display resultsre
        print(f"\nDetected {diarization.num_speakers} speakers")
        print("\nSample segments with speaker labels:")

        for segment in enhanced_segments[:5]:
            print(f"\n[{segment['speaker']}] {segment['start']} - {segment['end']}")
            print(f"  {segment['text']}")

    except ImportError as e:
        print(f"Error: {e}")
        raise RuntimeError(
            "pyannote-audio is required for real speaker diarization. "
            "Please install it: pip install pyannote-audio"
        )


def demonstrate_output_formats():
    """Show how speaker information appears in different output formats."""
    print("\n=== Output Format Examples ===\n")

    # Create mock enhanced result
    enhanced_result = {
        "filename": "podcast_episode.wav",
        "file_hash": "abc123...",
        "language": "en",
        "num_speakers": 2,
        "has_speaker_labels": True,
        "segments": [
            {
                "start": "00:00:00.000",
                "end": "00:00:05.230",
                "text": "Welcome to our podcast!",
                "speaker": "SPEAKER_0",
            },
            {
                "start": "00:00:05.230",
                "end": "00:00:08.150",
                "text": "Thanks for having me.",
                "speaker": "SPEAKER_1",
            },
        ],
        "full_text": "Welcome to our podcast! Thanks for having me.",
    }

    # JSON format
    print("JSON format with speakers:")
    print(json.dumps(enhanced_result["segments"][:2], indent=2))

    # CSV-style format
    print("\nCSV format with speakers:")
    print("Start,End,Speaker,Text")
    for seg in enhanced_result["segments"]:
        print(f'{seg["start"]},{seg["end"]},{seg["speaker"]},"{seg["text"]}"')

    # Conversation format
    print("\nConversation format:")
    current_speaker = None
    for seg in enhanced_result["segments"]:
        if seg["speaker"] != current_speaker:
            print(f"\n{seg['speaker']}:")
            current_speaker = seg["speaker"]
        print(f"  {seg['text']}")


if __name__ == "__main__":
    print("Speaker Diarization Example\n")
    
    import argparse
    parser = argparse.ArgumentParser(description='Speaker diarization examples')
    parser.add_argument('--mock', action='store_true', help='Use mock diarization (testing only)')
    args = parser.parse_args()

    if args.mock:
        print("WARNING: Using mock diarization for testing only\n")
        example_with_mock_diarization()
    else:
        # Default to real diarization
        example_with_real_diarization()

    # Show output format examples
    demonstrate_output_formats()
