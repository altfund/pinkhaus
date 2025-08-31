#!/usr/bin/env python3
"""
Demo script showing speaker diarization in action.
"""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from beige_book.transcriber import AudioTranscriber


def demo_mock_diarization():
    """Demo using mock diarization (no HF token needed)."""
    print("=== Mock Diarization Demo ===\n")

    # Create a sample audio file path (you'll need to provide a real file)
    audio_file = "sample_podcast.wav"  # Change this to your audio file

    if not os.path.exists(audio_file):
        print("Please provide an audio file path in the script.")
        print("Edit the 'audio_file' variable to point to your podcast file.")
        return

    # Initialize transcriber
    transcriber = AudioTranscriber(model_name="tiny")

    # Transcribe with mock diarization
    print("Transcribing with speaker diarization (mock mode)...")
    result = transcriber.transcribe_file(audio_file, enable_diarization=True)

    # Display results
    result_dict = result.to_dict()
    print("\nTranscription complete!")
    print(f"Language: {result_dict['language']}")
    print(f"Number of speakers: {result_dict.get('num_speakers', 'Unknown')}")
    print(f"Total segments: {len(result_dict['segments'])}")

    # Show first few segments with speakers
    print("\nFirst 5 segments with speaker labels:")
    for i, seg in enumerate(result_dict["segments"][:5]):
        speaker = seg.get("speaker", "UNKNOWN")
        print(f"\n[{speaker}] {seg['start']} - {seg['end']}")
        print(f"  {seg['text']}")

    # Save different formats
    print("\nSaving outputs...")

    # JSON with speaker info
    with open("output_with_speakers.json", "w") as f:
        f.write(result.to_json())
    print("✓ Saved: output_with_speakers.json")

    # CSV with speaker column
    with open("output_with_speakers.csv", "w") as f:
        f.write(result.to_csv())
    print("✓ Saved: output_with_speakers.csv")

    # Table format
    with open("output_with_speakers.txt", "w") as f:
        f.write(result.to_table())
    print("✓ Saved: output_with_speakers.txt")


def demo_real_diarization():
    """Demo using real pyannote diarization."""
    print("\n=== Real Diarization Demo ===\n")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("To use real speaker diarization:")
        print("1. Create account at https://huggingface.co")
        print(
            "2. Accept conditions at https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
        print("3. Create token at https://huggingface.co/settings/tokens")
        print("4. Run: export HF_TOKEN='your-token-here'")
        return

    audio_file = "sample_podcast.wav"  # Change this to your audio file

    if not os.path.exists(audio_file):
        print("Please provide an audio file path.")
        return

    # Initialize transcriber
    transcriber = AudioTranscriber(model_name="tiny")

    # Transcribe with real diarization
    print("Transcribing with real speaker diarization...")
    print("This may take a while on first run as models are downloaded...")

    try:
        result = transcriber.transcribe_file(
            audio_file, enable_diarization=True, hf_token=hf_token
        )

        print("\n✅ Real diarization successful!")

        # Display results
        result_dict = result.to_dict()
        print(f"Detected {result_dict.get('num_speakers', 'Unknown')} speakers")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Falling back to mock mode...")
        demo_mock_diarization()


if __name__ == "__main__":
    print("Speaker Diarization Demo\n")

    # Check if we have a real audio file
    test_files = ["sample.wav", "podcast.wav", "audio.wav", "test.wav"]
    audio_file = None

    for f in test_files:
        if os.path.exists(f):
            audio_file = f
            break

    if audio_file:
        # Update the script to use this file
        with open(__file__, "r") as f:
            content = f.read()
        content = content.replace(
            'audio_file = "sample_podcast.wav"', f'audio_file = "{audio_file}"'
        )
        with open(__file__, "w") as f:
            f.write(content)
        print(f"Found audio file: {audio_file}")

    # Try real diarization first
    demo_real_diarization()

    # Also show mock mode
    if not os.getenv("HF_TOKEN"):
        demo_mock_diarization()
