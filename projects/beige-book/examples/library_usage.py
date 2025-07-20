#!/usr/bin/env python3
"""
Example of using the beige_book library directly.
"""

from beige_book import AudioTranscriber

# Create a transcriber with the small model
transcriber = AudioTranscriber(model_name="tiny")

# Transcribe an audio file
result = transcriber.transcribe_file("../../../resources/audio/harvard.wav")

# Access the raw data
print(f"File: {result.filename}")
print(f"Hash: {result.file_hash}")
print(f"Language: {result.language}")
print(f"Number of segments: {len(result.segments)}")
print()

# Print first segment details
first_segment = result.segments[0]
print("First segment:")
print(f"  Start: {first_segment.format_time(first_segment.start)}")
print(f"  End: {first_segment.format_time(first_segment.end)}")
print(f"  Text: {first_segment.text}")
print()

# Export to different formats
print("JSON output (first 200 chars):")
print(result.to_json()[:200] + "...")
print()

print("CSV output (first 3 lines):")
print('\n'.join(result.to_csv().split('\n')[:3]))
print()

# You can also save to files
with open("transcription.json", "w") as f:
    f.write(result.to_json())
print("Saved to transcription.json")