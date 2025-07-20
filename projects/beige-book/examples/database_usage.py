#!/usr/bin/env python3
"""
Example of using the database functionality to store and retrieve transcriptions.
"""

from beige_book import AudioTranscriber, TranscriptionDatabase
from pathlib import Path

# Create database instance
db = TranscriptionDatabase("transcriptions.db")

# Ensure tables exist
db.create_tables()

# Transcribe a file
transcriber = AudioTranscriber(model_name="tiny")
audio_file = Path(__file__).parent.parent.parent / "resources" / "audio" / "harvard.wav"
result = transcriber.transcribe_file(str(audio_file))

# Save to database
print("Saving transcription to database...")
transcription_id = db.save_transcription(result, model_name="tiny")
print(f"Saved with ID: {transcription_id}")
print()

# Retrieve the transcription
print("Retrieving transcription from database...")
data = db.get_transcription(transcription_id)
print(f"Filename: {data['metadata']['filename']}")
print(f"Language: {data['metadata']['language']}")
print(f"Number of segments: {len(data['segments'])}")
print()

# Find by file hash
print("Finding all transcriptions for this file...")
transcriptions = db.find_by_hash(result.file_hash)
for trans in transcriptions:
    print(f"  ID: {trans['id']}, Model: {trans['model_name']}, Created: {trans['created_at']}")
print()

# Get recent transcriptions
print("Recent transcriptions:")
recent = db.get_recent_transcriptions(limit=5)
for trans in recent:
    print(f"  {trans['filename']} - {trans['created_at']}")
print()

# Export back to TranscriptionResult object
print("Exporting to TranscriptionResult object...")
exported = db.export_to_dict(transcription_id)
print(f"Exported filename: {exported.filename}")
print(f"First segment: {exported.segments[0].text}")

# You can also use custom table names
print("\nUsing custom table names...")
db.create_tables("my_metadata", "my_segments")
custom_id = db.save_transcription(
    result, 
    model_name="base",
    metadata_table="my_metadata",
    segments_table="my_segments"
)
print(f"Saved to custom tables with ID: {custom_id}")