#!/usr/bin/env python3
"""
Example: Converting API responses (JSON/TOML) to database entries.

This demonstrates how to:
1. Receive JSON/TOML data from the API
2. Parse it back into TranscriptionResult objects
3. Save to SQLite database
"""

import json
import requests
from beige_book import TranscriptionResult, TranscriptionDatabase


def save_json_response_to_db():
    """Example: Save JSON API response to database"""
    print("=== JSON to Database Example ===\n")
    
    # Simulate receiving JSON from API
    # In real usage, this would come from:
    # response = requests.post("http://localhost:8000/transcribe", json=request_data)
    # json_data = response.text
    
    json_data = """
    {
        "filename": "podcast_episode_123.mp3",
        "file_hash": "abc123def456",
        "language": "en",
        "segments": [
            {
                "start": "00:00:00.000",
                "end": "00:00:05.500",
                "text": "Welcome to our podcast."
            },
            {
                "start": "00:00:05.500",
                "end": "00:00:10.200",
                "text": "Today we're discussing AI and transcription."
            }
        ],
        "full_text": "Welcome to our podcast. Today we're discussing AI and transcription."
    }
    """
    
    # Parse JSON back to TranscriptionResult
    result = TranscriptionResult.from_json(json_data)
    print(f"Parsed transcription for: {result.filename}")
    print(f"Language: {result.language}")
    print(f"Segments: {len(result.segments)}")
    
    # Save to database
    db = TranscriptionDatabase("api_transcriptions.db")
    db.create_tables()
    
    transcription_id = db.save_transcription(result, model_name="medium")
    print(f"\nSaved to database with ID: {transcription_id}")
    
    # Verify by retrieving
    retrieved = db.get_transcription(transcription_id)
    print(f"Retrieved from DB: {retrieved['metadata']['filename']}")


def save_toml_response_to_db():
    """Example: Save TOML API response to database"""
    print("\n=== TOML to Database Example ===\n")
    
    # Simulate receiving TOML from API
    toml_data = """
[transcription]
filename = "interview_2025.wav"
file_hash = "xyz789abc123"
language = "en"
full_text = "This is an interview about technology. It's very interesting."

[[segments]]
index = 0
start = "00:00:00.000"
end = "00:00:03.500"
duration = 3.5
text = "This is an interview about technology."

[[segments]]
index = 1
start = "00:00:03.500"
end = "00:00:06.000"
duration = 2.5
text = "It's very interesting."
"""
    
    # Parse TOML back to TranscriptionResult
    result = TranscriptionResult.from_toml(toml_data)
    print(f"Parsed transcription for: {result.filename}")
    print(f"Language: {result.language}")
    print(f"Segments: {len(result.segments)}")
    
    # Save to database
    db = TranscriptionDatabase("api_transcriptions.db")
    transcription_id = db.save_transcription(result, model_name="base")
    print(f"\nSaved to database with ID: {transcription_id}")


def full_api_workflow_example():
    """Example: Complete workflow from API call to database"""
    print("\n=== Full API Workflow Example ===\n")
    
    # Step 1: Make API request (example)
    print("Step 1: Make API request")
    api_request = {
        "input": {
            "type": "file",
            "source": "/path/to/audio.mp3"
        },
        "processing": {
            "model": "medium",
            "verbose": False
        },
        "output": {
            "format": "json"
        }
    }
    
    print(f"Request: {json.dumps(api_request, indent=2)}")
    
    # Step 2: Receive response (simulated)
    print("\nStep 2: Receive JSON response from API")
    # In real usage:
    # response = requests.post("http://localhost:8000/transcribe", json=api_request)
    # api_response = response.json()
    
    # Simulated response
    api_response = {
        "filename": "audio.mp3",
        "file_hash": "1234567890abcdef",
        "language": "en",
        "segments": [
            {"start": "00:00:00.000", "end": "00:00:02.500", "text": "Hello world."},
            {"start": "00:00:02.500", "end": "00:00:05.000", "text": "This is a test."}
        ],
        "full_text": "Hello world. This is a test."
    }
    
    # Step 3: Parse and save to database
    print("\nStep 3: Parse response and save to database")
    result = TranscriptionResult.from_dict(api_response)
    
    db = TranscriptionDatabase("api_transcriptions.db")
    db.create_tables()
    
    # Save with additional metadata if processing feeds
    transcription_id = db.save_transcription(
        result, 
        model_name="medium",
        feed_url="https://example.com/feed.xml",  # Optional feed metadata
        feed_item_id="episode-123",
        feed_item_title="Episode 123: AI Discussion"
    )
    
    print(f"Successfully saved transcription ID: {transcription_id}")
    
    # Step 4: Export back to different format if needed
    print("\nStep 4: Export from database to different format")
    exported = db.export_to_dict(transcription_id)
    if exported:
        print(f"Exported as TOML:\n{exported.to_toml()}")


def batch_processing_example():
    """Example: Process multiple API responses"""
    print("\n=== Batch Processing Example ===\n")
    
    # Simulate multiple API responses
    responses = [
        {
            "filename": "episode1.mp3",
            "file_hash": "hash1",
            "language": "en",
            "segments": [{"start": 0, "end": 5, "text": "Episode 1 content"}],
            "full_text": "Episode 1 content"
        },
        {
            "filename": "episode2.mp3",
            "file_hash": "hash2",
            "language": "en",
            "segments": [{"start": 0, "end": 5, "text": "Episode 2 content"}],
            "full_text": "Episode 2 content"
        }
    ]
    
    db = TranscriptionDatabase("api_transcriptions.db")
    db.create_tables()
    
    for response in responses:
        result = TranscriptionResult.from_dict(response)
        trans_id = db.save_transcription(result, model_name="tiny")
        print(f"Saved {result.filename} with ID: {trans_id}")
    
    # List all transcriptions
    print("\nAll transcriptions in database:")
    all_trans = db.list_transcriptions()
    for t in all_trans:
        print(f"  - {t['filename']} (ID: {t['id']}, Model: {t['model_name']})")


if __name__ == "__main__":
    print("API to Database Examples")
    print("=" * 50)
    
    # Run examples
    save_json_response_to_db()
    save_toml_response_to_db()
    full_api_workflow_example()
    batch_processing_example()
    
    print("\n\nSummary:")
    print("- Use TranscriptionResult.from_json() to parse JSON responses")
    print("- Use TranscriptionResult.from_toml() to parse TOML responses")
    print("- Use TranscriptionResult.from_dict() to parse dictionary data")
    print("- Then use TranscriptionDatabase.save_transcription() to save to SQLite")