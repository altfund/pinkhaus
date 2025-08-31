#!/usr/bin/env python3
"""
Demo of voice matching functionality.

This example shows how to:
1. Build a database with speaker profiles from multiple audio files
2. Match a new audio file against the database to find known speakers
"""

import os
import tempfile
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinkhaus_models.database import TranscriptionDatabase
from beige_book.audio_processor import AudioProcessor
from beige_book.voice_matcher import find_voice_matches


def demo_voice_matching():
    """Demonstrate voice matching workflow."""
    
    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable is required")
        print("Please set: export HF_TOKEN='your-token-here'")
        return
        
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
        
    print(f"Creating temporary database: {db_path}")
    
    # Initialize database
    db = TranscriptionDatabase(db_path)
    db.create_tables()
    db.create_speaker_identity_tables()
    
    # Initialize audio processor
    processor = AudioProcessor(
        db=db,
        model_name="tiny",
        hf_token=hf_token,
        embedding_method="mock",  # Use mock for demo
        matcher_threshold=0.85
    )
    
    # Path to test audio
    harvard_path = Path(__file__).parent.parent.parent.parent / "resources" / "audio" / "harvard.wav"
    
    if not harvard_path.exists():
        print(f"Test audio not found: {harvard_path}")
        return
        
    print("\n=== Step 1: Process initial audio file ===")
    print(f"Processing: {harvard_path.name}")
    
    # Process first audio file to build speaker profiles
    result1 = processor.process_audio_file(
        audio_path=str(harvard_path),
        feed_url="https://example.com/podcast1.rss",
        enable_diarization=True,
        create_new_profiles=True,
        verbose=True
    )
    
    print(f"\nCreated {len(result1['speaker_profiles'])} speaker profiles")
    
    # Simulate processing another episode from same podcast
    print("\n=== Step 2: Process second file (simulating same speaker) ===")
    
    # For demo, we'll process the same file but pretend it's a different episode
    result2 = processor.process_audio_file(
        audio_path=str(harvard_path),
        feed_url="https://example.com/podcast1.rss",
        enable_diarization=True,
        create_new_profiles=True,
        verbose=True
    )
    
    print(f"\nMatched {sum(1 for m in result2['matches'].values() if not m['is_new'])} existing speakers")
    
    print("\n=== Step 3: Test voice matching on 'new' audio ===")
    print("Now testing if we can identify the speaker in a 'new' audio file...")
    
    # Use voice matcher to check if speaker exists in database
    matches = find_voice_matches(
        db=db,
        audio_path=str(harvard_path),
        hf_token=hf_token,
        embedding_method="mock",
        threshold=0.85,
        verbose=True
    )
    
    print("\n=== Results ===")
    
    if matches["summary"]["matched_speakers"] > 0:
        print(f"\n✓ Successfully identified {matches['summary']['matched_speakers']} speaker(s)!")
        
        for speaker_label, speaker_data in matches["speakers"].items():
            if speaker_data["profile_id"]:
                print(f"\n{speaker_label}:")
                print(f"  Matched to: {speaker_data['display_name']}")
                print(f"  Confidence: {speaker_data['confidence']:.3f}")
                print(f"  Previously heard in:")
                
                for match in speaker_data["matches"][:3]:  # Show first 3
                    print(f"    - {match['filename']}")
                    if match["feed_url"]:
                        print(f"      Feed: {match['feed_url']}")
    else:
        print("\nNo matching speakers found in database")
        
    # Cleanup
    print(f"\n\nDemo complete. Database saved at: {db_path}")
    print("(Delete when done testing)")
    
    return db_path


if __name__ == "__main__":
    db_path = demo_voice_matching()
    
    if db_path:
        print("\n\nTo test the CLI tool with your own audio:")
        print(f"  beige-book-match-voice /path/to/your/audio.wav {db_path}")
        print("\nOptions:")
        print("  --format detailed     # Show all appearances")
        print("  --format json        # Get JSON output")
        print("  --threshold 0.75     # Lower matching threshold")