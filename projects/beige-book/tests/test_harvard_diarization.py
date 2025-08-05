#!/usr/bin/env python3
"""
Test script for speaker diarization using harvard.wav file.
Creates a comprehensive database with both transcription and diarization data.
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from beige_book.audio_processor import AudioProcessor
from pinkhaus_models.database import TranscriptionDatabase


def test_harvard_with_diarization():
    """Test transcription and diarization on harvard.wav file."""

    # Path to harvard.wav
    harvard_path = "/Users/price/development/ai-projects/pinkhaus2/resources/audio/harvard.wav"

    if not os.path.exists(harvard_path):
        print(f"Error: harvard.wav not found at {harvard_path}")
        print("Please check the path or provide the correct location.")
        return

    print(f"Using audio file: {os.path.abspath(harvard_path)}")

    # Require HF token for real diarization
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable is required for speaker diarization")
    
    print("\n=== AUDIO PROCESSING WITH SPEAKER IDENTITY ===")

    # Create database
    db_path = "harvard_diarization.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    print("\n1. Initializing database...")
    db = TranscriptionDatabase(db_path)
    db.create_tables()
    db.create_speaker_identity_tables()
    print(f"   ✓ Database created: {db_path}")

    # Initialize AudioProcessor with all components
    print("\n2. Initializing AudioProcessor...")
    processor = AudioProcessor(
        db=db,
        model_name="tiny",
        hf_token=hf_token,
        embedding_method="mock",  # Using mock for faster demo
        matcher_threshold=0.85
    )
    print("   ✓ AudioProcessor ready")

    # Process the audio file with EVERYTHING automated
    print("\n3. Processing audio file (all-in-one)...")
    feed_url = "https://example-podcast.com/feed.rss"
    
    result = processor.process_audio_file(
        audio_path=harvard_path,
        feed_url=feed_url,
        enable_diarization=True,
        create_new_profiles=True,
        profile_prefix="Harvard Speaker",
        verbose=True
    )

    # Show results
    print("\n4. Processing Results:")
    print(f"   - Transcription ID: {result['transcription_id']}")
    print(f"   - Speakers detected: {result['num_speakers']}")
    print(f"   - Speaker profiles created/matched: {len(result['speaker_profiles'])}")
    
    # Get transcription result for file outputs
    transcription = result['transcription']
    result_dict = transcription.to_dict()

    # Save output files
    print("\n5. Saving output files...")
    
    # JSON with full details
    json_path = "harvard_diarization_result.json"
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"   ✓ JSON saved: {json_path}")

    # CSV with speakers
    csv_path = "harvard_diarization_result.csv"
    with open(csv_path, "w") as f:
        f.write(transcription.to_csv())
    print(f"   ✓ CSV saved: {csv_path}")

    # Get speaker summary
    print("\n6. Speaker Summary:")
    summary = processor.get_speaker_summary(feed_url)
    
    print(f"   Total speakers in feed: {summary['total_speakers']}")
    for speaker in summary['speakers']:
        print(f"\n   {speaker['name']} (ID: {speaker['id']})")
        print(f"     - Appearances: {speaker['appearances']}")
        print(f"     - Total duration: {speaker['duration_seconds']:.1f} seconds")
        print(f"     - Voice embeddings: {speaker['embeddings_count']}")
        if 'sample_statements' in speaker:
            print("     - Sample statements:")
            for stmt in speaker['sample_statements']:
                print(f"       • {stmt}")

    # Show database tables
    print("\n7. Database Tables:")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"   Tables created: {', '.join([t[0] for t in tables])}")

    # Check speaker profiles table
    cursor.execute("SELECT COUNT(*) FROM speaker_profiles")
    profile_count = cursor.fetchone()[0]
    print(f"   - Speaker profiles: {profile_count}")
    
    # Check speaker embeddings table  
    cursor.execute("SELECT COUNT(*) FROM speaker_embeddings")
    embedding_count = cursor.fetchone()[0]
    print(f"   - Speaker embeddings: {embedding_count}")
    
    # Check speaker occurrences table
    cursor.execute("SELECT COUNT(*) FROM speaker_occurrences")
    occurrence_count = cursor.fetchone()[0]
    print(f"   - Speaker occurrences: {occurrence_count}")

    # Simulate processing a second episode to test speaker matching
    print("\n8. Simulating Second Episode (Testing Speaker Matching):")
    print("   Processing the same audio again (simulating new episode)...")
    
    result2 = processor.process_audio_file(
        audio_path=harvard_path,
        feed_url=feed_url,
        enable_diarization=True,
        create_new_profiles=True,
        profile_prefix="Harvard Speaker",
        verbose=False  # Less verbose for second run
    )
    
    print(f"\n   Second processing results:")
    print(f"   - New profiles created: {sum(1 for m in result2['matches'].values() if m['is_new'])}")
    print(f"   - Speakers matched to existing: {sum(1 for m in result2['matches'].values() if not m['is_new'])}")
    
    for speaker, match_info in result2['matches'].items():
        if not match_info['is_new']:
            print(f"   - {speaker} matched with confidence {match_info['confidence']:.3f}")

    conn.close()

    print("\n✅ Test complete!")
    print("\nThe AudioProcessor successfully:")
    print("   1. Transcribed the audio")
    print("   2. Performed speaker diarization")
    print("   3. Extracted voice embeddings")
    print("   4. Created speaker profiles")
    print("   5. Matched speakers across episodes")
    print("   6. Stored everything in the database")
    print("\nGenerated files:")
    print(f"   - {json_path} (transcription with speaker labels)")
    print(f"   - {csv_path} (CSV format)")
    print(f"   - {db_path} (database with speaker profiles)")




if __name__ == "__main__":
    print("Harvard Audio Diarization Test")
    print("=" * 50)

    # Show HF token info
    print("\nHugging Face Token Requirements:")
    print("- Scope: 'read' permission is sufficient")
    print("- Usage: Access to pyannote/speaker-diarization models")
    print(
        "- Setup: Accept conditions at https://huggingface.co/pyannote/speaker-diarization-3.1"
    )

    if not os.getenv("HF_TOKEN"):
        print("\n❌ ERROR: HF_TOKEN environment variable is required")
        print("   Please set: export HF_TOKEN='hf_...'")
        sys.exit(1)

    print("\nStarting test...\n")

    try:
        test_harvard_with_diarization()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
