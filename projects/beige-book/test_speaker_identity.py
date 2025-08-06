#!/usr/bin/env python3
"""
Test/demo script for speaker identity tracking across recordings.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from pinkhaus_models.database import TranscriptionDatabase
from beige_book.transcriber import AudioTranscriber
from beige_book.voice_embeddings import VoiceEmbeddingExtractor, serialize_embedding
from beige_book.speaker_matcher import SpeakerMatcher


def demonstrate_speaker_identity():
    """Demonstrate speaker identity tracking."""

    print("=== Speaker Identity Tracking Demo ===\n")

    # Create a test database
    with tempfile.NamedTemporaryFile(
        suffix="_speaker_identity.db", delete=False
    ) as tmp:
        db_path = tmp.name

    print(f"1. Creating database: {db_path}")
    db = TranscriptionDatabase(db_path)
    db.create_tables()
    db.create_speaker_identity_tables()
    print("   ✓ Database tables created\n")

    # Simulate a podcast feed
    feed_url = "https://example.com/podcast/feed.rss"

    # Create some speaker profiles manually
    print("2. Creating speaker profiles for known hosts:")
    host_id = db.create_speaker_profile(
        display_name="John Doe (Host)", feed_url=feed_url, canonical_label="HOST"
    )
    print(f"   ✓ Created profile: John Doe (ID: {host_id})")

    cohost_id = db.create_speaker_profile(
        display_name="Jane Smith (Co-host)", feed_url=feed_url, canonical_label="COHOST"
    )
    print(f"   ✓ Created profile: Jane Smith (ID: {cohost_id})\n")

    # Simulate embeddings for these speakers (in real usage, extract from reference audio)
    print("3. Adding voice embeddings for known speakers:")
    extractor = VoiceEmbeddingExtractor(method="mock")

    # Add embeddings for host
    host_embedding, _ = extractor.extract_embedding_from_file("dummy.wav")  # Mock
    db.add_speaker_embedding(
        profile_id=host_id,
        embedding=serialize_embedding(host_embedding),
        embedding_dimension=256,
        quality_score=0.95,
        extraction_method="mock",
    )
    print("   ✓ Added embedding for John Doe")

    # Add embeddings for co-host
    cohost_embedding, _ = extractor.extract_embedding_from_file("dummy.wav")  # Mock
    db.add_speaker_embedding(
        profile_id=cohost_id,
        embedding=serialize_embedding(cohost_embedding),
        embedding_dimension=256,
        quality_score=0.95,
        extraction_method="mock",
    )
    print("   ✓ Added embedding for Jane Smith\n")

    # Simulate transcribing a new episode
    print("4. Simulating new episode transcription:")

    # Check for test audio
    test_audio = "../../../resources/audio/harvard.wav"
    if not os.path.exists(test_audio):
        print(f"   ⚠️  Test audio not found at {test_audio}")
        print("   Using mock transcription instead\n")

        # Create mock transcription with speakers
        from beige_book.transcriber import TranscriptionResult

        result = TranscriptionResult()
        result.filename = "episode_001.mp3"
        result.file_hash = "abc123"
        result.language = "en"
        result.full_text = "Welcome to our show. Thanks for having me."
        result._proto.num_speakers = 2
        result._proto.has_speaker_labels = True

        # Add segments
        result.add_segment(0, 3, "Welcome to our show.")
        result.segments[0].speaker = "SPEAKER_0"
        result.segments[0].confidence = 0.95

        result.add_segment(3, 6, "Thanks for having me.")
        result.segments[1].speaker = "SPEAKER_1"
        result.segments[1].confidence = 0.92

        # Add mock embeddings
        result._speaker_embeddings = {
            "SPEAKER_0": (host_embedding, 3.0, [0]),  # Will match host
            "SPEAKER_1": (cohost_embedding, 3.0, [1]),  # Will match co-host
        }
        result._feed_url = feed_url
    else:
        print(f"   Transcribing: {test_audio}")
        transcriber = AudioTranscriber(model_name="tiny")
        result = transcriber.transcribe_file(
            test_audio,
            enable_diarization=True,
            enable_speaker_identification=True,
            feed_url=feed_url,
        )

    print(f"   ✓ Transcription complete: {len(result.segments)} segments\n")

    # Save to database (this triggers speaker identification)
    print("5. Saving transcription with speaker identification:")
    trans_id = db.save_transcription(result, model_name="tiny", feed_url=feed_url)
    print(f"   ✓ Saved transcription ID: {trans_id}\n")

    # Check speaker identification results
    print("6. Checking speaker identification results:")

    # Get speaker occurrences
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT so.*, sp.display_name
            FROM speaker_occurrences so
            JOIN speaker_profiles sp ON so.profile_id = sp.id
            WHERE so.transcription_id = ?
        """,
            (trans_id,),
        )

        occurrences = cursor.fetchall()
        for occ in occurrences:
            print(
                f"   - {occ['temporary_label']} → {occ['display_name']} "
                f"(confidence: {occ['confidence']:.2f})"
            )

    print("\n7. Querying speaker history:")

    # Get all statements by the host
    host_statements = db.get_speaker_statements(host_id)
    print(f"   Host statements: {len(host_statements)}")
    for stmt in host_statements[:3]:
        print(f'     - "{stmt["text"][:50]}..."')

    # Get speaker statistics
    print("\n8. Speaker statistics for the feed:")
    profiles = db.get_speaker_profiles_for_feed(feed_url)
    for profile in profiles:
        print(
            f"   - {profile['display_name']}: "
            f"{profile['total_appearances']} appearances, "
            f"{profile['total_duration_seconds']:.1f}s total"
        )

    # Simulate matching a new speaker
    print("\n9. Testing unknown speaker handling:")
    matcher = SpeakerMatcher(db, threshold=0.85, embedding_method="mock")

    # Create a new embedding that won't match
    import numpy as np

    unknown_embedding = np.random.randn(256).astype(np.float32)

    profile_id, confidence = matcher.match_speaker(
        unknown_embedding, feed_url=feed_url, speaker_hint="Episode 1 Guest"
    )

    if confidence < 0.85:
        print(f"   ✓ Created new profile for unknown speaker (ID: {profile_id})")
    else:
        print(f"   Matched to existing profile (ID: {profile_id})")

    # Clean up
    print(f"\n10. Database saved at: {db_path}")
    print("    (Delete manually if not needed)")

    print("\n=== Demo Complete ===")
    print("\nKey Features Demonstrated:")
    print("- Speaker profiles with canonical labels (HOST, COHOST)")
    print("- Voice embeddings for speaker recognition")
    print("- Automatic speaker identification during transcription")
    print("- Linking temporary labels to persistent profiles")
    print("- Querying speaker history and statements")
    print("- Handling unknown speakers with new profile creation")


if __name__ == "__main__":
    demonstrate_speaker_identity()
