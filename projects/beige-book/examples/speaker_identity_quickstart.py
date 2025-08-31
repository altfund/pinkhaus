#!/usr/bin/env python3
"""
Quick start guide for speaker identity tracking.

This example shows the most common use case: transcribing a podcast
and automatically identifying recurring speakers.
"""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from beige_book.transcriber import AudioTranscriber
from pinkhaus_models.database import TranscriptionDatabase


def quickstart_speaker_tracking():
    """
    Minimal example to get started with speaker identity tracking.
    """
    # 1. Set up the database
    db = TranscriptionDatabase("my_podcast.db")
    db.create_tables()
    db.create_speaker_identity_tables()
    print("✓ Database initialized")

    # 2. Configure the transcriber
    transcriber = AudioTranscriber(model_name="tiny")

    # 3. Transcribe with speaker identification
    # Note: You need HF_TOKEN environment variable for real diarization
    result = transcriber.transcribe_file(
        "podcast_episode.mp3",
        enable_diarization=True,  # Enable speaker diarization
        enable_speaker_identification=True,  # Enable identity tracking
        feed_url="https://mypodcast.com/feed.rss",  # Scope speakers to this feed
    )

    print(f"✓ Transcription complete: {result.num_speakers} speakers detected")

    # 4. Save to database (automatically identifies speakers)
    trans_id = db.save_transcription(result, feed_url="https://mypodcast.com/feed.rss")
    print(f"✓ Saved to database with ID: {trans_id}")

    # 5. View the identified speakers
    print("\n📊 Speaker Summary:")
    profiles = db.get_speaker_profiles_for_feed("https://mypodcast.com/feed.rss")

    for profile in profiles:
        print(f"\n👤 {profile['display_name']}")
        print(f"   Episodes: {profile['total_appearances']}")
        print(f"   Speaking time: {profile['total_duration'] / 60:.1f} minutes")

        # Show sample statements
        statements = db.get_speaker_statements(profile["id"], limit=3)
        if statements:
            print("   Sample quotes:")
            for stmt in statements:
                quote = (
                    stmt["text"][:60] + "..."
                    if len(stmt["text"]) > 60
                    else stmt["text"]
                )
                print(f'   - "{quote}"')


def advanced_example_with_known_host():
    """
    Example showing how to pre-register known speakers for better accuracy.
    """
    from beige_book.voice_embeddings import VoiceEmbeddingExtractor, serialize_embedding

    db = TranscriptionDatabase("my_podcast.db")
    feed_url = "https://mypodcast.com/feed.rss"

    # Pre-register the host
    host_id = db.create_speaker_profile(
        display_name="Jane Smith", feed_url=feed_url, canonical_label="HOST"
    )

    # Optional: Add a reference voice sample
    if os.path.exists("host_intro.wav"):
        extractor = VoiceEmbeddingExtractor()
        embedding, quality = extractor.extract_embedding_from_file("host_intro.wav")

        db.add_speaker_embedding(
            profile_id=host_id,
            embedding=serialize_embedding(embedding),
            embedding_dimension=256,
            quality_score=quality,
        )
        print(f"✓ Added voice reference for host (quality: {quality:.2f})")

    # Now when you transcribe, the host will be automatically recognized
    print("✓ Host profile created - will be recognized in future episodes")


def batch_processing_example():
    """
    Example of processing multiple episodes and tracking speakers across them.
    """
    db = TranscriptionDatabase("my_podcast.db")
    transcriber = AudioTranscriber(model_name="tiny")
    feed_url = "https://mypodcast.com/feed.rss"

    # Process a batch of episodes
    episode_files = [
        "episode_001.mp3",
        "episode_002.mp3",
        "episode_003.mp3",
    ]

    for episode in episode_files:
        if not os.path.exists(episode):
            print(f"⚠️  Skipping {episode} (file not found)")
            continue

        print(f"\n📼 Processing {episode}...")

        result = transcriber.transcribe_file(
            episode,
            enable_diarization=True,
            enable_speaker_identification=True,
            feed_url=feed_url,
        )

        db.save_transcription(result, feed_url=feed_url)
        print(f"   ✓ {result.num_speakers} speakers identified")

    # Generate a speaker report
    print("\n📊 SPEAKER REPORT ACROSS ALL EPISODES:")
    profiles = db.get_speaker_profiles_for_feed(feed_url)

    for profile in sorted(profiles, key=lambda x: x["total_duration"], reverse=True):
        pct = (
            profile["total_duration"] / sum(p["total_duration"] for p in profiles)
        ) * 100
        print(f"\n👤 {profile['display_name']}:")
        print(f"   Episodes: {profile['total_appearances']}")
        print(
            f"   Speaking time: {profile['total_duration'] / 60:.1f} min ({pct:.1f}%)"
        )


if __name__ == "__main__":
    print("🎙️  Speaker Identity Tracking - Quick Start Examples")
    print("=" * 50)

    # Use mock embeddings for demo (no GPU required)
    os.environ["SPEAKER_EMBEDDING_METHOD"] = "mock"

    print("\n1️⃣  Basic Speaker Tracking:")
    print("-" * 30)
    try:
        quickstart_speaker_tracking()
    except FileNotFoundError:
        print("⚠️  podcast_episode.mp3 not found - using mock data")

    print("\n\n2️⃣  Pre-registering Known Speakers:")
    print("-" * 30)
    advanced_example_with_known_host()

    print("\n\n3️⃣  Batch Processing Multiple Episodes:")
    print("-" * 30)
    batch_processing_example()

    print("\n\n✅ Examples complete!")
    print("\n💡 Tips:")
    print("- Set HF_TOKEN environment variable for real speaker diarization")
    print("- Use SPEAKER_EMBEDDING_METHOD='speechbrain' for production")
    print("- Pre-register known speakers for better accuracy")
    print("- Process episodes in order for best cross-episode tracking")
