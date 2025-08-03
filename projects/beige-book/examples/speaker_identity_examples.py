#!/usr/bin/env python3
"""
Examples of using the speaker identity tracking feature.

This file demonstrates various usage scenarios for identifying
and tracking speakers across podcast episodes.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from beige_book.transcriber import AudioTranscriber
from beige_book.database import TranscriptionDatabase
from beige_book.speaker_matcher import SpeakerMatcher
from beige_book.voice_embeddings import (
    VoiceEmbeddingExtractor,
    serialize_embedding,
    deserialize_embedding,
    cosine_similarity
)


def example_1_basic_speaker_identification():
    """Example 1: Basic speaker identification workflow."""
    print("\n=== Example 1: Basic Speaker Identification ===\n")
    
    # Initialize database
    db = TranscriptionDatabase("example_podcast.db")
    db.create_tables()
    db.create_speaker_identity_tables()
    
    # Transcribe with speaker identification
    transcriber = AudioTranscriber(model_name="tiny")
    result = transcriber.transcribe_file(
        "podcast_episode.mp3",
        enable_diarization=True,
        enable_speaker_identification=True,
        feed_url="https://example.com/podcast/feed.rss"
    )
    
    # Save transcription (automatically identifies speakers)
    trans_id = db.save_transcription(
        result,
        feed_url="https://example.com/podcast/feed.rss"
    )
    
    print(f"Transcription saved with ID: {trans_id}")
    print(f"Number of speakers: {result.num_speakers}")
    
    # Get identified speakers
    profiles = db.get_speaker_profiles_for_feed("https://example.com/podcast/feed.rss")
    for profile in profiles:
        print(f"\nSpeaker: {profile['display_name']}")
        print(f"  Appearances: {profile['total_appearances']}")
        print(f"  Total speaking time: {profile['total_duration']:.1f} seconds")


def example_2_preregister_known_speakers():
    """Example 2: Pre-register known speakers for better accuracy."""
    print("\n=== Example 2: Pre-registering Known Speakers ===\n")
    
    db = TranscriptionDatabase("example_podcast.db")
    feed_url = "https://example.com/podcast/feed.rss"
    
    # Register the podcast host
    host_id = db.create_speaker_profile(
        display_name="Sarah Johnson",
        feed_url=feed_url,
        canonical_label="HOST"
    )
    print(f"Created host profile with ID: {host_id}")
    
    # Register a regular co-host
    cohost_id = db.create_speaker_profile(
        display_name="Mike Chen",
        feed_url=feed_url,
        canonical_label="COHOST"
    )
    print(f"Created co-host profile with ID: {cohost_id}")
    
    # Add reference embeddings from intro clips
    extractor = VoiceEmbeddingExtractor(method="mock")  # Use real method in production
    
    # Extract host embedding from intro
    host_embedding, quality = extractor.extract_embedding_from_file(
        "host_intro.wav",
        start_time=0.0,
        end_time=10.0
    )
    
    db.add_speaker_embedding(
        profile_id=host_id,
        embedding=serialize_embedding(host_embedding),
        embedding_dimension=256,
        quality_score=quality,
        extraction_method="mock",
        audio_source="host_intro.wav"
    )
    
    print(f"\nAdded reference embedding for host (quality: {quality:.2f})")


def example_3_query_speaker_history():
    """Example 3: Query speaker appearance history."""
    print("\n=== Example 3: Querying Speaker History ===\n")
    
    db = TranscriptionDatabase("example_podcast.db")
    
    # Get all speakers for a feed
    profiles = db.get_speaker_profiles_for_feed("https://example.com/podcast/feed.rss")
    
    if not profiles:
        print("No speaker profiles found. Run example 1 or 2 first.")
        return
    
    # Get history for the first speaker
    speaker = profiles[0]
    print(f"\nHistory for speaker: {speaker['display_name']}")
    
    # Get appearances in the last 30 days
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    history = db.get_speaker_history(
        profile_id=speaker['id'],
        start_date=start_date,
        end_date=end_date
    )
    
    for appearance in history[:5]:  # Show first 5
        print(f"\n  Episode: {appearance['filename']}")
        print(f"  Date: {appearance['created_at']}")
        print(f"  Duration: {appearance['total_duration']:.1f}s")
        print(f"  Segments: {appearance['segment_count']}")


def example_4_search_speaker_statements():
    """Example 4: Search for specific statements by a speaker."""
    print("\n=== Example 4: Searching Speaker Statements ===\n")
    
    db = TranscriptionDatabase("example_podcast.db")
    
    # Get speakers
    profiles = db.get_speaker_profiles_for_feed("https://example.com/podcast/feed.rss")
    
    if not profiles:
        print("No speaker profiles found. Run example 1 or 2 first.")
        return
    
    speaker = profiles[0]
    
    # Get all statements containing specific text
    statements = db.get_speaker_statements(
        profile_id=speaker['id'],
        search_text="technology",  # Search for mentions of technology
        min_duration=2.0  # At least 2 seconds long
    )
    
    print(f"\nStatements by {speaker['display_name']} about 'technology':")
    for stmt in statements[:5]:
        print(f"\n  Date: {stmt['transcription_date']}")
        print(f"  Text: \"{stmt['text']}\"")
        print(f"  Duration: {stmt['duration']:.1f}s")


def example_5_manual_speaker_verification():
    """Example 5: Manually verify and correct speaker identification."""
    print("\n=== Example 5: Manual Speaker Verification ===\n")
    
    db = TranscriptionDatabase("example_podcast.db")
    matcher = SpeakerMatcher(db)
    
    # Simulate a case where we need to manually correct a speaker match
    trans_id = 1  # Assume we have a transcription
    
    # Get the host profile
    profiles = db.get_speaker_profiles_for_feed("https://example.com/podcast/feed.rss")
    host_profile = next((p for p in profiles if p['canonical_label'] == 'HOST'), None)
    
    if not host_profile:
        print("No host profile found. Run example 2 first.")
        return
    
    # Manually verify that SPEAKER_0 is the host
    occurrence_id = db.link_speaker_occurrence(
        transcription_id=trans_id,
        temporary_label="SPEAKER_0",
        profile_id=host_profile['id'],
        confidence=1.0,  # 100% confidence since manually verified
        is_verified=True
    )
    
    print(f"Manually verified SPEAKER_0 as {host_profile['display_name']}")
    print(f"Occurrence ID: {occurrence_id}")


def example_6_merge_duplicate_profiles():
    """Example 6: Merge duplicate speaker profiles."""
    print("\n=== Example 6: Merging Duplicate Profiles ===\n")
    
    db = TranscriptionDatabase("example_podcast.db")
    matcher = SpeakerMatcher(db)
    
    # Create duplicate profiles (simulating auto-detection creating duplicates)
    profile1 = db.create_speaker_profile("John Smith", feed_url="https://example.com/feed")
    profile2 = db.create_speaker_profile("John S.", feed_url="https://example.com/feed")
    
    print(f"Created potentially duplicate profiles:")
    print(f"  Profile 1: ID={profile1}, Name='John Smith'")
    print(f"  Profile 2: ID={profile2}, Name='John S.'")
    
    # Merge the profiles
    success = matcher.merge_speaker_profiles(
        profile_id_keep=profile1,
        profile_id_merge=profile2
    )
    
    if success:
        print(f"\nSuccessfully merged profile {profile2} into {profile1}")
        print("All embeddings and occurrences have been transferred.")
    else:
        print("\nMerge failed - check error logs")


def example_7_cross_episode_tracking():
    """Example 7: Track a speaker across multiple episodes."""
    print("\n=== Example 7: Cross-Episode Speaker Tracking ===\n")
    
    db = TranscriptionDatabase("example_podcast.db")
    transcriber = AudioTranscriber(model_name="tiny")
    feed_url = "https://example.com/podcast/feed.rss"
    
    # Process multiple episodes
    episodes = [
        "episode_001.mp3",
        "episode_002.mp3",
        "episode_003.mp3"
    ]
    
    for episode in episodes:
        print(f"\nProcessing {episode}...")
        
        result = transcriber.transcribe_file(
            episode,
            enable_diarization=True,
            enable_speaker_identification=True,
            feed_url=feed_url
        )
        
        trans_id = db.save_transcription(result, feed_url=feed_url)
        print(f"  Saved transcription ID: {trans_id}")
        print(f"  Speakers detected: {result.num_speakers}")
    
    # Analyze speaker appearances
    print("\n\nSpeaker Analysis Across Episodes:")
    profiles = db.get_speaker_profiles_for_feed(feed_url)
    
    for profile in profiles:
        print(f"\n{profile['display_name']}:")
        print(f"  Total episodes: {profile['total_appearances']}")
        print(f"  Total speaking time: {profile['total_duration'] / 60:.1f} minutes")
        print(f"  First seen: {profile['first_seen']}")
        print(f"  Last seen: {profile['last_seen']}")


def example_8_speaker_embeddings_analysis():
    """Example 8: Analyze speaker voice embeddings."""
    print("\n=== Example 8: Voice Embedding Analysis ===\n")
    
    db = TranscriptionDatabase("example_podcast.db")
    extractor = VoiceEmbeddingExtractor(method="mock")
    
    # Get a speaker profile
    profiles = db.get_speaker_profiles_for_feed("https://example.com/podcast/feed.rss")
    if not profiles:
        print("No profiles found. Run other examples first.")
        return
    
    speaker = profiles[0]
    
    # Get all embeddings for this speaker
    embeddings = db.get_speaker_embeddings(speaker['id'])
    print(f"\nAnalyzing embeddings for: {speaker['display_name']}")
    print(f"Total embeddings: {len(embeddings)}")
    
    if len(embeddings) >= 2:
        # Compare similarity between embeddings
        emb1_data = embeddings[0]
        emb2_data = embeddings[1]
        
        emb1 = deserialize_embedding(emb1_data['embedding'], emb1_data['embedding_dimension'])
        emb2 = deserialize_embedding(emb2_data['embedding'], emb2_data['embedding_dimension'])
        
        similarity = cosine_similarity(emb1, emb2)
        print(f"\nSimilarity between first two embeddings: {similarity:.3f}")
        print(f"Quality scores: {emb1_data['quality_score']:.2f}, {emb2_data['quality_score']:.2f}")
        print(f"Extraction methods: {emb1_data['extraction_method']}, {emb2_data['extraction_method']}")


def example_9_feed_speaker_report():
    """Example 9: Generate a comprehensive speaker report for a podcast feed."""
    print("\n=== Example 9: Podcast Speaker Report ===\n")
    
    db = TranscriptionDatabase("example_podcast.db")
    feed_url = "https://example.com/podcast/feed.rss"
    
    print(f"\nSPEAKER REPORT FOR PODCAST FEED")
    print(f"Feed: {feed_url}")
    print("=" * 60)
    
    # Get all speakers
    profiles = db.get_speaker_profiles_for_feed(feed_url)
    
    # Summary statistics
    total_speakers = len(profiles)
    hosts = [p for p in profiles if p['canonical_label'] == 'HOST']
    cohosts = [p for p in profiles if p['canonical_label'] == 'COHOST']
    guests = [p for p in profiles if p['canonical_label'] == 'GUEST']
    
    print(f"\nSPEAKER SUMMARY:")
    print(f"  Total unique speakers: {total_speakers}")
    print(f"  Hosts: {len(hosts)}")
    print(f"  Co-hosts: {len(cohosts)}")
    print(f"  Guests: {len(guests)}")
    print(f"  Other: {total_speakers - len(hosts) - len(cohosts) - len(guests)}")
    
    # Detailed speaker info
    print("\n\nDETAILED SPEAKER INFORMATION:")
    for profile in sorted(profiles, key=lambda x: x['total_duration'], reverse=True):
        print(f"\n{profile['display_name']} ({profile['canonical_label'] or 'Unknown Role'}):")
        print(f"  Episodes appeared: {profile['total_appearances']}")
        print(f"  Total speaking time: {profile['total_duration'] / 3600:.1f} hours")
        
        if profile['total_appearances'] > 0:
            avg_time = profile['total_duration'] / profile['total_appearances'] / 60
            print(f"  Average time per episode: {avg_time:.1f} minutes")
        
        print(f"  Active from: {profile['first_seen']} to {profile['last_seen']}")
        
        # Get recent statements
        recent = db.get_speaker_statements(profile['id'], limit=3)
        if recent:
            print(f"  Recent statements:")
            for stmt in recent:
                preview = stmt['text'][:80] + "..." if len(stmt['text']) > 80 else stmt['text']
                print(f"    - \"{preview}\"")


def main():
    """Run all examples."""
    examples = [
        example_1_basic_speaker_identification,
        example_2_preregister_known_speakers,
        example_3_query_speaker_history,
        example_4_search_speaker_statements,
        example_5_manual_speaker_verification,
        example_6_merge_duplicate_profiles,
        example_7_cross_episode_tracking,
        example_8_speaker_embeddings_analysis,
        example_9_feed_speaker_report
    ]
    
    print("Speaker Identity Tracking Examples")
    print("==================================")
    print("\nNote: These examples use mock data for demonstration.")
    print("In production, use real audio files and set SPEAKER_EMBEDDING_METHOD='speechbrain'")
    
    # Set mock mode for examples
    os.environ['SPEAKER_EMBEDDING_METHOD'] = 'mock'
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            print("(This is normal if required data hasn't been created yet)")
    
    print("\n\nAll examples completed!")


if __name__ == "__main__":
    main()