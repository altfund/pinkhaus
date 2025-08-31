#!/usr/bin/env python3

"""
Simple test for speaker identification across recordings.
This demonstrates the core functionality of identifying the same speaker.
"""

import os
import tempfile
from pathlib import Path

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinkhaus_models.database import TranscriptionDatabase
from beige_book.transcriber import AudioTranscriber, TranscriptionResult, Segment
from beige_book.speaker_matcher import SpeakerMatcher
from beige_book.voice_embeddings import VoiceEmbeddingExtractor, serialize_embedding
import numpy as np


def simulate_speaker_identification():
    """Demonstrate speaker identification across recordings."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_speaker_id.db")
        db = TranscriptionDatabase(db_path)
        db.create_tables()
        db.create_speaker_identity_tables()
        
        feed_url = "https://test-podcast.com/feed.rss"
        
        print("\n=== SPEAKER IDENTIFICATION DEMO ===\n")
        
        # Initialize components
        extractor = VoiceEmbeddingExtractor(method="mock")
        matcher = SpeakerMatcher(db, threshold=0.85, embedding_method="mock")
        
        # Episode 1: Create initial speaker profiles
        print("1. Episode 1: First appearance of speakers")
        
        # Create host profile
        host_profile_id = db.create_speaker_profile(
            "John Doe (Host)", 
            feed_url=feed_url, 
            canonical_label="HOST"
        )
        
        # Generate and store host embedding
        host_embedding = np.random.randn(256).astype(np.float32)
        host_embedding = host_embedding / np.linalg.norm(host_embedding)
        db.add_speaker_embedding(
            host_profile_id, 
            serialize_embedding(host_embedding), 
            256, 
            quality_score=0.95
        )
        
        print(f"  Created host profile: ID={host_profile_id}")
        
        # Create guest profile for episode 1
        guest1_profile_id = db.create_speaker_profile(
            "Guest (Episode 1)",
            feed_url=feed_url,
            canonical_label="GUEST"
        )
        
        # Generate different embedding for guest
        guest1_embedding = np.random.randn(256).astype(np.float32)
        guest1_embedding = guest1_embedding / np.linalg.norm(guest1_embedding)
        db.add_speaker_embedding(
            guest1_profile_id,
            serialize_embedding(guest1_embedding),
            256,
            quality_score=0.90
        )
        
        print(f"  Created guest profile: ID={guest1_profile_id}")
        
        # Episode 2: Test speaker recognition
        print("\n2. Episode 2: Testing speaker recognition")
        
        # Simulate extracting embeddings from episode 2
        # The host should have a similar embedding (with small variations)
        host_embedding_ep2 = host_embedding + np.random.randn(256) * 0.1
        host_embedding_ep2 = host_embedding_ep2 / np.linalg.norm(host_embedding_ep2)
        
        # New guest has different embedding
        guest2_embedding = np.random.randn(256).astype(np.float32)
        guest2_embedding = guest2_embedding / np.linalg.norm(guest2_embedding)
        
        # Match speakers
        print("\n  Matching SPEAKER_0 (should be host):")
        host_matches = matcher.find_best_match(host_embedding_ep2, feed_url=feed_url)
        if host_matches:
            best_match = host_matches[0]
            print(f"    Best match: {best_match[2]['display_name']} (confidence: {best_match[1]:.3f})")
            if best_match[1] >= matcher.threshold:
                print("    ✓ Successfully identified as existing host!")
            
        print("\n  Matching SPEAKER_1 (new guest):")
        guest_matches = matcher.find_best_match(guest2_embedding, feed_url=feed_url)
        if guest_matches:
            best_match = guest_matches[0]
            print(f"    Best match: {best_match[2]['display_name']} (confidence: {best_match[1]:.3f})")
            if best_match[1] < matcher.threshold:
                print("    ✓ Correctly identified as new speaker")
                # Create new profile
                new_guest_id = db.create_speaker_profile(
                    "Guest (Episode 2)",
                    feed_url=feed_url,
                    canonical_label="GUEST"
                )
                db.add_speaker_embedding(
                    new_guest_id,
                    serialize_embedding(guest2_embedding),
                    256,
                    quality_score=0.88
                )
        
        # Episode 3: Verify consistency
        print("\n3. Episode 3: Verifying speaker tracking")
        
        # Host appears again with slight variation
        host_embedding_ep3 = host_embedding + np.random.randn(256) * 0.15
        host_embedding_ep3 = host_embedding_ep3 / np.linalg.norm(host_embedding_ep3)
        
        # Previous guest from episode 1 returns
        guest1_embedding_ep3 = guest1_embedding + np.random.randn(256) * 0.1
        guest1_embedding_ep3 = guest1_embedding_ep3 / np.linalg.norm(guest1_embedding_ep3)
        
        print("\n  Matching SPEAKER_0 (host again):")
        host_matches = matcher.find_best_match(host_embedding_ep3, feed_url=feed_url)
        if host_matches:
            best_match = host_matches[0]
            print(f"    Best match: {best_match[2]['display_name']} (confidence: {best_match[1]:.3f})")
            
        print("\n  Matching SPEAKER_1 (returning guest from episode 1):")
        guest_matches = matcher.find_best_match(guest1_embedding_ep3, feed_url=feed_url)
        if guest_matches:
            best_match = guest_matches[0]
            print(f"    Best match: {best_match[2]['display_name']} (confidence: {best_match[1]:.3f})")
            if best_match[0] == guest1_profile_id:
                print("    ✓ Successfully recognized returning guest!")
        
        # Show final statistics
        print("\n=== FINAL SPEAKER PROFILES ===\n")
        
        profiles = db.get_speaker_profiles_for_feed(feed_url)
        for profile in profiles:
            print(f"Profile: {profile['display_name']}")
            print(f"  ID: {profile['id']}")
            print(f"  Role: {profile['canonical_label']}")
            embeddings = db.get_speaker_embeddings(profile['id'])
            print(f"  Embeddings stored: {len(embeddings)}")
            print()
        
        # Test merging duplicate profiles
        print("=== TESTING PROFILE MERGE ===\n")
        
        # Create a duplicate host profile (as if mistakenly created)
        duplicate_host_id = db.create_speaker_profile(
            "John D. (Host)",  # Slightly different name
            feed_url=feed_url,
            canonical_label="HOST"
        )
        
        # Add an embedding that's very similar to original host
        duplicate_embedding = host_embedding + np.random.randn(256) * 0.05
        duplicate_embedding = duplicate_embedding / np.linalg.norm(duplicate_embedding)
        db.add_speaker_embedding(
            duplicate_host_id,
            serialize_embedding(duplicate_embedding),
            256,
            quality_score=0.92
        )
        
        print(f"Created duplicate host profile: ID={duplicate_host_id}")
        
        # Merge the profiles
        print(f"Merging profile {duplicate_host_id} into {host_profile_id}")
        matcher.merge_speaker_profiles(duplicate_host_id, host_profile_id)
        
        # Verify merge
        profiles_after = db.get_speaker_profiles_for_feed(feed_url)
        print(f"\nProfiles after merge: {len(profiles_after)}")
        
        # Check embeddings were transferred
        host_embeddings_after = db.get_speaker_embeddings(host_profile_id)
        print(f"Host embeddings after merge: {len(host_embeddings_after)}")
        
        print("\n✅ Speaker identification demo complete!")


if __name__ == "__main__":
    simulate_speaker_identification()