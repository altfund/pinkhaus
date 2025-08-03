#!/usr/bin/env python3
"""
Tests for speaker identity tracking system.
"""

import os
import tempfile
import unittest
import numpy as np
from pathlib import Path

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from beige_book.database import TranscriptionDatabase
from beige_book.voice_embeddings import (
    VoiceEmbeddingExtractor,
    cosine_similarity,
    serialize_embedding,
    deserialize_embedding,
)
from beige_book.speaker_matcher import SpeakerMatcher
from beige_book.transcriber import TranscriptionResult


class TestVoiceEmbeddings(unittest.TestCase):
    """Test voice embedding extraction and utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = VoiceEmbeddingExtractor(method="mock")
    
    def test_extract_embedding_from_file(self):
        """Test embedding extraction from audio file."""
        # Mock extraction
        embedding, quality = self.extractor.extract_embedding_from_file("dummy.wav")
        
        # Check embedding properties
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (256,))
        self.assertEqual(embedding.dtype, np.float32)
        
        # Check normalization
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)
        
        # Check quality score
        self.assertIsInstance(quality, float)
        self.assertGreaterEqual(quality, 0.0)
        self.assertLessEqual(quality, 1.0)
    
    def test_extract_embeddings_for_speaker(self):
        """Test embedding extraction from multiple segments."""
        segments = [
            {'start_time': 0.0, 'end_time': 3.0, 'text': 'Hello world'},
            {'start_time': 5.0, 'end_time': 8.0, 'text': 'How are you'},
            {'start_time': 10.0, 'end_time': 11.0, 'text': 'Great'},  # Long enough
        ]
        
        embedding, duration, indices = self.extractor.extract_embeddings_for_speaker(
            "dummy.wav", segments, min_duration=3.0
        )
        
        # Check results
        self.assertIsNotNone(embedding)
        self.assertEqual(duration, 7.0)  # 3 + 3 + 1 seconds
        self.assertEqual(indices, [0, 1, 2])
    
    def test_embedding_serialization(self):
        """Test embedding serialization/deserialization."""
        # Create test embedding
        original = np.random.randn(256).astype(np.float32)
        
        # Serialize and deserialize
        serialized = serialize_embedding(original)
        deserialized = deserialize_embedding(serialized, 256)
        
        # Check equality
        np.testing.assert_array_almost_equal(original, deserialized)
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Test identical embeddings
        emb1 = np.random.randn(256).astype(np.float32)
        similarity = cosine_similarity(emb1, emb1)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        # Test orthogonal embeddings
        emb2 = np.zeros(256, dtype=np.float32)
        emb2[0] = 1.0
        emb3 = np.zeros(256, dtype=np.float32)
        emb3[1] = 1.0
        similarity = cosine_similarity(emb2, emb3)
        self.assertAlmostEqual(similarity, 0.5, places=5)  # Normalized to 0-1 range


class TestSpeakerDatabase(unittest.TestCase):
    """Test database operations for speaker identity."""
    
    def setUp(self):
        """Create test database."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_file.name
        self.db = TranscriptionDatabase(self.db_path)
        self.db.create_tables()
        self.db.create_speaker_identity_tables()
    
    def tearDown(self):
        """Clean up test database."""
        self.temp_file.close()
        os.unlink(self.db_path)
    
    def test_create_speaker_profile(self):
        """Test creating speaker profiles."""
        # Create profile
        profile_id = self.db.create_speaker_profile(
            display_name="Test Speaker",
            feed_url="https://example.com/feed",
            canonical_label="HOST"
        )
        
        self.assertIsInstance(profile_id, int)
        self.assertGreater(profile_id, 0)
        
        # Test duplicate handling
        profile_id2 = self.db.create_speaker_profile(
            display_name="Test Speaker",
            feed_url="https://example.com/feed"
        )
        self.assertEqual(profile_id, profile_id2)
    
    def test_add_speaker_embedding(self):
        """Test adding embeddings to profiles."""
        # Create profile
        profile_id = self.db.create_speaker_profile("Test Speaker")
        
        # Add embedding
        embedding = np.random.randn(256).astype(np.float32)
        embedding_id = self.db.add_speaker_embedding(
            profile_id=profile_id,
            embedding=serialize_embedding(embedding),
            embedding_dimension=256,
            quality_score=0.9,
            extraction_method="mock"
        )
        
        self.assertIsInstance(embedding_id, int)
        self.assertGreater(embedding_id, 0)
        
        # Retrieve embeddings
        embeddings = self.db.get_speaker_embeddings(profile_id)
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(embeddings[0]['embedding_dimension'], 256)
        self.assertEqual(embeddings[0]['quality_score'], 0.9)
    
    def test_link_speaker_occurrence(self):
        """Test linking temporary speakers to profiles."""
        # Create profile
        profile_id = self.db.create_speaker_profile("Test Speaker")
        
        # Create dummy transcription
        result = TranscriptionResult()
        result.filename = "test.wav"
        result.file_hash = "test123"
        result.language = "en"
        result.full_text = "Test"
        trans_id = self.db.save_transcription(result)
        
        # Link occurrence
        occurrence_id = self.db.link_speaker_occurrence(
            transcription_id=trans_id,
            temporary_label="SPEAKER_0",
            profile_id=profile_id,
            confidence=0.95
        )
        
        self.assertIsInstance(occurrence_id, int)
        self.assertGreater(occurrence_id, 0)
    
    def test_get_speaker_profiles_for_feed(self):
        """Test retrieving profiles for a feed."""
        feed_url = "https://example.com/feed"
        
        # Create profiles
        self.db.create_speaker_profile("Speaker 1", feed_url=feed_url)
        self.db.create_speaker_profile("Speaker 2", feed_url=feed_url)
        self.db.create_speaker_profile("Other Speaker", feed_url="https://other.com")
        
        # Get profiles for feed
        profiles = self.db.get_speaker_profiles_for_feed(feed_url)
        self.assertEqual(len(profiles), 2)
        
        names = [p['display_name'] for p in profiles]
        self.assertIn("Speaker 1", names)
        self.assertIn("Speaker 2", names)
        self.assertNotIn("Other Speaker", names)


class TestSpeakerMatcher(unittest.TestCase):
    """Test speaker matching functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_file.name
        self.db = TranscriptionDatabase(self.db_path)
        self.db.create_tables()
        self.db.create_speaker_identity_tables()
        self.matcher = SpeakerMatcher(self.db, threshold=0.85, embedding_method="mock")
    
    def tearDown(self):
        """Clean up."""
        self.temp_file.close()
        os.unlink(self.db_path)
    
    def test_find_best_match(self):
        """Test finding best matching profiles."""
        # Create profiles with embeddings
        profile1_id = self.db.create_speaker_profile("Speaker 1")
        embedding1 = np.random.randn(256).astype(np.float32)
        self.db.add_speaker_embedding(
            profile1_id, 
            serialize_embedding(embedding1),
            256
        )
        
        profile2_id = self.db.create_speaker_profile("Speaker 2")
        embedding2 = np.random.randn(256).astype(np.float32)
        self.db.add_speaker_embedding(
            profile2_id,
            serialize_embedding(embedding2),
            256
        )
        
        # Find matches for embedding1 (should match itself)
        matches = self.matcher.find_best_match(embedding1)
        
        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0][0], profile1_id)
        self.assertGreater(matches[0][1], 0.9)  # High similarity
    
    def test_match_speaker_create_new(self):
        """Test creating new profile for unmatched speaker."""
        # New embedding with no matches
        embedding = np.random.randn(256).astype(np.float32)
        
        profile_id, confidence = self.matcher.match_speaker(
            embedding,
            create_if_not_found=True,
            speaker_hint="New Speaker"
        )
        
        self.assertIsNotNone(profile_id)
        self.assertEqual(confidence, 1.0)
        
        # Verify profile was created
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT display_name FROM speaker_profiles WHERE id = ?", (profile_id,))
            result = cursor.fetchone()
            self.assertEqual(result['display_name'], "New Speaker")
    
    def test_merge_speaker_profiles(self):
        """Test merging duplicate profiles."""
        # Create two profiles
        profile1_id = self.db.create_speaker_profile("Speaker A")
        profile2_id = self.db.create_speaker_profile("Speaker A (duplicate)")
        
        # Add embeddings to both
        embedding = np.random.randn(256).astype(np.float32)
        self.db.add_speaker_embedding(profile1_id, serialize_embedding(embedding), 256)
        self.db.add_speaker_embedding(profile2_id, serialize_embedding(embedding), 256)
        
        # Merge profiles
        success = self.matcher.merge_speaker_profiles(profile1_id, profile2_id)
        self.assertTrue(success)
        
        # Verify profile2 is deleted
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM speaker_profiles WHERE id = ?", (profile2_id,))
            result = cursor.fetchone()
            self.assertEqual(result['count'], 0)
        
        # Verify embeddings were transferred
        embeddings = self.db.get_speaker_embeddings(profile1_id)
        self.assertEqual(len(embeddings), 2)


class TestEndToEndSpeakerIdentity(unittest.TestCase):
    """Test complete speaker identity workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_file.name
        self.db = TranscriptionDatabase(self.db_path)
        self.db.create_tables()
        self.db.create_speaker_identity_tables()
    
    def tearDown(self):
        """Clean up."""
        self.temp_file.close()
        os.unlink(self.db_path)
    
    def test_speaker_identification_workflow(self):
        """Test complete workflow from transcription to speaker identification."""
        feed_url = "https://podcast.example.com/feed"
        
        # Step 1: Create known speaker profiles
        host_id = self.db.create_speaker_profile(
            "John Doe",
            feed_url=feed_url,
            canonical_label="HOST"
        )
        
        # Add reference embedding for host
        extractor = VoiceEmbeddingExtractor(method="mock")
        host_embedding, _ = extractor.extract_embedding_from_file("dummy.wav")
        self.db.add_speaker_embedding(
            host_id,
            serialize_embedding(host_embedding),
            256,
            quality_score=0.95
        )
        
        # Step 2: Create transcription with speakers
        result = TranscriptionResult()
        result.filename = "episode_001.mp3"
        result.file_hash = "abc123"
        result.language = "en"
        result.full_text = "Welcome to our podcast."
        result._proto.num_speakers = 2
        result._proto.has_speaker_labels = True
        
        # Add segments
        result.add_segment(0, 5, "Welcome to our podcast.")
        result.segments[0].speaker = "SPEAKER_0"
        result.segments[0].confidence = 0.95
        
        result.add_segment(5, 10, "Thanks for having me.")
        result.segments[1].speaker = "SPEAKER_1"
        result.segments[1].confidence = 0.92
        
        # Add embeddings (SPEAKER_0 will match host)
        guest_embedding, _ = extractor.extract_embedding_from_file("dummy2.wav")
        result._speaker_embeddings = {
            "SPEAKER_0": (host_embedding, 5.0, [0]),
            "SPEAKER_1": (guest_embedding, 5.0, [1]),
        }
        result._feed_url = feed_url
        
        # Step 3: Save transcription (triggers identification)
        trans_id = self.db.save_transcription(result, feed_url=feed_url)
        
        # Step 4: Verify speaker identification
        # Check occurrences
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM speaker_occurrences 
                WHERE transcription_id = ?
                ORDER BY temporary_label
            """, (trans_id,))
            occurrences = cursor.fetchall()
        
        self.assertEqual(len(occurrences), 2)
        
        # SPEAKER_0 should match host
        self.assertEqual(occurrences[0]['temporary_label'], "SPEAKER_0")
        self.assertEqual(occurrences[0]['profile_id'], host_id)
        self.assertGreater(occurrences[0]['confidence'], 0.8)
        
        # SPEAKER_1 should be new profile
        self.assertEqual(occurrences[1]['temporary_label'], "SPEAKER_1")
        self.assertNotEqual(occurrences[1]['profile_id'], host_id)
        
        # Step 5: Verify we can query by speaker
        host_statements = self.db.get_speaker_statements(host_id)
        self.assertEqual(len(host_statements), 1)
        self.assertEqual(host_statements[0]['text'], "Welcome to our podcast.")


if __name__ == "__main__":
    unittest.main()