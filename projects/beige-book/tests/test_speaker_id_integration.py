#!/usr/bin/env python3
"""
Integration test for speaker identification with transcription.

This test verifies that the speaker identification system correctly
integrates with the transcription pipeline.
"""

import os
import tempfile
import unittest
from pathlib import Path

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinkhaus_models.database import TranscriptionDatabase
from beige_book.transcriber import AudioTranscriber, TranscriptionResult, Segment
from beige_book.speaker_matcher import SpeakerMatcher
from beige_book.voice_embeddings import VoiceEmbeddingExtractor, serialize_embedding
from beige_book.speaker_diarizer import SpeakerDiarizer
import numpy as np
from .test_helpers import has_pyannote_requirements, skipUnlessRealDiarization, get_diarization_mode


class TestSpeakerIdentificationIntegration(unittest.TestCase):
    """Test speaker identification integrated with transcription."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db = TranscriptionDatabase(self.temp_db.name)
        self.db.create_tables()
        self.db.create_speaker_identity_tables()
        
        self.feed_url = "https://test-podcast.com/feed.rss"
        self.audio_path = Path(__file__).parent.parent.parent.parent / "resources" / "audio" / "harvard.wav"
        
    def tearDown(self):
        """Clean up."""
        self.temp_db.close()
        os.unlink(self.temp_db.name)
    
    @skipUnlessRealDiarization
    def test_speaker_diarization_integration(self):
        """Test speaker diarization with transcription."""
        # Always use real diarization in tests
        hf_token = os.getenv("HF_TOKEN")
        diarizer = SpeakerDiarizer(auth_token=hf_token)
        diarization = diarizer.diarize_file(str(self.audio_path), use_mock=False)
        
        self.assertIsNotNone(diarization)
        self.assertGreaterEqual(diarization.num_speakers, 1)  # At least 1 speaker
        self.assertGreater(len(diarization.segments), 0)
        
        # Verify segments have speaker labels
        for segment in diarization.segments:
            self.assertIsNotNone(segment.speaker)
            self.assertIn("SPEAKER_", segment.speaker)
            # Real diarization may not provide confidence scores
    
    def test_voice_embedding_extraction(self):
        """Test voice embedding extraction from transcription."""
        # Create a mock transcription result
        result = TranscriptionResult()
        result.filename = "test.mp3"
        result.file_hash = "test_hash"
        result.language = "en"
        result.full_text = "This is a test transcription."
        result.has_speaker_labels = True
        result.num_speakers = 2
        
        # Add segments with speaker labels
        seg1 = Segment(start_ms=0, end_ms=3000, text="Hello, this is speaker one.")
        seg1.speaker = "SPEAKER_0"
        seg1.confidence = 0.9
        result.segments.append(seg1)
        
        seg2 = Segment(start_ms=3000, end_ms=6000, text="And this is speaker two.")
        seg2.speaker = "SPEAKER_1"
        seg2.confidence = 0.85
        result.segments.append(seg2)
        
        seg3 = Segment(start_ms=6000, end_ms=9000, text="Speaker one again.")
        seg3.speaker = "SPEAKER_0"
        seg3.confidence = 0.88
        result.segments.append(seg3)
        
        # Extract embeddings
        extractor = VoiceEmbeddingExtractor(method="mock")
        embeddings = extractor.extract_embeddings_for_transcription(str(self.audio_path), result)
        
        # Verify embeddings were extracted
        self.assertEqual(len(embeddings), 2)
        self.assertIn("SPEAKER_0", embeddings)
        self.assertIn("SPEAKER_1", embeddings)
        
        # Verify embedding structure
        for speaker, (embedding, duration, indices) in embeddings.items():
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape, (256,))
            self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)
            self.assertGreater(duration, 0)
            self.assertIsInstance(indices, list)
    
    def test_speaker_matching_workflow(self):
        """Test the complete speaker matching workflow."""
        # Initialize matcher
        matcher = SpeakerMatcher(self.db, threshold=0.85, embedding_method="mock")
        
        # Episode 1: Create initial profiles
        host_id = self.db.create_speaker_profile(
            "Test Host",
            feed_url=self.feed_url,
            canonical_label="HOST"
        )
        
        # Generate and store host embedding
        np.random.seed(123)  # For consistent tests
        host_embedding = np.random.randn(256).astype(np.float32)
        host_embedding = host_embedding / np.linalg.norm(host_embedding)
        
        self.db.add_speaker_embedding(
            host_id,
            serialize_embedding(host_embedding),
            256,
            quality_score=0.95
        )
        
        # Episode 2: Test matching
        # Similar embedding (same speaker)
        similar_embedding = host_embedding + np.random.randn(256) * 0.05
        similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)
        
        matches = matcher.find_best_match(similar_embedding, feed_url=self.feed_url)
        
        self.assertIsNotNone(matches)
        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0][0], host_id)
        self.assertGreaterEqual(matches[0][1], matcher.threshold)
        
        # Different embedding (new speaker)
        different_embedding = np.random.randn(256).astype(np.float32)
        different_embedding = different_embedding / np.linalg.norm(different_embedding)
        
        matches = matcher.find_best_match(different_embedding, feed_url=self.feed_url)
        
        if matches:
            # Should not match or have low confidence
            self.assertLess(matches[0][1], matcher.threshold)
    
    def test_database_speaker_tracking(self):
        """Test database operations for speaker tracking."""
        # Create profiles
        host_id = self.db.create_speaker_profile(
            "John Doe",
            feed_url=self.feed_url,
            canonical_label="HOST"
        )
        
        guest_id = self.db.create_speaker_profile(
            "Jane Smith",
            feed_url=self.feed_url,
            canonical_label="GUEST"
        )
        
        # Add embeddings
        embedding1 = np.random.randn(256).astype(np.float32)
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        
        self.db.add_speaker_embedding(
            host_id,
            serialize_embedding(embedding1),
            256,
            quality_score=0.9
        )
        
        # Create transcription
        result = TranscriptionResult()
        result.filename = "test_episode.mp3"
        result.file_hash = "test_hash_123"
        result.language = "en"
        result.full_text = "Test transcription"
        
        trans_id = self.db.save_transcription(result, feed_url=self.feed_url)
        
        # Link speaker occurrences
        occ_id = self.db.link_speaker_occurrence(
            transcription_id=trans_id,
            temporary_label="SPEAKER_0",
            profile_id=host_id,
            confidence=0.95,
            is_verified=True
        )
        
        self.assertIsNotNone(occ_id)
        
        # Verify speaker profiles
        profiles = self.db.get_speaker_profiles_for_feed(self.feed_url)
        self.assertEqual(len(profiles), 2)
        
        # Verify embeddings
        embeddings = self.db.get_speaker_embeddings(host_id)
        self.assertEqual(len(embeddings), 1)
        
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        if not self.audio_path.exists():
            self.skipTest(f"Audio file not found: {self.audio_path}")
        
        transcriber = AudioTranscriber(model_name="tiny")
        
        # Transcribe without diarization
        result = transcriber.transcribe_file(str(self.audio_path), verbose=False)
        
        # Manually add speaker information
        result.has_speaker_labels = True
        result.num_speakers = 2
        
        # Assign speakers to segments
        for i, seg in enumerate(result.segments[:6]):  # Just first 6 segments
            seg.speaker = f"SPEAKER_{i % 2}"
            seg.confidence = 0.9
        
        # Extract embeddings
        extractor = VoiceEmbeddingExtractor(method="mock")
        embeddings = extractor.extract_embeddings_for_transcription(
            str(self.audio_path),
            result
        )
        
        # Create profiles and save
        if embeddings:
            # Create host profile for SPEAKER_0
            host_id = self.db.create_speaker_profile(
                "Test Host",
                feed_url=self.feed_url,
                canonical_label="HOST"
            )
            
            if "SPEAKER_0" in embeddings:
                embedding, duration, indices = embeddings["SPEAKER_0"]
                self.db.add_speaker_embedding(
                    host_id,
                    serialize_embedding(embedding),
                    256,
                    quality_score=0.95
                )
            
            # Save transcription
            trans_id = self.db.save_transcription(result, feed_url=self.feed_url)
            
            # Link speaker
            if "SPEAKER_0" in embeddings:
                self.db.link_speaker_occurrence(
                    transcription_id=trans_id,
                    temporary_label="SPEAKER_0",
                    profile_id=host_id,
                    confidence=0.95,
                    is_verified=True
                )
            
            # Verify
            profiles = self.db.get_speaker_profiles_for_feed(self.feed_url)
            self.assertGreater(len(profiles), 0)


if __name__ == "__main__":
    unittest.main()