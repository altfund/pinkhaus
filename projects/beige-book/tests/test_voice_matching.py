#!/usr/bin/env python3
"""
Test voice matching functionality.
"""

import os
import tempfile
import unittest
import json
from pathlib import Path

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinkhaus_models.database import TranscriptionDatabase
from beige_book.audio_processor import AudioProcessor
from beige_book.voice_matcher import find_voice_matches, format_results


class TestVoiceMatching(unittest.TestCase):
    """Test voice matching functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once."""
        cls.harvard_path = Path(__file__).parent.parent.parent.parent / "resources" / "audio" / "harvard.wav"
        if not cls.harvard_path.exists():
            raise FileNotFoundError(f"Test audio not found: {cls.harvard_path}")
            
        # Check for HF token
        cls.hf_token = os.getenv("HF_TOKEN")
        if not cls.hf_token:
            raise RuntimeError("HF_TOKEN required for this test")
    
    def setUp(self):
        """Set up for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize database
        self.db = TranscriptionDatabase(self.db_path)
        self.db.create_tables()
        self.db.create_speaker_identity_tables()
        
    def tearDown(self):
        """Clean up."""
        import os
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_voice_matching_same_speaker(self):
        """Test matching the same speaker across recordings."""
        # Initialize processor
        processor = AudioProcessor(
            db=self.db,
            model_name="tiny",
            hf_token=self.hf_token,
            embedding_method="mock",
            matcher_threshold=0.85
        )
        
        # Process first "episode"
        result1 = processor.process_audio_file(
            audio_path=str(self.harvard_path),
            feed_url="https://test.com/podcast1.rss",
            enable_diarization=True,
            create_new_profiles=True,
            verbose=False
        )
        
        self.assertGreater(len(result1['speaker_profiles']), 0)
        
        # Process second "episode" (same audio, but system doesn't know that)
        result2 = processor.process_audio_file(
            audio_path=str(self.harvard_path),
            feed_url="https://test.com/podcast1.rss", 
            enable_diarization=True,
            create_new_profiles=True,
            verbose=False
        )
        
        # Should match existing speaker
        print(f"\nResult2 matches: {result2['matches']}")
        matched_speakers = sum(1 for m in result2['matches'].values() if not m['is_new'])
        self.assertGreater(matched_speakers, 0, "Should match at least one existing speaker")
        
        # Now test voice matching on "unknown" audio
        matches = find_voice_matches(
            db=self.db,
            audio_path=str(self.harvard_path),
            hf_token=self.hf_token,
            embedding_method="mock",
            threshold=0.70,  # Lower threshold for mock mode
            verbose=True  # Enable verbose to see what's happening
        )
        
        # Verify results structure
        self.assertIn('speakers', matches)
        self.assertIn('appearances', matches)
        self.assertIn('summary', matches)
        
        # Should find matches
        self.assertGreater(matches['summary']['matched_speakers'], 0)
        self.assertGreater(matches['summary']['total_matches'], 0)
        
        # Check speaker details
        for speaker_label, speaker_data in matches['speakers'].items():
            self.assertIn('profile_id', speaker_data)
            self.assertIn('display_name', speaker_data)
            self.assertIn('confidence', speaker_data)
            self.assertIn('duration', speaker_data)
            self.assertIn('matches', speaker_data)
            
            if speaker_data['profile_id']:
                # Matched speaker should have appearances
                self.assertGreater(len(speaker_data['matches']), 0)
                
                # Check match structure
                for match in speaker_data['matches']:
                    self.assertIn('filename', match)
                    self.assertIn('feed_url', match)
                    self.assertIn('transcription_id', match)
    
    def test_voice_matching_no_match(self):
        """Test when no matching speakers are found."""
        # Empty database - no speakers to match
        matches = find_voice_matches(
            db=self.db,
            audio_path=str(self.harvard_path),
            hf_token=self.hf_token,
            embedding_method="mock",
            threshold=0.85,
            verbose=False
        )
        
        # Should detect speakers but not match any
        self.assertGreater(matches['summary']['total_speakers'], 0)
        self.assertEqual(matches['summary']['matched_speakers'], 0)
        self.assertEqual(matches['summary']['total_matches'], 0)
        
        # All speakers should be unknown
        for speaker_label, speaker_data in matches['speakers'].items():
            self.assertIsNone(speaker_data['profile_id'])
            self.assertEqual(speaker_data['display_name'], 'Unknown')
            self.assertEqual(len(speaker_data['matches']), 0)
    
    def test_format_results(self):
        """Test result formatting."""
        # Create mock results
        results = {
            "speakers": {
                "SPEAKER_00": {
                    "profile_id": 1,
                    "display_name": "Test Speaker",
                    "confidence": 0.95,
                    "duration": 30.5,
                    "matches": [
                        {
                            "filename": "episode1.mp3",
                            "feed_url": "https://test.com/feed.rss",
                            "feed_title": "Episode 1",
                            "published": "2024-01-01",
                            "transcription_id": 1
                        }
                    ]
                },
                "SPEAKER_01": {
                    "profile_id": None,
                    "display_name": "Unknown",
                    "confidence": 0.0,
                    "duration": 15.2,
                    "matches": []
                }
            },
            "appearances": [],
            "summary": {
                "total_speakers": 2,
                "matched_speakers": 1,
                "total_matches": 1,
                "unique_podcasts": 1,
                "unique_feeds": 1
            }
        }
        
        # Test summary format
        summary = format_results(results, "summary")
        self.assertIn("Total speakers detected: 2", summary)
        self.assertIn("Test Speaker", summary)
        self.assertIn("0.950", summary)
        
        # Test detailed format
        detailed = format_results(results, "detailed")
        self.assertIn("episode1.mp3", detailed)
        self.assertIn("Episode 1", detailed)
        
        # Test JSON format
        json_output = format_results(results, "json")
        parsed = json.loads(json_output)
        self.assertEqual(parsed['summary']['total_speakers'], 2)


if __name__ == "__main__":
    unittest.main()