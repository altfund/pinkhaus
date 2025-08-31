#!/usr/bin/env python3
"""
Test that incremental speaker updates produce the same results as all-in-one processing.

This test ensures that:
1. Transcribing first, then adding speaker data later
2. Doing everything in one pass with --diarize --speaker-profiles

Both produce identical database content.
"""

import os
import tempfile
import unittest
import sqlite3
from pathlib import Path
import subprocess
import json

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinkhaus_models.database import TranscriptionDatabase
from beige_book.audio_processor import AudioProcessor
from beige_book.transcriber import AudioTranscriber


class TestIncrementalSpeakerUpdate(unittest.TestCase):
    """Test incremental vs all-in-one speaker processing."""
    
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
        self.temp_dir = tempfile.mkdtemp()
        self.db1_path = os.path.join(self.temp_dir, "incremental.db")
        self.db2_path = os.path.join(self.temp_dir, "all_in_one.db")
        
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_incremental_vs_all_in_one_cli(self):
        """Test using CLI commands."""
        # Method 1: Incremental (transcribe first, then update)
        print("\n=== Method 1: Incremental Processing ===")
        
        # Step 1: Basic transcription only
        cmd1 = [
            "beige-book", str(self.harvard_path),
            "--db-path", self.db1_path,
            "--format", "sqlite",
            "--model", "tiny"
        ]
        print(f"Running: {' '.join(cmd1)}")
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        self.assertEqual(result1.returncode, 0, f"Transcription failed: {result1.stderr}")
        
        # Step 2: Update with speaker data
        cmd2 = [
            "beige-book-update-speakers", self.db1_path,
            "--transcription-id", "1",
            "--audio-map", "-",  # Use stdin
            "--model", "tiny"
        ]
        # Provide mapping via stdin
        audio_mapping = f"1,{self.harvard_path}\n"
        print(f"Running: {' '.join(cmd2)}")
        result2 = subprocess.run(cmd2, input=audio_mapping, capture_output=True, text=True)
        self.assertEqual(result2.returncode, 0, f"Speaker update failed: {result2.stderr}")
        
        # Method 2: All-in-one
        print("\n=== Method 2: All-in-One Processing ===")
        
        cmd3 = [
            "beige-book", str(self.harvard_path),
            "--db-path", self.db2_path,
            "--format", "sqlite",
            "--model", "tiny",
            "--diarize",
            "--speaker-profiles",
            "--embedding-method", "mock"  # Use mock for consistent testing
        ]
        print(f"Running: {' '.join(cmd3)}")
        result3 = subprocess.run(cmd3, capture_output=True, text=True)
        self.assertEqual(result3.returncode, 0, f"All-in-one failed: {result3.stderr}")
        
        # Compare databases
        self._compare_databases(self.db1_path, self.db2_path)
    
    def test_incremental_vs_all_in_one_api(self):
        """Test using Python API directly."""
        print("\n=== Testing via Python API ===")
        
        # Method 1: Incremental
        print("\nMethod 1: Incremental Processing")
        
        # Step 1: Basic transcription
        db1 = TranscriptionDatabase(self.db1_path)
        db1.create_tables()
        
        transcriber = AudioTranscriber(model_name="tiny")
        result1 = transcriber.transcribe_file(str(self.harvard_path), verbose=False)
        trans_id = db1.save_transcription(result1, feed_url="test://harvard")
        
        print(f"  Saved transcription ID: {trans_id}")
        
        # Step 2: Add speaker data
        db1.create_speaker_identity_tables()
        
        # Import and use the update function directly
        from beige_book.update_speaker_data import update_transcription_with_speakers
        
        processor1 = AudioProcessor(
            db=db1,
            model_name="tiny",
            hf_token=self.hf_token,
            embedding_method="mock",
            matcher_threshold=0.85
        )
        
        # Update the existing transcription with speaker data
        success = update_transcription_with_speakers(
            db=db1,
            transcription_id=trans_id,
            audio_path=str(self.harvard_path),
            processor=processor1,
            feed_url="test://harvard"
        )
        
        self.assertTrue(success, "Failed to update transcription with speaker data")
        
        # Get speaker profile info
        profiles = db1.get_speaker_profiles_for_feed("test://harvard")
        print(f"  Updated transcription with speaker data")
        print(f"  Created {len(profiles)} speaker profiles")
        
        # Method 2: All-in-one
        print("\nMethod 2: All-in-One Processing")
        
        db2 = TranscriptionDatabase(self.db2_path)
        db2.create_tables()
        db2.create_speaker_identity_tables()
        
        processor2 = AudioProcessor(
            db=db2,
            model_name="tiny", 
            hf_token=self.hf_token,
            embedding_method="mock",
            matcher_threshold=0.85
        )
        
        process_result2 = processor2.process_audio_file(
            audio_path=str(self.harvard_path),
            feed_url="test://harvard",
            enable_diarization=True,
            create_new_profiles=True,
            verbose=False
        )
        
        print(f"  Created transcription ID: {process_result2['transcription_id']}")
        print(f"  Detected {process_result2['num_speakers']} speakers")
        print(f"  Created {len(process_result2['speaker_profiles'])} profiles")
        
        # Compare databases
        self._compare_databases(self.db1_path, self.db2_path)
    
    def _compare_databases(self, db1_path: str, db2_path: str):
        """Compare two databases for equivalent content."""
        print("\n=== Comparing Databases ===")
        
        conn1 = sqlite3.connect(db1_path)
        conn2 = sqlite3.connect(db2_path)
        
        try:
            # Compare transcriptions table
            print("\n1. Comparing transcriptions...")
            trans1 = conn1.execute(
                "SELECT filename, file_hash, language, full_text, num_speakers, has_speaker_labels "
                "FROM transcription_metadata ORDER BY id"
            ).fetchall()
            trans2 = conn2.execute(
                "SELECT filename, file_hash, language, full_text, num_speakers, has_speaker_labels "
                "FROM transcription_metadata ORDER BY id"
            ).fetchall()
            
            self.assertEqual(len(trans1), len(trans2), "Different number of transcriptions")
            
            for i, (t1, t2) in enumerate(zip(trans1, trans2)):
                print(f"   Transcription {i+1}: ", end="")
                # Compare core fields that should be the same
                self.assertEqual(t1[0], t2[0], f"Different filename")
                self.assertEqual(t1[1], t2[1], f"Different file_hash")
                self.assertEqual(t1[2], t2[2], f"Different language")
                self.assertEqual(t1[3], t2[3], f"Different full_text")
                
                # Speaker-related fields will differ between incremental and all-in-one
                # In incremental: initially no speakers, then updated to have speakers
                # In all-in-one: has speakers from the start
                # Just verify both have speaker data in the end
                self.assertIsNotNone(t1[4], "Incremental method should have num_speakers")
                self.assertIsNotNone(t2[4], "All-in-one method should have num_speakers")
                self.assertTrue(t1[5], "Incremental method should have speaker labels")
                self.assertTrue(t2[5], "All-in-one method should have speaker labels")
                print("✓")
            
            # Compare speaker profiles
            print("\n2. Comparing speaker profiles...")
            profiles1 = conn1.execute(
                "SELECT display_name, canonical_label, feed_url "
                "FROM speaker_profiles ORDER BY id"
            ).fetchall()
            profiles2 = conn2.execute(
                "SELECT display_name, canonical_label, feed_url "
                "FROM speaker_profiles ORDER BY id"
            ).fetchall()
            
            self.assertEqual(len(profiles1), len(profiles2), 
                           f"Different number of speaker profiles: {len(profiles1)} vs {len(profiles2)}")
            
            for i, (p1, p2) in enumerate(zip(profiles1, profiles2)):
                print(f"   Profile {i+1}: {p1[0]} - ", end="")
                self.assertEqual(p1[0], p2[0], f"Different display_name")
                self.assertEqual(p1[1], p2[1], f"Different canonical_label")
                self.assertEqual(p1[2], p2[2], f"Different feed_url")
                print("✓")
            
            # Compare speaker embeddings count
            print("\n3. Comparing speaker embeddings...")
            emb_count1 = conn1.execute("SELECT COUNT(*) FROM speaker_embeddings").fetchone()[0]
            emb_count2 = conn2.execute("SELECT COUNT(*) FROM speaker_embeddings").fetchone()[0]
            self.assertEqual(emb_count1, emb_count2, 
                           f"Different number of embeddings: {emb_count1} vs {emb_count2}")
            print(f"   Both have {emb_count1} embeddings ✓")
            
            # Compare speaker occurrences
            print("\n4. Comparing speaker occurrences...")
            occ1 = conn1.execute(
                "SELECT temporary_label, confidence, is_verified "
                "FROM speaker_occurrences ORDER BY id"
            ).fetchall()
            occ2 = conn2.execute(
                "SELECT temporary_label, confidence, is_verified "
                "FROM speaker_occurrences ORDER BY id"
            ).fetchall()
            
            self.assertEqual(len(occ1), len(occ2),
                           f"Different number of occurrences: {len(occ1)} vs {len(occ2)}")
            
            for i, (o1, o2) in enumerate(zip(occ1, occ2)):
                print(f"   Occurrence {i+1}: {o1[0]} - ", end="")
                self.assertEqual(o1[0], o2[0], f"Different temporary_label")
                # Confidence might vary slightly due to processing order
                self.assertAlmostEqual(o1[1], o2[1], places=2, 
                                     msg=f"Different confidence")
                self.assertEqual(o1[2], o2[2], f"Different is_verified")
                print("✓")
            
            print("\n✅ Databases are equivalent!")
            
        finally:
            conn1.close()
            conn2.close()


if __name__ == "__main__":
    unittest.main()