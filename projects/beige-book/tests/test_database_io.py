"""
Test DatabaseIO utility for batch export/import operations.
"""

import json
import os
import tempfile
import toml
import pytest
from beige_book import DatabaseIO, TranscriptionResult, AudioTranscriber
from pinkhaus_models import TranscriptionDatabase, TranscriptionResult as PinkhausResult, Segment


class TestDatabaseIO:
    """Test DatabaseIO batch operations."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        # Initialize database
        db = TranscriptionDatabase(db_path)
        db.create_tables()
        
        yield db_path
        
        # Cleanup
        os.unlink(db_path)
    
    @pytest.fixture
    def populated_db(self, temp_db_path):
        """Create a database with sample data."""
        db = TranscriptionDatabase(temp_db_path)
        
        # Add multiple transcriptions
        for i in range(3):
            result = PinkhausResult(
                filename=f"episode_{i}.mp3",
                file_hash=f"hash_{i}",
                language="en",
                segments=[
                    Segment(0.0, 5.0, f"Episode {i} intro"),
                    Segment(5.0, 10.0, f"Episode {i} main content"),
                    Segment(10.0, 15.0, f"Episode {i} conclusion")
                ],
                full_text=f"Episode {i} intro Episode {i} main content Episode {i} conclusion"
            )
            
            db.save_transcription(
                result,
                model_name="base",
                feed_url=f"https://example{i}.com/feed.xml",
                feed_item_id=f"episode-{i}-guid",
                feed_item_title=f"Episode {i}: Test Content",
                feed_item_published=f"2025-07-{i+1:02d}T10:00:00Z"
            )
        
        return temp_db_path
    
    def test_export_all_to_json(self, populated_db):
        """Test exporting all transcriptions to JSON."""
        io = DatabaseIO(populated_db)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            # Export
            count = io.export_all_to_json(json_path)
            assert count == 3
            
            # Verify file contents
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            assert "version" in data
            assert data["version"] == "1.0"
            assert "transcriptions" in data
            assert len(data["transcriptions"]) == 3
            
            # Check first transcription
            first = data["transcriptions"][0]
            assert "transcription" in first
            assert "metadata" in first
            
            # Verify transcription fields
            trans = first["transcription"]
            assert trans["filename"] == "episode_2.mp3"  # Ordered by date
            assert trans["file_hash"] == "hash_2"
            assert trans["language"] == "en"
            assert len(trans["segments"]) == 3
            
            # Verify metadata fields
            meta = first["metadata"]
            assert meta["model_name"] == "base"
            assert meta["feed_url"] == "https://example2.com/feed.xml"
            assert meta["feed_item_id"] == "episode-2-guid"
            assert meta["feed_item_title"] == "Episode 2: Test Content"
            
        finally:
            os.unlink(json_path)
    
    def test_export_all_to_toml(self, populated_db):
        """Test exporting all transcriptions to TOML."""
        io = DatabaseIO(populated_db)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml_path = f.name
        
        try:
            # Export
            count = io.export_all_to_toml(toml_path)
            assert count == 3
            
            # Verify file contents
            with open(toml_path, 'r') as f:
                data = toml.load(f)
            
            assert data["version"] == "1.0"
            assert len(data["transcriptions"]) == 3
            
            # Check structure
            first = data["transcriptions"][0]
            assert first["filename"] == "episode_2.mp3"
            assert "metadata" in first
            assert first["metadata"]["model_name"] == "base"
            
        finally:
            os.unlink(toml_path)
    
    def test_import_from_json(self, temp_db_path):
        """Test importing transcriptions from JSON."""
        io = DatabaseIO(temp_db_path)
        
        # Create test JSON file
        test_data = {
            "version": "1.0",
            "transcriptions": [
                {
                    "transcription": {
                        "filename": "test1.mp3",
                        "file_hash": "hash1",
                        "language": "en",
                        "full_text": "Test transcription 1",
                        "segments": [
                            {
                                "start": 0.0,
                                "end": 5.0,
                                "text": "Test transcription 1",
                                "duration": 5.0
                            }
                        ]
                    },
                    "metadata": {
                        "model_name": "base",
                        "feed_url": "https://test.com/feed.xml",
                        "feed_item_id": "test1",
                        "feed_item_title": "Test Episode 1",
                        "feed_item_published": "2025-07-01T10:00:00Z"
                    }
                },
                {
                    "transcription": {
                        "filename": "test2.mp3",
                        "file_hash": "hash2",
                        "language": "en",
                        "full_text": "Test transcription 2",
                        "segments": []
                    },
                    "metadata": {
                        "model_name": "base"
                    }
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            json_path = f.name
        
        try:
            # Import
            result = io.import_from_json(json_path)
            assert result["imported"] == 2
            assert result["skipped"] == 0
            
            # Verify imported data
            db = TranscriptionDatabase(temp_db_path)
            all_trans = db.get_all_transcriptions()
            assert len(all_trans) == 2
            
            # Check first transcription
            trans1 = all_trans[0]
            assert trans1.filename == "test1.mp3"
            assert trans1.feed_url == "https://test.com/feed.xml"
            
            # Test duplicate handling
            result2 = io.import_from_json(json_path, skip_duplicates=True)
            assert result2["imported"] == 0
            assert result2["skipped"] == 2
            
        finally:
            os.unlink(json_path)
    
    def test_import_from_toml(self, temp_db_path):
        """Test importing transcriptions from TOML."""
        io = DatabaseIO(temp_db_path)
        
        # Create test TOML file
        test_data = {
            "version": "1.0",
            "transcriptions": [
                {
                    "filename": "test_toml.mp3",
                    "file_hash": "hash_toml",
                    "language": "en",
                    "full_text": "TOML test transcription",
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 3.0,
                            "text": "TOML test",
                            "duration": 3.0
                        }
                    ],
                    "metadata": {
                        "model_name": "medium",
                        "feed_url": "",
                        "feed_item_id": "",
                        "feed_item_title": "",
                        "feed_item_published": ""
                    }
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(test_data, f)
            toml_path = f.name
        
        try:
            # Import
            result = io.import_from_toml(toml_path)
            assert result["imported"] == 1
            assert result["skipped"] == 0
            
            # Verify
            db = TranscriptionDatabase(temp_db_path)
            all_trans = db.get_all_transcriptions()
            assert len(all_trans) == 1
            assert all_trans[0].filename == "test_toml.mp3"
            assert all_trans[0].model_name == "medium"
            
        finally:
            os.unlink(toml_path)
    
    def test_export_single_transcription(self, populated_db):
        """Test exporting a single transcription."""
        db = TranscriptionDatabase(populated_db)
        io = DatabaseIO(populated_db)
        
        # Get first transcription ID
        all_trans = db.get_all_transcriptions()
        trans_id = all_trans[0].id
        
        # Export to JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            success = io.export_transcription_to_json(trans_id, json_path)
            assert success is True
            
            # Verify
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            assert "transcription" in data
            assert "metadata" in data
            assert data["transcription"]["filename"] == all_trans[0].filename
            
        finally:
            os.unlink(json_path)
        
        # Export to TOML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml_path = f.name
        
        try:
            success = io.export_transcription_to_toml(trans_id, toml_path)
            assert success is True
            
            # Verify
            with open(toml_path, 'r') as f:
                data = toml.load(f)
            
            assert data["filename"] == all_trans[0].filename
            assert "metadata" in data
            
        finally:
            os.unlink(toml_path)
    
    def test_round_trip_json(self, populated_db):
        """Test complete round-trip export/import with JSON."""
        io1 = DatabaseIO(populated_db)
        
        # Export all
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            export_count = io1.export_all_to_json(export_path)
            assert export_count == 3
            
            # Create new database and import
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                new_db_path = f.name
            
            try:
                new_db = TranscriptionDatabase(new_db_path)
                new_db.create_tables()
                
                io2 = DatabaseIO(new_db_path)
                import_result = io2.import_from_json(export_path)
                assert import_result["imported"] == 3
                
                # Compare databases
                orig_trans = TranscriptionDatabase(populated_db).get_all_transcriptions()
                new_trans = new_db.get_all_transcriptions()
                
                assert len(orig_trans) == len(new_trans)
                
                for orig, new in zip(orig_trans, new_trans):
                    assert orig.filename == new.filename
                    assert orig.file_hash == new.file_hash
                    assert orig.language == new.language
                    assert orig.feed_url == new.feed_url
                    
            finally:
                os.unlink(new_db_path)
                
        finally:
            os.unlink(export_path)