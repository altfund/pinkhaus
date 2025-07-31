"""
Test database export/import functionality for round-trip JSON/TOML operations.
"""

import json
import os
import tempfile
import toml
import pytest
from pinkhaus_models import TranscriptionDatabase, TranscriptionResult, Segment
from beige_book.transcriber import TranscriptionResult as BeigeTranscriptionResult


class TestDatabaseExportImport:
    """Test export and import functionality for database transcriptions."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        db = TranscriptionDatabase(db_path)
        db.create_tables()
        
        yield db
        
        # Cleanup
        os.unlink(db_path)
    
    @pytest.fixture
    def sample_transcription(self):
        """Create a sample transcription result."""
        result = TranscriptionResult(
            filename="test_podcast.mp3",
            file_hash="abc123def456",
            language="en",
            segments=[
                Segment(0.0, 5.0, "Hello and welcome to the podcast."),
                Segment(5.0, 10.0, "Today we'll discuss economics."),
                Segment(10.0, 15.0, "Let's start with inflation.")
            ],
            full_text="Hello and welcome to the podcast. Today we'll discuss economics. Let's start with inflation."
        )
        return result
    
    @pytest.fixture
    def sample_beige_transcription(self):
        """Create a sample beige-book style transcription result."""
        result = BeigeTranscriptionResult()
        result.filename = "test_beige.mp3"
        result.file_hash = "xyz789"
        result.language = "en"
        result.full_text = "This is a test transcription."
        result.add_segment(0.0, 3.0, "This is a test")
        result.add_segment(3.0, 5.0, "transcription.")
        return result
    
    def test_single_transcription_export_import_json(self, temp_db, sample_transcription):
        """Test exporting a single transcription to JSON and re-importing it."""
        # Save to database
        transcription_id = temp_db.save_transcription(
            sample_transcription,
            model_name="base",
            feed_url="https://example.com/feed.xml",
            feed_item_id="episode-1",
            feed_item_title="Episode 1: Economics",
            feed_item_published="2025-07-01T10:00:00Z"
        )
        
        # Export from database
        exported_result = temp_db.export_to_dict(transcription_id)
        assert exported_result is not None
        
        # Convert to JSON
        json_str = json.dumps(exported_result.to_dict(), indent=2)
        
        # Create new database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            new_db_path = f.name
        
        try:
            new_db = TranscriptionDatabase(new_db_path)
            new_db.create_tables()
            
            # Import from JSON
            data = json.loads(json_str)
            imported_result = TranscriptionResult.from_dict(data)
            
            # Save imported result
            new_id = new_db.save_transcription(
                imported_result,
                model_name="base",
                feed_url="https://example.com/feed.xml",
                feed_item_id="episode-1",
                feed_item_title="Episode 1: Economics",
                feed_item_published="2025-07-01T10:00:00Z"
            )
            
            # Verify data matches
            original_data = temp_db.get_transcription(transcription_id)
            imported_data = new_db.get_transcription(new_id)
            
            assert original_data["metadata"]["filename"] == imported_data["metadata"]["filename"]
            assert original_data["metadata"]["file_hash"] == imported_data["metadata"]["file_hash"]
            assert original_data["metadata"]["language"] == imported_data["metadata"]["language"]
            assert original_data["metadata"]["full_text"] == imported_data["metadata"]["full_text"]
            assert len(original_data["segments"]) == len(imported_data["segments"])
            
            for orig_seg, imp_seg in zip(original_data["segments"], imported_data["segments"]):
                assert orig_seg["start_time"] == imp_seg["start_time"]
                assert orig_seg["end_time"] == imp_seg["end_time"]
                assert orig_seg["text"] == imp_seg["text"]
                
        finally:
            os.unlink(new_db_path)
    
    def test_beige_transcription_json_round_trip(self, temp_db, sample_beige_transcription):
        """Test beige-book TranscriptionResult JSON round-trip."""
        # Export to JSON
        json_str = sample_beige_transcription.to_json()
        
        # Import from JSON
        imported = BeigeTranscriptionResult.from_json(json_str)
        
        # Verify
        assert imported.filename == sample_beige_transcription.filename
        assert imported.file_hash == sample_beige_transcription.file_hash
        assert imported.language == sample_beige_transcription.language
        assert imported.full_text == sample_beige_transcription.full_text
        assert len(imported.segments) == len(sample_beige_transcription.segments)
        
        # Save to database
        transcription_id = temp_db.save_transcription(
            imported,
            model_name="base"
        )
        
        # Retrieve and verify
        data = temp_db.get_transcription(transcription_id)
        assert data["metadata"]["filename"] == sample_beige_transcription.filename
        assert data["metadata"]["file_hash"] == sample_beige_transcription.file_hash
    
    def test_beige_transcription_toml_round_trip(self, temp_db, sample_beige_transcription):
        """Test beige-book TranscriptionResult TOML round-trip."""
        # Export to TOML
        toml_str = sample_beige_transcription.to_toml()
        
        # Import from TOML
        imported = BeigeTranscriptionResult.from_toml(toml_str)
        
        # Verify
        assert imported.filename == sample_beige_transcription.filename
        assert imported.file_hash == sample_beige_transcription.file_hash
        assert imported.language == sample_beige_transcription.language
        assert imported.full_text == sample_beige_transcription.full_text
        assert len(imported.segments) == len(sample_beige_transcription.segments)
        
        # Save to database
        transcription_id = temp_db.save_transcription(
            imported,
            model_name="base"
        )
        
        # Retrieve and verify
        data = temp_db.get_transcription(transcription_id)
        assert data["metadata"]["filename"] == sample_beige_transcription.filename
        assert data["metadata"]["file_hash"] == sample_beige_transcription.file_hash
    
    def test_multiple_transcriptions_export_import(self, temp_db):
        """Test exporting and importing multiple transcriptions."""
        # Create multiple transcriptions
        transcriptions = []
        ids = []
        
        for i in range(3):
            result = TranscriptionResult(
                filename=f"episode_{i}.mp3",
                file_hash=f"hash_{i}",
                language="en",
                segments=[
                    Segment(0.0, 5.0, f"Episode {i} intro"),
                    Segment(5.0, 10.0, f"Episode {i} content")
                ],
                full_text=f"Episode {i} intro Episode {i} content"
            )
            transcriptions.append(result)
            
            tid = temp_db.save_transcription(
                result,
                model_name="base",
                feed_url="https://example.com/feed.xml",
                feed_item_id=f"episode-{i}",
                feed_item_title=f"Episode {i}"
            )
            ids.append(tid)
        
        # Export all to JSON
        all_data = []
        for tid in ids:
            result = temp_db.export_to_dict(tid)
            if result:
                # Get metadata from database for feed info
                db_data = temp_db.get_transcription(tid)
                metadata = db_data["metadata"]
                
                export_data = {
                    "transcription": result.to_dict(),
                    "metadata": {
                        "model_name": metadata["model_name"],
                        "feed_url": metadata["feed_url"],
                        "feed_item_id": metadata["feed_item_id"],
                        "feed_item_title": metadata["feed_item_title"],
                        "feed_item_published": metadata["feed_item_published"]
                    }
                }
                all_data.append(export_data)
        
        # Save to JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"transcriptions": all_data}, f, indent=2)
            json_path = f.name
        
        try:
            # Create new database and import
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                new_db_path = f.name
            
            try:
                new_db = TranscriptionDatabase(new_db_path)
                new_db.create_tables()
                
                # Load and import
                with open(json_path, 'r') as f:
                    loaded_data = json.load(f)
                
                imported_ids = []
                for item in loaded_data["transcriptions"]:
                    result = TranscriptionResult.from_dict(item["transcription"])
                    metadata = item["metadata"]
                    
                    new_id = new_db.save_transcription(
                        result,
                        model_name=metadata["model_name"],
                        feed_url=metadata["feed_url"],
                        feed_item_id=metadata["feed_item_id"],
                        feed_item_title=metadata["feed_item_title"],
                        feed_item_published=metadata["feed_item_published"]
                    )
                    imported_ids.append(new_id)
                
                # Verify all transcriptions
                assert len(imported_ids) == len(ids)
                
                all_original = temp_db.get_all_transcriptions()
                all_imported = new_db.get_all_transcriptions()
                
                assert len(all_original) == len(all_imported)
                
            finally:
                os.unlink(new_db_path)
                
        finally:
            os.unlink(json_path)
    
    def test_export_with_feed_metadata(self, temp_db, sample_transcription):
        """Test that feed metadata is preserved in export/import."""
        # Save with feed metadata
        feed_metadata = {
            "feed_url": "https://podcast.example.com/feed.xml",
            "feed_item_id": "guid-12345",
            "feed_item_title": "Episode 42: The Answer",
            "feed_item_published": "2025-07-15T08:00:00Z"
        }
        
        transcription_id = temp_db.save_transcription(
            sample_transcription,
            model_name="medium",
            **feed_metadata
        )
        
        # Get full data including metadata
        db_data = temp_db.get_transcription(transcription_id)
        
        # Create export structure
        export_data = {
            "transcription": temp_db.export_to_dict(transcription_id).to_dict(),
            "metadata": {
                "model_name": db_data["metadata"]["model_name"],
                "feed_url": db_data["metadata"]["feed_url"],
                "feed_item_id": db_data["metadata"]["feed_item_id"],
                "feed_item_title": db_data["metadata"]["feed_item_title"],
                "feed_item_published": db_data["metadata"]["feed_item_published"]
            }
        }
        
        # Convert to JSON and back
        json_str = json.dumps(export_data, indent=2)
        loaded = json.loads(json_str)
        
        # Verify all metadata is preserved
        assert loaded["metadata"]["model_name"] == "medium"
        assert loaded["metadata"]["feed_url"] == feed_metadata["feed_url"]
        assert loaded["metadata"]["feed_item_id"] == feed_metadata["feed_item_id"]
        assert loaded["metadata"]["feed_item_title"] == feed_metadata["feed_item_title"]
        assert loaded["metadata"]["feed_item_published"] == feed_metadata["feed_item_published"]