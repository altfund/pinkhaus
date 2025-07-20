"""
Tests for the database module.
"""

import os
import sqlite3
import tempfile
import pytest
from pathlib import Path

from beige_book import AudioTranscriber, TranscriptionDatabase
from beige_book.transcriber import TranscriptionResult, Segment


class TestTranscriptionDatabase:
    """Test the TranscriptionDatabase class"""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def sample_result(self):
        """Create a sample TranscriptionResult for testing"""
        segments = [
            Segment(start=0.0, end=4.5, text="The stale smell of old beer lingers."),
            Segment(start=4.5, end=7.0, text="It takes heat to bring out the odor."),
            Segment(start=7.0, end=10.0, text="A cold dip restores health in zest.")
        ]
        
        return TranscriptionResult(
            filename="test_audio.wav",
            file_hash="abc123def456",
            language="en",
            segments=segments,
            full_text="The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health in zest."
        )
    
    def test_create_tables(self, temp_db):
        """Test table creation"""
        db = TranscriptionDatabase(temp_db)
        db.create_tables()
        
        # Verify tables exist
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # Check metadata table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='transcription_metadata'
        """)
        assert cursor.fetchone() is not None
        
        # Check segments table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='transcription_segments'
        """)
        assert cursor.fetchone() is not None
        
        conn.close()
    
    def test_save_transcription(self, temp_db, sample_result):
        """Test saving a transcription"""
        db = TranscriptionDatabase(temp_db)
        db.create_tables()
        
        # Save transcription
        transcription_id = db.save_transcription(sample_result, model_name="tiny")
        assert transcription_id > 0
        
        # Verify data was saved
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # Check metadata
        cursor.execute("SELECT * FROM transcription_metadata WHERE id = ?", (transcription_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == "test_audio.wav"  # filename
        assert row[2] == "abc123def456"     # file_hash
        assert row[3] == "en"               # language
        
        # Check segments
        cursor.execute("SELECT COUNT(*) FROM transcription_segments WHERE transcription_id = ?", 
                      (transcription_id,))
        count = cursor.fetchone()[0]
        assert count == 3
        
        conn.close()
    
    def test_duplicate_prevention(self, temp_db, sample_result):
        """Test that duplicate transcriptions are not saved"""
        db = TranscriptionDatabase(temp_db)
        db.create_tables()
        
        # Save twice with same file_hash and model
        id1 = db.save_transcription(sample_result, model_name="tiny")
        id2 = db.save_transcription(sample_result, model_name="tiny")
        
        # Should return the same ID
        assert id1 == id2
        
        # But different model should create new entry
        id3 = db.save_transcription(sample_result, model_name="base")
        assert id3 != id1
    
    def test_get_transcription(self, temp_db, sample_result):
        """Test retrieving a transcription"""
        db = TranscriptionDatabase(temp_db)
        db.create_tables()
        
        # Save and retrieve
        transcription_id = db.save_transcription(sample_result, model_name="tiny")
        data = db.get_transcription(transcription_id)
        
        assert data is not None
        assert data['metadata']['filename'] == "test_audio.wav"
        assert data['metadata']['file_hash'] == "abc123def456"
        assert len(data['segments']) == 3
        assert data['segments'][0]['text'] == "The stale smell of old beer lingers."
    
    def test_find_by_hash(self, temp_db, sample_result):
        """Test finding transcriptions by file hash"""
        db = TranscriptionDatabase(temp_db)
        db.create_tables()
        
        # Save with different models
        db.save_transcription(sample_result, model_name="tiny")
        db.save_transcription(sample_result, model_name="base")
        
        # Find by hash
        results = db.find_by_hash("abc123def456")
        assert len(results) == 2
        
        model_names = [r['model_name'] for r in results]
        assert "tiny" in model_names
        assert "base" in model_names
    
    def test_delete_transcription(self, temp_db, sample_result):
        """Test deleting a transcription"""
        db = TranscriptionDatabase(temp_db)
        db.create_tables()
        
        # Save and delete
        transcription_id = db.save_transcription(sample_result, model_name="tiny")
        assert db.delete_transcription(transcription_id) is True
        
        # Verify it's gone
        assert db.get_transcription(transcription_id) is None
        
        # Verify segments are also deleted (cascade)
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM transcription_segments WHERE transcription_id = ?", 
                      (transcription_id,))
        count = cursor.fetchone()[0]
        assert count == 0
        conn.close()
    
    def test_export_to_dict(self, temp_db, sample_result):
        """Test exporting back to TranscriptionResult"""
        db = TranscriptionDatabase(temp_db)
        db.create_tables()
        
        # Save and export
        transcription_id = db.save_transcription(sample_result, model_name="tiny")
        exported = db.export_to_dict(transcription_id)
        
        assert exported is not None
        assert isinstance(exported, TranscriptionResult)
        assert exported.filename == sample_result.filename
        assert exported.file_hash == sample_result.file_hash
        assert exported.language == sample_result.language
        assert len(exported.segments) == len(sample_result.segments)
        assert exported.full_text == sample_result.full_text
    
    def test_custom_table_names(self, temp_db, sample_result):
        """Test using custom table names"""
        db = TranscriptionDatabase(temp_db)
        
        # Create with custom names
        db.create_tables("my_metadata", "my_segments")
        
        # Save with custom names
        transcription_id = db.save_transcription(
            sample_result, 
            model_name="tiny",
            metadata_table="my_metadata",
            segments_table="my_segments"
        )
        
        # Retrieve with custom names
        data = db.get_transcription(
            transcription_id,
            metadata_table="my_metadata",
            segments_table="my_segments"
        )
        
        assert data is not None
        assert data['metadata']['filename'] == "test_audio.wav"
    
    def test_recent_transcriptions(self, temp_db):
        """Test getting recent transcriptions"""
        db = TranscriptionDatabase(temp_db)
        db.create_tables()
        
        # Create multiple transcriptions
        for i in range(5):
            segments = [Segment(start=0.0, end=1.0, text=f"Test {i}")]
            result = TranscriptionResult(
                filename=f"test_{i}.wav",
                file_hash=f"hash_{i}",
                language="en",
                segments=segments,
                full_text=f"Test {i}"
            )
            db.save_transcription(result, model_name="tiny")
        
        # Get recent
        recent = db.get_recent_transcriptions(limit=3)
        assert len(recent) == 3
        
        # Should be in reverse chronological order (most recent first)
        # Since they're all created at nearly the same time, they'll be ordered by ID descending
        filenames = [r['filename'] for r in recent]
        # The last 3 created should be present
        assert "test_4.wav" in filenames
        assert "test_3.wav" in filenames
        assert "test_2.wav" in filenames


class TestDatabaseIntegration:
    """Test database integration with real transcription"""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def harvard_result(self):
        """Get real transcription result for harvard.wav"""
        transcriber = AudioTranscriber(model_name="tiny")
        # Path to harvard.wav in resources directory
        # From test file: parent -> tests, parent.parent -> beige-book  
        # Then ../../resources/audio/harvard.wav from beige-book
        harvard_path = Path(__file__).parent.parent / ".." / ".." / "resources" / "audio" / "harvard.wav"
        return transcriber.transcribe_file(str(harvard_path), verbose=False)
    
    def test_real_transcription_storage(self, temp_db, harvard_result):
        """Test storing a real transcription"""
        db = TranscriptionDatabase(temp_db)
        db.create_tables()
        
        # Save real transcription
        transcription_id = db.save_transcription(harvard_result, model_name="tiny")
        
        # Verify
        data = db.get_transcription(transcription_id)
        assert data is not None
        assert data['metadata']['filename'] == "harvard.wav"
        assert data['metadata']['file_hash'] == "971b4163670445c415c6b0fb6813c38093409ecac2f6b4d429ae3574d24ad470"
        assert len(data['segments']) == 6
        
        # Check a specific segment
        first_segment = data['segments'][0]
        assert "stale smell" in first_segment['text'].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])