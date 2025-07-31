"""
Test database operations in pinkhaus-models.
"""

import pytest
import tempfile
import os
import sqlite3
from datetime import datetime, timedelta
from pinkhaus_models import TranscriptionDatabase, TranscriptionResult, Segment
from pinkhaus_models.models import TranscriptionMetadata, TranscriptionSegment


class TestTranscriptionDatabase:
    """Test the TranscriptionDatabase class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db = TranscriptionDatabase(db_path)
        db.create_tables()

        yield db

        # Cleanup
        os.unlink(db_path)

    @pytest.fixture
    def sample_result(self):
        """Create a sample transcription result."""
        return TranscriptionResult(
            filename="test_audio.mp3",
            file_hash="hash123abc",
            language="en",
            segments=[
                Segment(0.0, 5.0, "First segment of the transcription."),
                Segment(5.0, 10.0, "Second segment here."),
                Segment(10.0, 15.0, "Final segment of audio."),
            ],
            full_text="First segment of the transcription. Second segment here. Final segment of audio.",
        )

    def test_create_tables(self, temp_db):
        """Test that tables are created correctly."""
        # Tables should already be created by fixture
        # Check they exist by querying sqlite_master
        with temp_db._get_connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

        assert "transcription_metadata" in tables
        assert "transcription_segments" in tables

    def test_create_custom_tables(self):
        """Test creating tables with custom names."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = TranscriptionDatabase(db_path)
            db.create_tables(
                metadata_table="custom_metadata", segments_table="custom_segments"
            )

            with db._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = [row[0] for row in cursor.fetchall()]

            assert "custom_metadata" in tables
            assert "custom_segments" in tables

        finally:
            os.unlink(db_path)

    def test_save_transcription(self, temp_db, sample_result):
        """Test saving a transcription to the database."""
        transcription_id = temp_db.save_transcription(
            sample_result, model_name="whisper-base"
        )

        assert isinstance(transcription_id, int)
        assert transcription_id > 0

        # Verify it was saved
        metadata = temp_db.get_transcription_metadata(transcription_id)
        assert metadata is not None
        assert metadata.filename == "test_audio.mp3"
        assert metadata.file_hash == "hash123abc"
        assert metadata.language == "en"
        assert metadata.model_name == "whisper-base"

    def test_save_transcription_with_feed_data(self, temp_db, sample_result):
        """Test saving a transcription with feed metadata."""
        transcription_id = temp_db.save_transcription(
            sample_result,
            model_name="whisper-medium",
            feed_url="https://example.com/podcast/feed.xml",
            feed_item_id="episode-42",
            feed_item_title="Episode 42: The Answer",
            feed_item_published="2025-07-15T10:00:00Z",
        )

        metadata = temp_db.get_transcription_metadata(transcription_id)
        assert metadata.feed_url == "https://example.com/podcast/feed.xml"
        assert metadata.feed_item_id == "episode-42"
        assert metadata.feed_item_title == "Episode 42: The Answer"
        assert metadata.feed_item_published is not None

    def test_duplicate_prevention(self, temp_db, sample_result):
        """Test that duplicates are prevented."""
        # Save once
        id1 = temp_db.save_transcription(sample_result, model_name="base")

        # Try to save again with same hash and model
        id2 = temp_db.save_transcription(sample_result, model_name="base")

        # Should return the same ID
        assert id1 == id2

        # But different model should create new entry
        id3 = temp_db.save_transcription(sample_result, model_name="large")
        assert id3 != id1

    def test_get_transcription(self, temp_db, sample_result):
        """Test retrieving a complete transcription."""
        transcription_id = temp_db.save_transcription(sample_result, model_name="base")

        data = temp_db.get_transcription(transcription_id)

        assert data is not None
        assert "metadata" in data
        assert "segments" in data

        # Check metadata
        metadata = data["metadata"]
        assert metadata["filename"] == "test_audio.mp3"
        assert metadata["file_hash"] == "hash123abc"

        # Check segments
        segments = data["segments"]
        assert len(segments) == 3
        assert segments[0]["text"] == "First segment of the transcription."
        assert segments[0]["start_time"] == 0.0
        assert segments[0]["end_time"] == 5.0

    def test_get_all_transcriptions(self, temp_db):
        """Test getting all transcriptions."""
        # Save multiple transcriptions
        for i in range(3):
            result = TranscriptionResult(
                filename=f"audio_{i}.mp3",
                file_hash=f"hash_{i}",
                language="en",
                segments=[Segment(0.0, 5.0, f"Content {i}")],
                full_text=f"Content {i}",
            )
            temp_db.save_transcription(result, model_name="base")

        all_transcriptions = temp_db.get_all_transcriptions()

        assert len(all_transcriptions) == 3
        assert all(isinstance(t, TranscriptionMetadata) for t in all_transcriptions)

        # Check all filenames are present (order may vary due to same timestamps)
        filenames = [t.filename for t in all_transcriptions]
        assert set(filenames) == {"audio_0.mp3", "audio_1.mp3", "audio_2.mp3"}

    def test_search_transcriptions(self, temp_db):
        """Test searching transcriptions by text."""
        # Save transcriptions with different content
        results = [
            TranscriptionResult(
                filename="python_talk.mp3",
                file_hash="hash1",
                language="en",
                segments=[Segment(0.0, 5.0, "Python programming")],
                full_text="Today we discuss Python programming and its applications.",
            ),
            TranscriptionResult(
                filename="java_talk.mp3",
                file_hash="hash2",
                language="en",
                segments=[Segment(0.0, 5.0, "Java development")],
                full_text="Java development best practices and patterns.",
            ),
            TranscriptionResult(
                filename="python_advanced.mp3",
                file_hash="hash3",
                language="en",
                segments=[Segment(0.0, 5.0, "Advanced Python")],
                full_text="Advanced Python techniques and optimization.",
            ),
        ]

        for result in results:
            temp_db.save_transcription(result, model_name="base")

        # Search for Python
        python_results = temp_db.search_transcriptions("Python")
        assert len(python_results) == 2
        assert all("Python" in r.full_text for r in python_results)

        # Search for Java
        java_results = temp_db.search_transcriptions("Java")
        assert len(java_results) == 1
        assert java_results[0].filename == "java_talk.mp3"

    def test_delete_transcription(self, temp_db, sample_result):
        """Test deleting a transcription."""
        transcription_id = temp_db.save_transcription(sample_result, model_name="base")

        # Verify it exists
        assert temp_db.get_transcription(transcription_id) is not None

        # Delete it
        success = temp_db.delete_transcription(transcription_id)
        assert success is True

        # Verify it's gone
        assert temp_db.get_transcription(transcription_id) is None

        # Verify segments were also deleted (cascade)
        segments = temp_db.get_segments_for_transcription(transcription_id)
        assert len(segments) == 0

    def test_check_feed_item_exists(self, temp_db, sample_result):
        """Test checking if a feed item exists."""
        # Should not exist initially
        exists = temp_db.check_feed_item_exists(
            "https://example.com/feed.xml", "episode-1"
        )
        assert exists is False

        # Save with feed data
        temp_db.save_transcription(
            sample_result,
            model_name="base",
            feed_url="https://example.com/feed.xml",
            feed_item_id="episode-1",
        )

        # Now it should exist
        exists = temp_db.check_feed_item_exists(
            "https://example.com/feed.xml", "episode-1"
        )
        assert exists is True

        # Different feed URL should not exist
        exists = temp_db.check_feed_item_exists(
            "https://other.com/feed.xml", "episode-1"
        )
        assert exists is False

    def test_get_recent_transcriptions(self, temp_db):
        """Test getting recent transcriptions."""
        # Save multiple transcriptions
        for i in range(5):
            result = TranscriptionResult(
                filename=f"recent_{i}.mp3",
                file_hash=f"recent_hash_{i}",
                language="en",
                segments=[Segment(0.0, 1.0, f"Recent {i}")],
                full_text=f"Recent content {i}",
            )
            temp_db.save_transcription(result, model_name="base")

        # Get recent 3
        recent = temp_db.get_recent_transcriptions(limit=3)

        assert len(recent) == 3
        assert recent[0]["filename"] == "recent_4.mp3"
        assert recent[1]["filename"] == "recent_3.mp3"
        assert recent[2]["filename"] == "recent_2.mp3"

    def test_export_to_dict(self, temp_db, sample_result):
        """Test exporting a transcription back to TranscriptionResult."""
        transcription_id = temp_db.save_transcription(sample_result, model_name="base")

        exported = temp_db.export_to_dict(transcription_id)

        assert isinstance(exported, TranscriptionResult)
        assert exported.filename == sample_result.filename
        assert exported.file_hash == sample_result.file_hash
        assert exported.language == sample_result.language
        assert exported.full_text == sample_result.full_text
        assert len(exported.segments) == len(sample_result.segments)

        for orig, exp in zip(sample_result.segments, exported.segments):
            assert exp.start == orig.start
            assert exp.end == orig.end
            assert exp.text == orig.text

    def test_get_transcriptions_by_date_range(self, temp_db):
        """Test getting transcriptions by date range."""
        # Save transcriptions at different times
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        two_days_ago = now - timedelta(days=2)
        three_days_ago = now - timedelta(days=3)

        # We can't directly set created_at, so we'll use feed_item_published
        # as a proxy for testing date filtering
        for i, date in enumerate([three_days_ago, two_days_ago, yesterday, now]):
            result = TranscriptionResult(
                filename=f"dated_{i}.mp3",
                file_hash=f"dated_hash_{i}",
                language="en",
                segments=[Segment(0.0, 1.0, f"Dated {i}")],
                full_text=f"Dated content {i}",
            )
            temp_db.save_transcription(
                result,
                model_name="base",
                feed_url="https://example.com/feed.xml",
                feed_item_id=f"dated-{i}",
                feed_item_published=date.isoformat(),
            )

        # Get transcriptions from two days ago to yesterday
        recent = temp_db.get_transcriptions_by_date_range(
            start_date=(two_days_ago - timedelta(hours=1)).isoformat(),
            end_date=(yesterday + timedelta(hours=1)).isoformat(),
        )

        assert len(recent) == 2
        # Should include dated_1 (two days ago) and dated_2 (yesterday)
        filenames = [t.filename for t in recent]
        assert "dated_1.mp3" in filenames
        assert "dated_2.mp3" in filenames

    def test_get_full_transcription(self, temp_db, sample_result):
        """Test getting full transcription (alias for get_transcription)."""
        transcription_id = temp_db.save_transcription(sample_result, model_name="base")

        # get_full_transcription is an alias for get_transcription
        data = temp_db.get_full_transcription(transcription_id)

        assert data is not None
        assert "metadata" in data
        assert "segments" in data

        # This should be the same as get_transcription
        data2 = temp_db.get_transcription(transcription_id)
        assert data == data2

    def test_foreign_key_constraint(self, temp_db):
        """Test that foreign key constraints are enforced."""
        # Try to insert a segment with non-existent transcription_id
        with temp_db._get_connection() as conn:
            cursor = conn.cursor()

            with pytest.raises(sqlite3.IntegrityError):
                cursor.execute(
                    """
                    INSERT INTO transcription_segments
                    (transcription_id, segment_index, start_time, end_time, duration, text)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (99999, 0, 0.0, 5.0, 5.0, "Orphan segment"),
                )

    def test_unique_feed_constraint(self, temp_db, sample_result):
        """Test that feed URL + item ID unique constraint works."""
        # Save once
        temp_db.save_transcription(
            sample_result,
            model_name="base",
            feed_url="https://example.com/feed.xml",
            feed_item_id="unique-episode",
        )

        # Try to save again with same feed URL and item ID but different content
        different_result = TranscriptionResult(
            filename="different.mp3",
            file_hash="different_hash",
            language="es",
            segments=[Segment(0.0, 1.0, "Different")],
            full_text="Different content",
        )

        with temp_db._get_connection() as conn:
            cursor = conn.cursor()

            # This should raise an integrity error due to unique constraint
            with pytest.raises(sqlite3.IntegrityError):
                cursor.execute(
                    """
                    INSERT INTO transcription_metadata
                    (filename, file_hash, language, full_text, model_name,
                     feed_url, feed_item_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        different_result.filename,
                        different_result.file_hash,
                        different_result.language,
                        different_result.full_text,
                        "base",
                        "https://example.com/feed.xml",
                        "unique-episode",
                    ),
                )

    def test_empty_database_operations(self, temp_db):
        """Test operations on empty database."""
        # Get all should return empty list
        assert len(temp_db.get_all_transcriptions()) == 0

        # Get non-existent ID should return None
        assert temp_db.get_transcription(999) is None
        assert temp_db.get_transcription_metadata(999) is None
        assert temp_db.export_to_dict(999) is None

        # Delete non-existent should return False
        assert temp_db.delete_transcription(999) is False

        # Search should return empty list
        assert len(temp_db.search_transcriptions("anything")) == 0

        # Recent should return empty list
        assert len(temp_db.get_recent_transcriptions()) == 0


class TestDatabaseAsync:
    """Test async database operations."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db = TranscriptionDatabase(db_path)
        db.create_tables()

        yield db

        # Cleanup
        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_get_all_transcriptions_async(self, temp_db):
        """Test async version of get_all_transcriptions."""
        # Save some data first
        for i in range(3):
            result = TranscriptionResult(
                filename=f"async_{i}.mp3",
                file_hash=f"async_hash_{i}",
                language="en",
                segments=[Segment(0.0, 1.0, f"Async {i}")],
                full_text=f"Async content {i}",
            )
            temp_db.save_transcription(result, model_name="base")

        # Get async
        all_transcriptions = await temp_db.get_all_transcriptions_async()

        assert len(all_transcriptions) == 3
        assert all(isinstance(t, TranscriptionMetadata) for t in all_transcriptions)

    @pytest.mark.asyncio
    async def test_get_segments_async(self, temp_db):
        """Test async version of get_segments."""
        result = TranscriptionResult(
            filename="async_segments.mp3",
            file_hash="async_seg_hash",
            language="en",
            segments=[Segment(0.0, 2.0, "Segment 1"), Segment(2.0, 4.0, "Segment 2")],
            full_text="Segment 1 Segment 2",
        )

        transcription_id = temp_db.save_transcription(result, model_name="base")

        # Get segments async
        segments = await temp_db.get_segments_for_transcription_async(transcription_id)

        assert len(segments) == 2
        assert all(isinstance(s, TranscriptionSegment) for s in segments)
        assert segments[0].text == "Segment 1"
        assert segments[1].text == "Segment 2"
