"""
Test the data models in pinkhaus-models.
"""

import pytest
from datetime import datetime
from pinkhaus_models.models import (
    Segment,
    TranscriptionResult,
    TranscriptionMetadata,
    TranscriptionSegment,
)


class TestSegment:
    """Test the Segment model."""

    def test_segment_creation(self):
        """Test creating a segment with basic properties."""
        segment = Segment(start=0.0, end=5.5, text="Hello world")

        assert segment.start == 0.0
        assert segment.end == 5.5
        assert segment.text == "Hello world"
        assert segment.duration == 5.5

    def test_segment_milliseconds(self):
        """Test millisecond conversion properties."""
        segment = Segment(start=1.234, end=5.678, text="Test")

        assert segment.start_ms == 1234
        assert segment.end_ms == 5678

    def test_segment_to_dict(self):
        """Test converting segment to dictionary."""
        segment = Segment(start=0.0, end=10.0, text="  Test segment  ")
        result = segment.to_dict()

        assert result["start"] == 0.0
        assert result["end"] == 10.0
        assert result["text"] == "Test segment"  # Should be stripped
        assert result["duration"] == 10.0

    def test_segment_duration_calculation(self):
        """Test that duration is calculated correctly."""
        segment = Segment(start=2.5, end=7.3, text="Test")
        assert segment.duration == pytest.approx(4.8, rel=1e-9)


class TestTranscriptionResult:
    """Test the TranscriptionResult model."""

    def test_transcription_result_creation(self):
        """Test creating a transcription result."""
        segments = [
            Segment(0.0, 5.0, "First segment"),
            Segment(5.0, 10.0, "Second segment"),
        ]

        result = TranscriptionResult(
            filename="test.mp3",
            file_hash="abc123",
            language="en",
            segments=segments,
            full_text="First segment Second segment",
        )

        assert result.filename == "test.mp3"
        assert result.file_hash == "abc123"
        assert result.language == "en"
        assert len(result.segments) == 2
        assert result.full_text == "First segment Second segment"

    def test_transcription_result_to_dict(self):
        """Test converting transcription result to dictionary."""
        segments = [Segment(0.0, 3.0, "Hello"), Segment(3.0, 5.0, "World")]

        result = TranscriptionResult(
            filename="audio.wav",
            file_hash="hash123",
            language="en",
            segments=segments,
            full_text="Hello World",
        )

        data = result.to_dict()

        assert data["filename"] == "audio.wav"
        assert data["file_hash"] == "hash123"
        assert data["language"] == "en"
        assert data["full_text"] == "Hello World"
        assert len(data["segments"]) == 2
        assert data["segments"][0]["text"] == "Hello"
        assert data["segments"][0]["duration"] == 3.0

    def test_transcription_result_from_dict(self):
        """Test creating transcription result from dictionary."""
        data = {
            "filename": "test.mp3",
            "file_hash": "xyz789",
            "language": "es",
            "full_text": "Hola Mundo",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hola"},
                {"start": 2.0, "end": 4.0, "text": "Mundo"},
            ],
        }

        result = TranscriptionResult.from_dict(data)

        assert result.filename == "test.mp3"
        assert result.file_hash == "xyz789"
        assert result.language == "es"
        assert result.full_text == "Hola Mundo"
        assert len(result.segments) == 2
        assert result.segments[0].text == "Hola"
        assert result.segments[1].text == "Mundo"

    def test_round_trip_dict_conversion(self):
        """Test that to_dict and from_dict are inverses."""
        original = TranscriptionResult(
            filename="round_trip.mp3",
            file_hash="rt123",
            language="fr",
            segments=[Segment(0.0, 1.5, "Bonjour"), Segment(1.5, 3.0, "le monde")],
            full_text="Bonjour le monde",
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = TranscriptionResult.from_dict(data)

        assert restored.filename == original.filename
        assert restored.file_hash == original.file_hash
        assert restored.language == original.language
        assert restored.full_text == original.full_text
        assert len(restored.segments) == len(original.segments)

        for orig_seg, rest_seg in zip(original.segments, restored.segments):
            assert rest_seg.start == orig_seg.start
            assert rest_seg.end == orig_seg.end
            assert rest_seg.text == orig_seg.text


class TestTranscriptionMetadata:
    """Test the TranscriptionMetadata model."""

    def test_metadata_creation_minimal(self):
        """Test creating metadata with minimal fields."""
        metadata = TranscriptionMetadata()

        assert metadata.id is None
        assert metadata.filename == ""
        assert metadata.file_hash == ""
        assert metadata.language == ""
        assert metadata.full_text == ""
        assert metadata.model_name is None
        assert metadata.feed_url is None
        assert metadata.created_at is None

    def test_metadata_creation_full(self):
        """Test creating metadata with all fields."""
        now = datetime.now()
        metadata = TranscriptionMetadata(
            id=1,
            filename="episode.mp3",
            file_hash="hash456",
            language="en",
            full_text="Full transcription text",
            model_name="whisper-base",
            feed_url="https://example.com/feed.xml",
            feed_item_id="episode-1",
            feed_item_title="Episode 1: Introduction",
            feed_item_published=now,
            created_at=now,
        )

        assert metadata.id == 1
        assert metadata.filename == "episode.mp3"
        assert metadata.file_hash == "hash456"
        assert metadata.language == "en"
        assert metadata.full_text == "Full transcription text"
        assert metadata.model_name == "whisper-base"
        assert metadata.feed_url == "https://example.com/feed.xml"
        assert metadata.feed_item_id == "episode-1"
        assert metadata.feed_item_title == "Episode 1: Introduction"
        assert metadata.feed_item_published == now
        assert metadata.created_at == now

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        now = datetime.now()
        metadata = TranscriptionMetadata(
            id=42,
            filename="test.wav",
            file_hash="testhash",
            language="de",
            full_text="Test text",
            model_name="medium",
            feed_url="https://feed.example.com",
            feed_item_id="item-1",
            feed_item_title="Test Episode",
            feed_item_published=now,
            created_at=now,
        )

        data = metadata.to_dict()

        assert data["id"] == 42
        assert data["filename"] == "test.wav"
        assert data["file_hash"] == "testhash"
        assert data["language"] == "de"
        assert data["full_text"] == "Test text"
        assert data["model_name"] == "medium"
        assert data["feed_url"] == "https://feed.example.com"
        assert data["feed_item_id"] == "item-1"
        assert data["feed_item_title"] == "Test Episode"
        assert data["feed_item_published"] == now.isoformat()
        assert data["created_at"] == now.isoformat()

    def test_metadata_to_dict_with_none_dates(self):
        """Test that None dates are handled correctly."""
        metadata = TranscriptionMetadata(
            filename="test.mp3", file_hash="hash", language="en", full_text="text"
        )

        data = metadata.to_dict()

        assert data["feed_item_published"] is None
        assert data["created_at"] is None

    def test_metadata_from_row(self):
        """Test creating metadata from database row."""
        row_data = {
            "id": 5,
            "filename": "podcast.mp3",
            "file_hash": "podhash",
            "language": "en",
            "full_text": "Podcast content",
            "model_name": "large",
            "feed_url": "https://podcast.com/feed",
            "feed_item_id": "pod-1",
            "feed_item_title": "Episode Title",
            "feed_item_published": "2025-07-15T10:00:00",
            "created_at": "2025-07-15T11:00:00",
        }

        metadata = TranscriptionMetadata.from_row(row_data)

        assert metadata.id == 5
        assert metadata.filename == "podcast.mp3"
        assert metadata.file_hash == "podhash"
        assert metadata.language == "en"
        assert metadata.full_text == "Podcast content"
        assert metadata.model_name == "large"
        assert metadata.feed_url == "https://podcast.com/feed"
        assert metadata.feed_item_id == "pod-1"
        assert metadata.feed_item_title == "Episode Title"
        assert isinstance(metadata.feed_item_published, datetime)
        assert isinstance(metadata.created_at, datetime)

    def test_metadata_from_row_with_missing_fields(self):
        """Test creating metadata from row with missing optional fields."""
        row_data = {
            "id": 1,
            "filename": "audio.wav",
            "file_hash": "hash123",
            "language": "en",
            "full_text": "Text content",
        }

        metadata = TranscriptionMetadata.from_row(row_data)

        assert metadata.id == 1
        assert metadata.filename == "audio.wav"
        assert metadata.file_hash == "hash123"
        assert metadata.language == "en"
        assert metadata.full_text == "Text content"
        assert metadata.model_name is None
        assert metadata.feed_url is None
        assert metadata.created_at is None


class TestTranscriptionSegment:
    """Test the TranscriptionSegment model."""

    def test_segment_creation(self):
        """Test creating a transcription segment."""
        segment = TranscriptionSegment(
            id=1,
            transcription_id=42,
            segment_index=0,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            text="Segment text",
        )

        assert segment.id == 1
        assert segment.transcription_id == 42
        assert segment.segment_index == 0
        assert segment.start_time == 0.0
        assert segment.end_time == 5.0
        assert segment.duration == 5.0
        assert segment.text == "Segment text"

    def test_segment_defaults(self):
        """Test segment default values."""
        segment = TranscriptionSegment()

        assert segment.id is None
        assert segment.transcription_id == 0
        assert segment.segment_index == 0
        assert segment.start_time == 0.0
        assert segment.end_time == 0.0
        assert segment.duration == 0.0
        assert segment.text == ""

    def test_segment_to_dict(self):
        """Test converting segment to dictionary."""
        segment = TranscriptionSegment(
            id=10,
            transcription_id=5,
            segment_index=3,
            start_time=10.5,
            end_time=15.3,
            duration=4.8,
            text="Test segment content",
        )

        data = segment.to_dict()

        assert data["id"] == 10
        assert data["transcription_id"] == 5
        assert data["segment_index"] == 3
        assert data["start_time"] == 10.5
        assert data["end_time"] == 15.3
        assert data["duration"] == 4.8
        assert data["text"] == "Test segment content"

    def test_segment_to_segment(self):
        """Test converting TranscriptionSegment to Segment."""
        trans_segment = TranscriptionSegment(
            id=1,
            transcription_id=1,
            segment_index=0,
            start_time=2.5,
            end_time=7.5,
            duration=5.0,
            text="Convert me",
        )

        segment = trans_segment.to_segment()

        assert isinstance(segment, Segment)
        assert segment.start == 2.5
        assert segment.end == 7.5
        assert segment.text == "Convert me"
        assert segment.duration == 5.0

    def test_segment_from_row(self):
        """Test creating segment from database row."""
        row_data = {
            "id": 20,
            "transcription_id": 10,
            "segment_index": 5,
            "start_time": 30.0,
            "end_time": 35.0,
            "duration": 5.0,
            "text": "Row segment",
        }

        segment = TranscriptionSegment.from_row(row_data)

        assert segment.id == 20
        assert segment.transcription_id == 10
        assert segment.segment_index == 5
        assert segment.start_time == 30.0
        assert segment.end_time == 35.0
        assert segment.duration == 5.0
        assert segment.text == "Row segment"

    def test_segment_from_row_with_defaults(self):
        """Test creating segment from row with missing fields."""
        row_data = {"id": 1}

        segment = TranscriptionSegment.from_row(row_data)

        assert segment.id == 1
        assert segment.transcription_id == 0
        assert segment.segment_index == 0
        assert segment.start_time == 0.0
        assert segment.end_time == 0.0
        assert segment.duration == 0.0
        assert segment.text == ""
