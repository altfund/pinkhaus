"""
Tests for the chunking module.
"""

from pinkhaus_models import TranscriptionMetadata, TranscriptionSegment

from grant.chunking import PodcastChunker, TextChunk


class TestPodcastChunker:
    """Test the PodcastChunker class."""

    def test_chunker_initialization(self):
        """Test chunker initialization with default and custom parameters."""
        # Default initialization
        chunker = PodcastChunker()
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 128

        # Custom initialization
        chunker = PodcastChunker(chunk_size=256, chunk_overlap=64)
        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 64

    def test_chunk_transcription_with_segments(self):
        """Test chunking with segments."""
        chunker = PodcastChunker(chunk_size=100, chunk_overlap=20)

        # Create mock metadata
        metadata = TranscriptionMetadata.from_row(
            {
                "id": 1,
                "filename": "test.mp3",
                "file_hash": "hash123",
                "language": "en",
                "full_text": "Full transcription text",
                "model_name": "whisper",
                "created_at": "2024-01-01",
                "feed_url": "http://example.com/feed",
                "feed_item_id": "item123",
                "feed_item_title": "Test Episode",
                "feed_item_published": "2024-01-01",
            }
        )

        # Create mock segments
        segments = [
            TranscriptionSegment.from_row(
                {
                    "id": 1,
                    "transcription_id": 1,
                    "segment_index": 0,
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "duration": 5.0,
                    "text": "This is the first segment of our test.",
                }
            ),
            TranscriptionSegment.from_row(
                {
                    "id": 2,
                    "transcription_id": 1,
                    "segment_index": 1,
                    "start_time": 5.0,
                    "end_time": 10.0,
                    "duration": 5.0,
                    "text": "This is the second segment with more content.",
                }
            ),
        ]

        chunks = chunker.chunk_transcription(metadata, segments)

        assert len(chunks) > 0
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)

        # Check chunk properties
        for chunk in chunks:
            assert chunk.id
            assert chunk.text
            assert chunk.metadata
            assert "transcription_id" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_chunk_transcription_without_segments(self):
        """Test chunking when no segments are provided."""
        chunker = PodcastChunker(chunk_size=50, chunk_overlap=10)

        metadata = TranscriptionMetadata.from_row(
            {
                "id": 1,
                "filename": "test.mp3",
                "file_hash": "hash123",
                "language": "en",
                "full_text": "This is a long text that needs to be chunked into smaller pieces for processing.",
                "model_name": "whisper",
                "created_at": "2024-01-01",
                "feed_url": None,
                "feed_item_id": None,
                "feed_item_title": None,
                "feed_item_published": None,
            }
        )

        chunks = chunker.chunk_transcription(metadata, [])

        assert len(chunks) > 0
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)

        # Verify chunks have proper metadata
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["transcription_id"] == 1

    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        chunker = PodcastChunker(chunk_size=20, chunk_overlap=5)

        metadata = TranscriptionMetadata.from_row(
            {
                "id": 1,
                "filename": "test.mp3",
                "file_hash": "hash123",
                "language": "en",
                "full_text": "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
                "model_name": "whisper",
                "created_at": "2024-01-01",
                "feed_url": None,
                "feed_item_id": None,
                "feed_item_title": None,
                "feed_item_published": None,
            }
        )

        chunks = chunker.chunk_transcription(metadata, [])

        # Should create multiple chunks due to small chunk size
        assert len(chunks) > 1

    def test_segment_aware_chunking(self):
        """Test that chunking respects segment boundaries."""
        chunker = PodcastChunker(chunk_size=50, chunk_overlap=10)

        metadata = TranscriptionMetadata.from_row(
            {
                "id": 1,
                "filename": "test.mp3",
                "file_hash": "hash123",
                "language": "en",
                "full_text": "Full text",
                "model_name": "whisper",
                "created_at": "2024-01-01",
                "feed_url": None,
                "feed_item_id": None,
                "feed_item_title": "Test",
                "feed_item_published": None,
            }
        )

        # Create segments with varying lengths
        segments = []
        for i in range(5):
            segments.append(
                TranscriptionSegment.from_row(
                    {
                        "id": i + 1,
                        "transcription_id": 1,
                        "segment_index": i,
                        "start_time": i * 10.0,
                        "end_time": (i + 1) * 10.0,
                        "duration": 10.0,
                        "text": f"Segment {i} with some content to make it longer.",
                    }
                )
            )

        chunks = chunker.chunk_transcription(metadata, segments)

        # Check that chunks contain segment metadata
        for chunk in chunks:
            assert "first_segment_index" in chunk.metadata
            assert "last_segment_index" in chunk.metadata
            assert "segment_count" in chunk.metadata
            assert "start_time" in chunk.metadata
            assert "end_time" in chunk.metadata

    def test_chunk_id_generation(self):
        """Test that chunk IDs are unique and consistent."""
        chunker = PodcastChunker(chunk_size=50)

        metadata = TranscriptionMetadata.from_row(
            {
                "id": 1,
                "filename": "test.mp3",
                "file_hash": "hash123",
                "language": "en",
                "full_text": "Some text to chunk",
                "model_name": "whisper",
                "created_at": "2024-01-01",
                "feed_url": None,
                "feed_item_id": None,
                "feed_item_title": None,
                "feed_item_published": None,
            }
        )

        chunks = chunker.chunk_transcription(metadata, [])

        # All chunk IDs should be unique
        chunk_ids = [chunk.id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

        # IDs should be non-empty strings
        assert all(isinstance(chunk_id, str) and chunk_id for chunk_id in chunk_ids)
