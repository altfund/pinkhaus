"""
Tests for the vector store module.
"""

from unittest.mock import MagicMock, patch

from grant.vector_store import PodcastVectorStore
from grant.chunking import TextChunk


class TestPodcastVectorStore:
    """Test the PodcastVectorStore class."""

    @patch("chromadb.PersistentClient")
    def test_store_initialization(self, mock_chroma_client, temp_dir):
        """Test vector store initialization."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.side_effect = ValueError("Not found")
        mock_client.create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        store = PodcastVectorStore(
            persist_directory=str(temp_dir), collection_name="test_podcasts"
        )

        # Verify ChromaDB client was created with correct path
        mock_chroma_client.assert_called_once()
        call_args = mock_chroma_client.call_args
        assert call_args[1]["path"] == str(temp_dir)

        # Verify collection was created
        mock_client.create_collection.assert_called_once_with(
            name="test_podcasts",
            metadata={"description": "Podcast transcription segments"},
        )

        assert store.collection == mock_collection

    @patch("chromadb.PersistentClient")
    def test_add_chunks(self, mock_chroma_client):
        """Test adding chunks to the vector store."""
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        store = PodcastVectorStore()

        # Test data
        chunks = [
            TextChunk(
                id="123_0",
                text="First chunk about neural networks",
                metadata={
                    "transcription_id": 123,
                    "chunk_index": 0,
                    "start_time": 0.0,
                    "end_time": 10.0,
                },
            ),
            TextChunk(
                id="123_1",
                text="Second chunk about deep learning",
                metadata={
                    "transcription_id": 123,
                    "chunk_index": 1,
                    "start_time": 10.0,
                    "end_time": 20.0,
                },
            ),
        ]
        embeddings = [[0.1] * 384, [0.2] * 384]

        # Add chunks
        store.add_chunks(chunks, embeddings)

        # Verify add was called with correct parameters
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]

        assert call_args["ids"] == ["123_0", "123_1"]
        assert call_args["documents"] == [
            "First chunk about neural networks",
            "Second chunk about deep learning",
        ]
        assert call_args["embeddings"] == embeddings
        assert len(call_args["metadatas"]) == 2

    @patch("chromadb.PersistentClient")
    def test_search(self, mock_chroma_client):
        """Test searching the vector store."""
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["123_0", "123_1", "456_0"]],
            "documents": [
                ["Neural networks...", "Deep learning...", "Machine learning..."]
            ],
            "metadatas": [
                [
                    {"transcription_id": 123, "start_time": 0.0, "end_time": 10.0},
                    {"transcription_id": 123, "start_time": 10.0, "end_time": 20.0},
                    {"transcription_id": 456, "start_time": 0.0, "end_time": 15.0},
                ]
            ],
            "distances": [[0.1, 0.2, 0.3]],
        }
        mock_client.get_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        store = PodcastVectorStore()

        # Perform search
        query_embedding = [0.15] * 384
        results = store.search(query_embedding, n_results=3)

        # Verify query was called correctly
        mock_collection.query.assert_called_once_with(
            query_embeddings=[query_embedding],
            n_results=3,
            where=None,
            include=["documents", "metadatas", "distances"],
        )

        # Check results
        assert len(results) == 3
        assert results[0][0].text == "Neural networks..."
        assert results[0][0].metadata["transcription_id"] == 123
        assert 0.9 <= results[0][1] <= 1.0  # Similarity score

    @patch("chromadb.PersistentClient")
    def test_delete_by_transcription(self, mock_chroma_client):
        """Test deleting all chunks for a transcription."""
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        store = PodcastVectorStore()

        # Delete transcription
        store.delete_by_transcription(123)

        # Verify delete was called with correct filter
        mock_collection.delete.assert_called_once_with(where={"transcription_id": 123})

    @patch("chromadb.PersistentClient")
    def test_get_collection_stats(self, mock_chroma_client):
        """Test getting collection statistics."""
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_client.get_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        store = PodcastVectorStore()

        stats = store.get_collection_stats()

        assert stats["total_chunks"] == 42
        assert stats["collection_name"] == "podcast_segments"
        mock_collection.count.assert_called_once()

    @patch("chromadb.PersistentClient")
    def test_empty_search_results(self, mock_chroma_client):
        """Test handling empty search results."""
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_client.get_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        store = PodcastVectorStore()

        results = store.search([0.1] * 384)

        assert results == []

    @patch("chromadb.PersistentClient")
    def test_add_empty_chunks(self, mock_chroma_client):
        """Test adding empty chunks list."""
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        store = PodcastVectorStore()

        # Add empty chunks
        store.add_chunks([], [])

        # Should not call add
        mock_collection.add.assert_not_called()
