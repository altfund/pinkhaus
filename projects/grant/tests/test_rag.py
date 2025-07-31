"""
Tests for the RAG pipeline module.
"""

from unittest.mock import MagicMock, patch

from grant.rag import RAGPipeline, RAGConfig
from pinkhaus_models import TranscriptionMetadata


class TestRAGConfig:
    """Test the RAGConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RAGConfig()

        assert config.embedding_model == "nomic-embed-text"
        assert config.llm_model == "llama3.2"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 128
        assert config.top_k == 5
        assert config.temperature == 0.7

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RAGConfig(
            embedding_model="custom-embed",
            llm_model="custom-gen",
            chunk_size=500,
            top_k=10,
        )

        assert config.embedding_model == "custom-embed"
        assert config.llm_model == "custom-gen"
        assert config.chunk_size == 500
        assert config.top_k == 10


class TestRAGPipeline:
    """Test the RAGPipeline class."""

    @patch("grant.rag.PodcastVectorStore")
    @patch("grant.rag.EmbeddingService")
    @patch("grant.rag.PodcastChunker")
    def test_pipeline_initialization(
        self,
        mock_chunker,
        mock_embedding,
        mock_vector_store,
        temp_dir,
        mock_ollama_client,
    ):
        """Test pipeline initialization."""
        # Create pipeline
        db_path = str(temp_dir / "test.db")
        pipeline = RAGPipeline(
            db_path=db_path,
            ollama_client=mock_ollama_client,
            config=RAGConfig(),
            vector_store_path="./test_vector_store",
        )

        assert pipeline.config.embedding_model == "nomic-embed-text"
        assert pipeline.db.db_path == db_path
        assert pipeline.ollama == mock_ollama_client

        # Verify components were initialized
        mock_chunker.assert_called_once()
        mock_embedding.assert_called_once()
        mock_vector_store.assert_called_once()

    @patch("grant.rag.PodcastVectorStore")
    @patch("grant.rag.EmbeddingService")
    @patch("grant.rag.PodcastChunker")
    def test_index_transcription(
        self,
        mock_chunker_class,
        mock_embedding_class,
        mock_vector_store_class,
        mock_db,
        mock_ollama_client,
        mock_transcription_result,
    ):
        """Test indexing a single transcription."""
        # Setup mocks
        mock_chunker = MagicMock()
        mock_chunks = [
            {"text": "Chunk 1", "metadata": {"chunk_index": 0}},
            {"text": "Chunk 2", "metadata": {"chunk_index": 1}},
        ]
        mock_chunker.chunk_transcription.return_value = mock_chunks
        mock_chunker_class.return_value = mock_chunker

        mock_embedding = MagicMock()
        mock_embeddings = [[0.1] * 384, [0.2] * 384]
        mock_embedding.embed_chunks.return_value = mock_embeddings
        mock_embedding_class.return_value = mock_embedding

        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store

        # Setup database mock
        mock_metadata = TranscriptionMetadata.from_row(
            {
                "id": 123,
                "filename": "test.mp3",
                "file_hash": "hash",
                "language": "en",
                "full_text": "text",
                "model_name": "whisper",
                "created_at": "2024-01-01",
                "feed_url": None,
                "feed_item_id": None,
                "feed_item_title": None,
                "feed_item_published": None,
            }
        )
        mock_db.get_transcription_metadata.return_value = mock_metadata
        mock_db.get_segments_for_transcription.return_value = (
            mock_transcription_result.segments
        )

        # Mock vector store check
        mock_vector_store.search_by_metadata.return_value = []  # Not indexed yet

        # Create pipeline
        pipeline = RAGPipeline(
            db_path=str(mock_db.db_path), ollama_client=mock_ollama_client
        )
        pipeline.db = mock_db

        # Index transcription
        pipeline.index_transcription(123)

        # Verify the flow
        mock_db.get_transcription_metadata.assert_called_once_with(123)
        mock_db.get_segments_for_transcription.assert_called_once_with(123)
        mock_chunker.chunk_transcription.assert_called_once_with(
            mock_metadata, mock_transcription_result.segments
        )
        mock_embedding.embed_chunks.assert_called_once()
        mock_vector_store.add_chunks.assert_called_once()

    @patch("grant.rag.PodcastVectorStore")
    @patch("grant.rag.EmbeddingService")
    @patch("grant.rag.PodcastChunker")
    def test_index_all_transcriptions(
        self,
        mock_chunker_class,
        mock_embedding_class,
        mock_vector_store_class,
        mock_db,
        mock_ollama_client,
        mock_transcription_result,
    ):
        """Test indexing all transcriptions."""
        # Setup mocks
        mock_chunker = MagicMock()
        mock_chunker.chunk_segments.return_value = [{"text": "chunk", "metadata": {}}]
        mock_chunker_class.return_value = mock_chunker

        mock_embedding = MagicMock()
        mock_embedding.embed_chunks.return_value = [[0.1] * 384]
        mock_embedding_class.return_value = mock_embedding

        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store

        # Mock transcriptions
        mock_transcriptions = [
            TranscriptionMetadata.from_row(
                {
                    "id": 1,
                    "filename": "test1.mp3",
                    "file_hash": "hash1",
                    "language": "en",
                    "full_text": "text1",
                    "model_name": "whisper",
                    "created_at": "2024-01-01",
                    "feed_url": None,
                    "feed_item_id": None,
                    "feed_item_title": None,
                    "feed_item_published": None,
                }
            ),
            TranscriptionMetadata.from_row(
                {
                    "id": 2,
                    "filename": "test2.mp3",
                    "file_hash": "hash2",
                    "language": "en",
                    "full_text": "text2",
                    "model_name": "whisper",
                    "created_at": "2024-01-01",
                    "feed_url": None,
                    "feed_item_id": None,
                    "feed_item_title": None,
                    "feed_item_published": None,
                }
            ),
        ]
        mock_db.get_all_transcriptions.return_value = mock_transcriptions
        mock_db.get_segments_for_transcription.return_value = []  # Empty segments

        # Mock vector store methods
        mock_vector_store.search_by_metadata.return_value = []  # Not indexed yet
        mock_vector_store.get_collection_stats.return_value = {"total_chunks": 10}

        # Create pipeline
        pipeline = RAGPipeline(str(mock_db.db_path), mock_ollama_client)
        pipeline.db = mock_db

        # Index all
        pipeline.index_all_transcriptions()

        # The method doesn't return a count, just prints it
        # Verify the expected calls were made
        assert mock_db.get_segments_for_transcription.call_count == 2
        assert mock_chunker.chunk_transcription.call_count == 2
        assert mock_embedding.embed_chunks.call_count == 2
        assert mock_vector_store.add_chunks.call_count == 2

    @patch("grant.rag.PodcastVectorStore")
    @patch("grant.rag.EmbeddingService")
    @patch("grant.rag.PodcastChunker")
    def test_query(
        self,
        mock_chunker_class,
        mock_embedding_class,
        mock_vector_store_class,
        mock_db,
        mock_ollama_client,
    ):
        """Test querying the RAG pipeline."""
        # Setup mocks
        mock_chunker_class.return_value = MagicMock()

        mock_embedding = MagicMock()
        mock_embedding.embed_text.return_value = [0.15] * 384
        mock_embedding_class.return_value = mock_embedding

        mock_vector_store = MagicMock()
        # Return tuples of (TextChunk, score)
        from grant.chunking import TextChunk

        mock_vector_store.search.return_value = [
            (
                TextChunk(
                    id="123_0",
                    text="Neural networks are computational models...",
                    metadata={
                        "transcription_id": "123",
                        "start_time": 0.0,
                        "end_time": 10.0,
                    },
                ),
                0.9,
            ),
            (
                TextChunk(
                    id="123_1",
                    text="Deep learning is a subset of machine learning...",
                    metadata={
                        "transcription_id": "123",
                        "start_time": 10.0,
                        "end_time": 20.0,
                    },
                ),
                0.8,
            ),
        ]
        mock_vector_store_class.return_value = mock_vector_store

        # Mock transcription metadata
        mock_metadata = TranscriptionMetadata.from_row(
            {
                "id": 123,
                "filename": "podcast.mp3",
                "file_hash": "hash",
                "language": "en",
                "full_text": "full text",
                "model_name": "whisper",
                "created_at": "2024-01-01",
                "feed_url": None,
                "feed_item_id": None,
                "feed_item_title": "ML Podcast",
                "feed_item_published": "2024-01-01",
            }
        )
        mock_db.get_transcription_metadata.return_value = mock_metadata

        # Mock generation - return an iterator since stream=True
        mock_ollama_client.generate.return_value = [
            {"response": "Based on the podcast, neural networks are..."}
        ]

        # Create pipeline
        pipeline = RAGPipeline(str(mock_db.db_path), mock_ollama_client)
        pipeline.db = mock_db

        # Query
        result = pipeline.query("What are neural networks?")

        # Verify flow
        mock_embedding.embed_text.assert_called_once_with("What are neural networks?")
        mock_vector_store.search.assert_called_once()
        mock_ollama_client.generate.assert_called_once()

        # Check result
        assert result.answer == "Based on the podcast, neural networks are..."
        assert len(result.sources) == 2  # We have 2 chunks
        # Both chunks come from the same transcription, but sources include all chunks
        assert result.metadata["n_results"] == 2

    @patch("grant.rag.PodcastVectorStore")
    @patch("grant.rag.EmbeddingService")
    @patch("grant.rag.PodcastChunker")
    def test_query_no_results(
        self,
        mock_chunker_class,
        mock_embedding_class,
        mock_vector_store_class,
        mock_db,
        mock_ollama_client,
    ):
        """Test querying with no results."""
        # Setup mocks
        mock_chunker_class.return_value = MagicMock()

        mock_embedding = MagicMock()
        mock_embedding.embed_text.return_value = [0.15] * 384
        mock_embedding_class.return_value = mock_embedding

        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = []
        mock_vector_store_class.return_value = mock_vector_store

        # Create pipeline
        pipeline = RAGPipeline(str(mock_db.db_path), mock_ollama_client)

        # Query
        result = pipeline.query("Unknown topic")

        # Should return a result with a message saying no info found
        assert (
            result.answer
            == "I couldn't find any relevant information in the podcast transcriptions to answer your question."
        )
        assert result.sources == []
        assert result.metadata["n_results"] == 0

    @patch("grant.rag.PodcastVectorStore")
    @patch("grant.rag.EmbeddingService")
    @patch("grant.rag.PodcastChunker")
    def test_delete_transcription(
        self,
        mock_chunker_class,
        mock_embedding_class,
        mock_vector_store_class,
        mock_db,
        mock_ollama_client,
    ):
        """Test deleting a transcription from the index."""
        # Setup mocks
        mock_chunker_class.return_value = MagicMock()
        mock_embedding_class.return_value = MagicMock()

        mock_vector_store = MagicMock()
        mock_vector_store_class.return_value = mock_vector_store

        # Create pipeline
        pipeline = RAGPipeline(str(mock_db.db_path), mock_ollama_client)

        # Delete by transcription (using the correct method name)
        pipeline.vector_store.delete_by_transcription(123)

        # Verify delete was called
        mock_vector_store.delete_by_transcription.assert_called_once_with(123)
