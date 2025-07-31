"""
Tests for the embeddings module.
"""

from unittest.mock import Mock
from pathlib import Path

from grant.embeddings import EmbeddingService


class TestEmbeddingService:
    """Test the EmbeddingService class."""

    def test_service_initialization(self, mock_ollama_client, temp_dir):
        """Test service initialization."""
        # Without cache
        service = EmbeddingService(
            ollama_client=mock_ollama_client, model="nomic-embed-text", cache_dir=None
        )
        assert service.model == "nomic-embed-text"
        assert service.cache_dir is None

        # With cache
        cache_path = str(temp_dir / "cache")
        service = EmbeddingService(
            ollama_client=mock_ollama_client,
            model="nomic-embed-text",
            cache_dir=cache_path,
        )
        assert service.cache_dir == Path(cache_path)
        assert service.cache_dir.exists()

    def test_embed_text(self, mock_ollama_client):
        """Test embedding single text."""
        service = EmbeddingService(
            mock_ollama_client, "nomic-embed-text", cache_dir=None
        )

        embedding = service.embed_text("Test text", use_cache=False)

        # Should return the mock embedding
        assert embedding == [0.1] * 384
        mock_ollama_client.embeddings.assert_called_once_with(
            "nomic-embed-text", "Test text"
        )

    def test_embed_texts_batch(self, mock_ollama_client):
        """Test embedding multiple texts."""
        service = EmbeddingService(
            mock_ollama_client, "nomic-embed-text", cache_dir=None
        )

        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = service.embed_texts(texts, use_cache=False)

        # Should return embeddings for all texts
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
        assert mock_ollama_client.embeddings.call_count == 3

    def test_embed_with_cache(self, mock_ollama_client, temp_dir):
        """Test embedding with caching enabled."""
        cache_dir = temp_dir / "cache"
        service = EmbeddingService(
            mock_ollama_client, "nomic-embed-text", cache_dir=str(cache_dir)
        )

        # First call - should hit Ollama
        embedding1 = service.embed_text("Test text")
        assert mock_ollama_client.embeddings.call_count == 1

        # Check cache file was created
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) == 1

        # Second call - should hit cache
        embedding2 = service.embed_text("Test text")
        assert mock_ollama_client.embeddings.call_count == 1  # No additional call
        assert embedding1 == embedding2

    def test_embed_chunks(self, mock_ollama_client):
        """Test embedding chunks."""
        from grant.chunking import TextChunk

        service = EmbeddingService(
            mock_ollama_client, "nomic-embed-text", cache_dir=None
        )

        chunks = [
            TextChunk(id="1", text="Chunk 1", metadata={"index": 0}),
            TextChunk(id="2", text="Chunk 2", metadata={"index": 1}),
        ]

        embeddings = service.embed_chunks(chunks, use_cache=False)

        assert len(embeddings) == 2
        assert mock_ollama_client.embeddings.call_count == 2

    def test_parallel_embedding(self, mock_ollama_client):
        """Test parallel embedding of texts."""
        service = EmbeddingService(
            mock_ollama_client, "nomic-embed-text", cache_dir=None
        )

        texts = [f"Text {i}" for i in range(10)]
        embeddings = service.embed_texts_parallel(texts, max_workers=4, use_cache=False)

        assert len(embeddings) == 10
        assert mock_ollama_client.embeddings.call_count == 10

    def test_empty_text_handling(self, mock_ollama_client):
        """Test handling of empty text."""
        service = EmbeddingService(
            mock_ollama_client, "nomic-embed-text", cache_dir=None
        )

        # Empty string should still get an embedding
        embedding = service.embed_text("", use_cache=False)
        assert embedding == [0.1] * 384

        # Empty list should return empty list
        embeddings = service.embed_texts([], use_cache=False)
        assert embeddings == []

    def test_cache_key_generation(self, mock_ollama_client, temp_dir):
        """Test that cache keys are generated correctly."""
        cache_dir = temp_dir / "cache"
        service = EmbeddingService(
            mock_ollama_client, "nomic-embed-text", cache_dir=str(cache_dir)
        )

        # Embed some text
        service.embed_text("Test text")

        # Check that a cache file was created with correct naming
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) == 1

        # The filename should be a hash
        filename = cache_files[0].stem
        assert len(filename) == 64  # SHA256 hash length

    def test_cache_persistence(self, mock_ollama_client, temp_dir):
        """Test that cache persists across service instances."""
        cache_dir = temp_dir / "cache"

        # First service instance
        service1 = EmbeddingService(
            mock_ollama_client, "nomic-embed-text", cache_dir=str(cache_dir)
        )
        embedding1 = service1.embed_text("Test text")

        # Reset mock
        mock_ollama_client.reset_mock()
        mock_response = Mock()
        mock_response.embedding = [0.2] * 384  # Different embedding
        mock_ollama_client.embeddings.return_value = mock_response

        # Second service instance
        service2 = EmbeddingService(
            mock_ollama_client, "nomic-embed-text", cache_dir=str(cache_dir)
        )

        # Should use cached value, not the new mock
        embedding2 = service2.embed_text("Test text")
        assert mock_ollama_client.embeddings.call_count == 0
        assert embedding2 == embedding1  # Should be the original cached value
