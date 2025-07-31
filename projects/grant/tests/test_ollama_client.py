"""
Tests for the Ollama client module.
"""

import pytest
from unittest.mock import Mock, patch
import httpx

from grant.ollama_client import (
    OllamaClient,
    OllamaGenerateRequest,
    OllamaEmbeddingRequest,
    OllamaError,
)


class TestOllamaClient:
    """Test the OllamaClient class."""

    def test_client_initialization(self):
        """Test client initialization with default and custom base URL."""
        # Default initialization
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"

        # Custom URL
        custom_url = "http://custom:8080"
        client = OllamaClient(base_url=custom_url)
        assert client.base_url == custom_url

    @patch("httpx.Client.post")
    def test_generate_success(self, mock_post):
        """Test successful text generation."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "model": "llama3.2",
            "created_at": "2024-01-01T00:00:00Z",
            "response": "Hello, world!",
            "done": True,
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()
        response = client.generate("llama3.2", "Say hello")

        assert response.response == "Hello, world!"
        assert response.model == "llama3.2"
        assert response.done is True

        # Verify request
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/generate"
        assert call_args[1]["json"]["model"] == "llama3.2"
        assert call_args[1]["json"]["prompt"] == "Say hello"

    @patch("httpx.Client.post")
    def test_embeddings_success(self, mock_post):
        """Test successful embedding generation."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()
        response = client.embeddings("nomic-embed-text", "Test text")

        assert response.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

        # Verify request
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/embeddings"
        assert call_args[1]["json"]["model"] == "nomic-embed-text"
        assert call_args[1]["json"]["prompt"] == "Test text"

    @patch("httpx.Client.post")
    def test_generate_with_system_prompt(self, mock_post):
        """Test generation with system prompt."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "model": "llama3.2",
            "created_at": "2024-01-01T00:00:00Z",
            "response": "I am a helpful assistant.",
            "done": True,
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = OllamaClient()
        response = client.generate(
            "llama3.2", "Who are you?", system="You are a helpful assistant."
        )

        # Verify system prompt was included
        call_args = mock_post.call_args
        assert call_args[1]["json"]["system"] == "You are a helpful assistant."

    @patch("httpx.Client.post")
    def test_http_error_handling(self, mock_post):
        """Test handling of HTTP errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=Mock()
        )
        mock_post.return_value = mock_response

        client = OllamaClient()

        with pytest.raises(httpx.HTTPStatusError):
            client.generate("llama3.2", "Test")

    @patch("httpx.Client.post")
    def test_connection_error_handling(self, mock_post):
        """Test handling of connection errors."""
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        client = OllamaClient()

        with pytest.raises(OllamaError) as exc_info:
            client.generate("llama3.2", "Test")

        assert "Failed to connect to Ollama" in str(exc_info.value)

    def test_generate_request_validation(self):
        """Test OllamaGenerateRequest validation."""
        # Valid request
        request = OllamaGenerateRequest(model="llama3.2", prompt="Test prompt")
        assert request.model == "llama3.2"
        assert request.prompt == "Test prompt"
        assert request.stream is False  # default

        # With options
        request = OllamaGenerateRequest(
            model="llama3.2", prompt="Test", temperature=0.7, top_k=40
        )
        data = request.model_dump()
        assert data["options"]["temperature"] == 0.7
        assert data["options"]["top_k"] == 40

    def test_embedding_request_validation(self):
        """Test OllamaEmbeddingRequest validation."""
        request = OllamaEmbeddingRequest(model="nomic-embed-text", prompt="Test text")
        assert request.model == "nomic-embed-text"
        assert request.prompt == "Test text"

    @patch("httpx.Client.post")
    def test_embeddings_connection_error(self, mock_post):
        """Test handling of connection errors for embeddings."""
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        client = OllamaClient()

        with pytest.raises(OllamaError) as exc_info:
            client.embeddings("nomic-embed-text", "Test")

        assert "Failed to connect to Ollama" in str(exc_info.value)
