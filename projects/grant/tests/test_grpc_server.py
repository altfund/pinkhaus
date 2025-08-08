"""Test cases for the gRPC server."""

import pytest
import grpc
import sys
from pathlib import Path

# Add grant to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinkhaus_models.proto.ominari import external_pb2, external_pb2_grpc
from grant.grpc_server import SignalServicer
from grant.ollama_client import OllamaClient


class TestGRPCServer:
    """Test the gRPC server functionality."""

    @pytest.fixture
    def mock_ollama_client(self, mocker):
        """Create a mocked Ollama client."""
        client = mocker.MagicMock(spec=OllamaClient)
        return client

    @pytest.fixture
    def servicer(self, mock_ollama_client, tmp_path):
        """Create a SignalServicer instance with mocked dependencies."""
        vector_store_path = str(tmp_path / "test_chroma_db")
        return SignalServicer(mock_ollama_client, vector_store_path)

    def test_signal_servicer_initialization(self, servicer):
        """Test that SignalServicer initializes correctly."""
        assert servicer.ollama_client is not None
        assert servicer.vector_store_path is not None
        assert servicer.default_config.llm_model == "llama3.2"
        assert servicer.default_config.embedding_model == "nomic-embed-text"

    def test_get_probabilities_single_request(self, servicer, mocker):
        """Test GetProbabilities with a single request."""
        # Mock the RAG pipeline
        mock_rag = mocker.patch("grant.grpc_server.RAGPipeline")
        mock_result = mocker.MagicMock()
        mock_result.answer = (
            "This is a test answer about inflation and economic conditions."
        )
        mock_rag.return_value.query.return_value = mock_result

        # Create a request
        request = external_pb2.SignalBatchRequest(
            requests=[
                external_pb2.SignalRequest(
                    source_id="test-1",
                    normalized_outcome="HOME_WIN",
                    as_of_time="2024-01-01T00:00:00Z",
                    query="What are the inflation trends?",
                    model="llama3.2",
                )
            ]
        )

        # Call the method
        response = servicer.GetProbabilities(request, None)

        # Verify response
        assert len(response.probabilities) == 1
        assert -1.0 == response.probabilities[0]

        # Verify RAG was called correctly
        assert mock_rag.call_count == 1
        mock_rag.return_value.query.assert_called_with(
            "What are the inflation trends? \n\n Remember, the answer must be one decimal number from 0 to 1. \n\n A semi-educated guess is fine."
        )

    def test_get_probabilities_multiple_requests(self, servicer, mocker):
        """Test GetProbabilities with multiple requests."""
        # Mock the RAG pipeline
        mock_rag = mocker.patch("grant.grpc_server.RAGPipeline")
        mock_result = mocker.MagicMock()
        mock_result.answer = "Test answer"
        mock_rag.return_value.query.return_value = mock_result

        # Create a request with multiple items
        requests = []
        for i in range(3):
            requests.append(
                external_pb2.SignalRequest(
                    source_id=f"test-{i}",
                    normalized_outcome="DRAW",
                    as_of_time="2024-01-01T00:00:00Z",
                    query=f"Query {i}",
                    model="mistral" if i == 1 else "",  # Test with different model
                )
            )

        request = external_pb2.SignalBatchRequest(requests=requests)

        # Call the method
        response = servicer.GetProbabilities(request, None)

        # Verify response
        assert len(response.probabilities) == 3
        for prob in response.probabilities:
            assert -1.0 == prob

        # Verify RAG was called for each request
        assert mock_rag.call_count == 3

    def test_get_probabilities_error_handling(self, servicer, mocker):
        """Test GetProbabilities handles errors gracefully."""
        # Mock the RAG pipeline to raise an exception
        mock_rag = mocker.patch("grant.grpc_server.RAGPipeline")
        mock_rag.return_value.query.side_effect = Exception("Test error")

        # Create a request
        request = external_pb2.SignalBatchRequest(
            requests=[
                external_pb2.SignalRequest(
                    source_id="test-error",
                    normalized_outcome="HOME_WIN",
                    as_of_time="2024-01-01T00:00:00Z",
                    query="This will cause an error",
                    model="llama3.2",
                )
            ]
        )

        # Call the method
        response = servicer.GetProbabilities(request, None)

        # Verify response returns 0.0 on error
        assert len(response.probabilities) == 1
        assert response.probabilities[0] == -1.0


def test_grpc_client_integration():
    """Integration test with actual gRPC client/server communication."""
    # This test requires the server to be running
    # It's marked as integration test and can be skipped in unit tests

    try:
        # Try to connect to a running server
        channel = grpc.insecure_channel("localhost:50051")
        stub = external_pb2_grpc.SignalServiceStub(channel)

        # Create a test request
        request = external_pb2.SignalBatchRequest(
            requests=[
                external_pb2.SignalRequest(
                    source_id="integration-test",
                    normalized_outcome="HOME_WIN",
                    as_of_time="2024-01-01T00:00:00Z",
                    query="What are the main economic indicators discussed?",
                    model="llama3.2",
                )
            ]
        )

        # Make the call with a timeout
        response = stub.GetProbabilities(request, timeout=5.0)

        # Verify we got a response
        assert len(response.probabilities) == 1
        assert 0.0 <= response.probabilities[0] <= 1.0

        print(f"Integration test passed. Got probability: {response.probabilities[0]}")

    except grpc.RpcError as e:
        # Server might not be running, skip the test
        pytest.skip(f"gRPC server not running: {e}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
