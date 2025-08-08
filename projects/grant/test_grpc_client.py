#!/usr/bin/env python3
"""Test client for the gRPC SignalService."""
import grpc
import sys
from pathlib import Path

# Add grant to path
sys.path.insert(0, str(Path(__file__).parent))

from pinkhaus_models.proto.ominari import external_pb2, external_pb2_grpc


def test_signal_service():
    """Test the SignalService with sample requests."""
    # Create a channel to the server
    channel = grpc.insecure_channel('127.0.0.1:50051')
    stub = external_pb2_grpc.SignalServiceStub(channel)

    model = "deepseek-r1:8b"
    # Create test requests
    requests = [
        external_pb2.SignalRequest(
            source_id="test-1",
            normalized_outcome="HOME_WIN",
            as_of_time="2024-01-01T00:00:00Z",
            query="What are the chances that liverpool win their first home game against nottingham forest? Probability between 0 and 1 inclusive.",
            model=model
        ),
        external_pb2.SignalRequest(
            source_id="test-2",
            normalized_outcome="DRAW",
            as_of_time="2024-01-15T00:00:00Z",
            query="What are the chances that liverpool draw their first home game against nottingham forest. Probability between 0 and 1 inclusive",
            model=model
        ),
        external_pb2.SignalRequest(
            source_id="test-3",
            normalized_outcome="HOME_LOSE",
            as_of_time="2024-02-01T00:00:00Z",
            query="What are the chances that liverpool lose their first home game against nottingham forest. Probability between 0 and 1 inclusive",
            model=model
        )
    ]
    
    # Create batch request
    batch_request = external_pb2.SignalBatchRequest(requests=requests)
    
    try:
        print("Sending batch request to gRPC server...")
        print(f"Connecting to localhost:50051")
        print(f"Number of requests: {len(batch_request.requests)}")
        response = stub.GetProbabilities(batch_request, timeout=300.0)  # 5 minutes
        
        print(f"\nReceived {len(response.probabilities)} probabilities:")
        for i, (req, prob) in enumerate(zip(requests, response.probabilities)):
            print(f"\n{i+1}. Query: {req.query[:50]}...")
            print(f"   Model: {req.model or 'default'}")
            print(f"   Probability: {prob:.4f}")
            
    except grpc.RpcError as e:
        print(f"Error calling gRPC service: {e.code()} - {e.details()}")
        sys.exit(1)


if __name__ == "__main__":
    print("Testing gRPC SignalService...")
    print("Make sure the server is running with: uv run grant serve")
    print("-" * 60)
    test_signal_service()