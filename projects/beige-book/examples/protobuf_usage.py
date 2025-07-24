#!/usr/bin/env python3
"""
Example: Using Protocol Buffers for TranscriptionResult.

This demonstrates:
1. Using protobuf serialization with standard TranscriptionResult
2. Binary serialization/deserialization
3. Network-friendly data exchange
4. Extended results with feed metadata
"""

import base64
from beige_book import (
    AudioTranscriber,
    TranscriptionResult,
    TranscriptionDatabase,
    Segment,
    create_extended_result
)


def basic_protobuf_usage():
    """Example: Basic protobuf transcription and serialization"""
    print("=== Basic Protobuf Usage ===\n")
    
    # Create standard transcriber
    transcriber = AudioTranscriber(model_name="tiny")
    
    # Transcribe a file (simulated - would use real file in practice)
    # result = transcriber.transcribe_file("audio.wav")
    
    # Create a sample result for demonstration
    result = TranscriptionResult()
    result.filename = "sample_audio.wav"
    result.file_hash = "abc123def456"
    result.language = "en"
    result.full_text = "This is a sample transcription using protocol buffers."
    result.add_segment(0.0, 2.5, "This is a sample transcription")
    result.add_segment(2.5, 5.0, "using protocol buffers.")
    
    # Serialize to protobuf bytes
    proto_bytes = result.to_protobuf_bytes()
    print(f"Serialized size: {len(proto_bytes)} bytes")
    
    # Base64 encode for transmission
    encoded = result.to_protobuf_base64()
    print(f"Base64 encoded: {encoded[:50]}...")
    
    # Deserialize back
    restored = TranscriptionResult.from_protobuf_base64(encoded)
    
    print(f"\nRestored transcription:")
    print(f"  Filename: {restored.filename}")
    print(f"  Language: {restored.language}")
    print(f"  Segments: {len(restored.segments)}")
    print(f"  Text: {restored.full_text}")


def network_exchange_example():
    """Example: Simulating network data exchange"""
    print("\n=== Network Exchange Example ===\n")
    
    # Create a result
    result = TranscriptionResult()
    result.filename = "podcast_episode.mp3"
    result.file_hash = "xyz789"
    result.language = "en"
    result.full_text = "Welcome to our podcast. Today we discuss AI."
    result.add_segment(0.0, 3.0, "Welcome to our podcast.")
    result.add_segment(3.0, 6.0, "Today we discuss AI.")
    
    # Serialize for network transmission
    data_to_send = result.to_protobuf_bytes()
    print(f"Sending {len(data_to_send)} bytes over network...")
    
    # Simulate receiving on the other end
    received_data = data_to_send  # In real scenario, this comes from network
    
    # Deserialize
    received_result = TranscriptionResult.from_protobuf_bytes(received_data)
    print(f"Received transcription for: {received_result.filename}")
    
    # Convert to JSON if needed for API response
    json_response = received_result.to_json()
    print(f"JSON response:\n{json_response}")


def extended_result_example():
    """Example: Using extended results with feed metadata"""
    print("\n=== Extended Result with Feed Metadata ===\n")
    
    # Create a transcription result
    result = TranscriptionResult()
    result.filename = "episode_123.mp3"
    result.file_hash = "hash123"
    result.language = "en"
    result.full_text = "This is episode 123 of our tech podcast."
    result.add_segment(0.0, 5.0, "This is episode 123 of our tech podcast.")
    
    # Create extended result with feed metadata
    extended = create_extended_result(
        result,
        feed_url="https://example.com/podcast.xml",
        item_id="episode-123",
        title="Episode 123: AI and the Future",
        audio_url="https://example.com/episodes/123.mp3",
        published="2025-07-22T10:00:00Z"
    )
    
    # Serialize the extended result
    extended_bytes = extended.SerializeToString()
    print(f"Extended result size: {len(extended_bytes)} bytes")
    
    # Access the data
    print(f"Transcription filename: {extended.transcription.filename}")
    print(f"Feed URL: {extended.feed_metadata.feed_url}")
    print(f"Episode title: {extended.feed_metadata.title}")


def database_with_protobuf_example():
    """Example: Saving protobuf results to database"""
    print("\n=== Database with Protobuf Example ===\n")
    
    # Create a result
    result = TranscriptionResult()
    result.filename = "interview.wav"
    result.file_hash = "interview_hash"
    result.language = "en"
    result.full_text = "This is an interview about technology."
    result.add_segment(0.0, 4.0, "This is an interview about technology.")
    
    # The TranscriptionDatabase expects the wrapper class
    # So we can still use it with our protobuf version
    db = TranscriptionDatabase("protobuf_transcriptions.db")
    db.create_tables()
    
    # Save to database (the wrapper class handles conversion)
    trans_id = db.save_transcription(result, model_name="base")
    print(f"Saved to database with ID: {trans_id}")
    
    # Store the raw protobuf bytes separately if needed
    proto_bytes = result.to_protobuf_bytes()
    print(f"Protobuf bytes could be stored in a blob column: {len(proto_bytes)} bytes")


def format_comparison_example():
    """Example: Compare different serialization formats"""
    print("\n=== Format Comparison ===\n")
    
    # Create a result with multiple segments
    result = TranscriptionResult()
    result.filename = "comparison.wav"
    result.file_hash = "compare123"
    result.language = "en"
    result.full_text = "This is segment one. This is segment two. This is segment three."
    result.add_segment(0.0, 2.0, "This is segment one.")
    result.add_segment(2.0, 4.0, "This is segment two.")
    result.add_segment(4.0, 6.0, "This is segment three.")
    
    # Compare sizes
    proto_bytes = result.to_protobuf_bytes()
    json_str = result.to_json()
    toml_str = result.to_toml()
    
    print(f"Serialization format sizes:")
    print(f"  Protocol Buffers: {len(proto_bytes)} bytes")
    print(f"  JSON: {len(json_str)} bytes")
    print(f"  TOML: {len(toml_str)} bytes")
    print(f"  Protobuf is {len(json_str) / len(proto_bytes):.1f}x smaller than JSON")
    
    # Show protobuf format output
    print(f"\nProtobuf format (base64): {result.to_protobuf_base64()[:80]}...")


if __name__ == "__main__":
    print("Protocol Buffers Usage Examples")
    print("=" * 50)
    
    # Run examples
    basic_protobuf_usage()
    network_exchange_example()
    extended_result_example()
    database_with_protobuf_example()
    format_comparison_example()
    
    print("\n\nSummary:")
    print("- Use standard TranscriptionResult with protobuf methods")
    print("- to_protobuf_bytes() / from_protobuf_bytes() for binary serialization")
    print("- Protobuf is more compact than JSON/TOML")
    print("- create_extended_result() for adding feed metadata")
    print("- Compatible with existing database operations")