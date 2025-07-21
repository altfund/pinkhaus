#!/usr/bin/env python3
"""
Example of using the beige_book library directly.

This example shows both the legacy direct API and the new request/response API.
"""

from beige_book import (
    # Legacy API
    AudioTranscriber,
    # New API
    TranscriptionService,
    create_file_request,
    create_feed_request,
    TranscriptionRequest,
    InputConfig,
    ProcessingConfig,
    OutputConfig,
    DatabaseConfig
)
import json


def legacy_api_example():
    """Example using the legacy direct API"""
    print("=== Legacy API Example ===\n")

    # Create a transcriber with the small model
    transcriber = AudioTranscriber(model_name="tiny")

    # Transcribe an audio file
    result = transcriber.transcribe_file("../../../resources/audio/harvard.wav")

    # Access the raw data
    print(f"File: {result.filename}")
    print(f"Hash: {result.file_hash}")
    print(f"Language: {result.language}")
    print(f"Number of segments: {len(result.segments)}")
    print()

    # Print first segment details
    if result.segments:
        first_segment = result.segments[0]
        print("First segment:")
        print(f"  Start: {first_segment.format_time(first_segment.start)}")
        print(f"  End: {first_segment.format_time(first_segment.end)}")
        print(f"  Text: {first_segment.text}")
        print()


def new_api_simple_example():
    """Example using the new request/response API - simple case"""
    print("\n=== New API - Simple Example ===\n")

    # Method 1: Using convenience function
    request = create_file_request(
        filename="../../../resources/audio/harvard.wav",
        model="tiny",
        format="json"
    )

    # Process the request
    service = TranscriptionService()
    response = service.process(request)

    # Handle the response
    if response.success:
        print("Transcription successful!")
        result = response.results[0]
        print(f"File: {result.filename}")
        print(f"Language: {result.language}")
        print(f"Segments: {len(result.segments)}")
        print(f"Text preview: {result.full_text[:100]}...")
    else:
        print("Transcription failed!")
        for error in response.errors:
            print(f"Error: {error.message}")


def new_api_advanced_example():
    """Example using the new request/response API - advanced features"""
    print("\n=== New API - Advanced Example ===\n")

    # Build request manually with all options
    request = TranscriptionRequest(
        input=InputConfig(type="file", source="../../../resources/audio/harvard.wav"),
        processing=ProcessingConfig(model="tiny", verbose=False),
        output=OutputConfig(
            format="sqlite",
            database=DatabaseConfig(
                db_path="example_transcriptions.db",
                metadata_table="transcriptions",
                segments_table="segments"
            )
        )
    )

    # Validate request before processing
    try:
        request.validate()
        print("Request validation passed")
    except ValueError as e:
        print(f"Request validation failed: {e}")
        return

    # Process
    service = TranscriptionService()
    response = service.process(request)

    if response.success:
        print("Saved to database successfully!")

    # Demonstrate JSON serialization (useful for REST APIs)
    print("\nRequest as JSON:")
    print(request.to_json()[:200] + "...")

    print("\nResponse summary:")
    if response.summary:
        print(f"  Processed in {response.summary.elapsed_time:.2f}s")


def feed_processing_example():
    """Example: Process RSS feeds"""
    print("\n=== Feed Processing Example ===\n")

    # Note: This example assumes you have a feeds.toml file
    request = create_feed_request(
        toml_path="feeds.toml",
        model="tiny",
        format="json",
        output_path="feed_transcriptions.json",
        limit=2,  # Only process 2 items per feed
        order="newest",
        verbose=True
    )

    service = TranscriptionService()
    response = service.process(request)

    if response.summary:
        print(f"Feed Processing Summary:")
        print(f"  Total items found: {response.summary.total_items}")
        print(f"  Successfully processed: {response.summary.processed}")
        print(f"  Skipped (already done): {response.summary.skipped}")
        print(f"  Failed: {response.summary.failed}")
        print(f"  Time: {response.summary.elapsed_time:.1f}s")


def api_integration_example():
    """Example: REST API integration pattern"""
    print("\n=== REST API Integration Pattern ===\n")

    # Simulate receiving JSON from an API endpoint
    api_request = {
        "input": {
            "type": "file",
            "source": "/path/to/audio.mp3"
        },
        "processing": {
            "model": "base",
            "verbose": False
        },
        "output": {
            "format": "json"
        }
    }

    # Convert to request object
    request = TranscriptionRequest.from_json(json.dumps(api_request))
    print(f"Received request for: {request.input.source}")

    # Process (would actually process here)
    # response = service.process(request)

    # Convert response to JSON for API response
    # api_response = json.loads(response.to_json())

    print("Request successfully parsed and ready for processing")
    print(f"Model: {request.processing.model}")
    print(f"Output format: {request.output.format}")


if __name__ == "__main__":
    # Run examples
    print("Beige Book Library Usage Examples")
    print("=" * 50)

    # Show legacy API usage
    try:
        legacy_api_example()
    except FileNotFoundError:
        print("Legacy example skipped - audio file not found")

    # Show new API usage
    try:
        new_api_simple_example()
    except FileNotFoundError:
        print("New API example skipped - audio file not found")

    # Show advanced features
    new_api_advanced_example()

    # Show API integration
    api_integration_example()

    print("\n" + "=" * 50)
    print("For feed processing example, ensure you have a feeds.toml file")
    print("and uncomment the feed_processing_example() call below:")
    # feed_processing_example()