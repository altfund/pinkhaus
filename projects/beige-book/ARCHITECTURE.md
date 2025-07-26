# Beige Book Architecture

## Overview

The beige-book library has been refactored to use a clean request/response architecture that provides a serializable boundary between different interfaces (CLI, REST API, etc.) and the core transcription functionality.

## Architecture Components

### 1. Request/Response Models (`proto_models.py` and `models_betterproto.py`)

The system uses betterproto-generated dataclasses from Protocol Buffer definitions that are fully serializable:

- **TranscriptionRequest**: The main request object containing:
  - `InputConfig`: Specifies the input source (file or feed)
  - `ProcessingConfig`: Model selection, verbosity, feed options
  - `OutputConfig`: Output format and destination

- **TranscriptionResponse**: The response object containing:
  - `success`: Boolean indicating overall success
  - `results`: List of transcription results
  - `errors`: List of processing errors
  - `summary`: Processing statistics (for batch operations)

### 2. Service Layer (`service.py`)

- **TranscriptionService**: Main service class that:
  - Accepts `TranscriptionRequest` objects
  - Orchestrates the transcription process
  - Returns `TranscriptionResponse` objects
  - Handles both single files and RSS feed processing

- **OutputFormatter**: Helper for formatting results based on output requirements

### 3. Core Components

- **AudioTranscriber**: Whisper model wrapper for actual transcription
- **FeedParser**: RSS/Atom feed parsing with retry logic
- **AudioDownloader**: Downloads audio files with retry and progress tracking
- **TranscriptionDatabase**: SQLite storage for results and resumability

### 4. CLI Interface (`cli.py`)

The CLI now:
1. Parses command-line arguments
2. Converts them to a `TranscriptionRequest`
3. Passes the request to `TranscriptionService`
4. Handles the response appropriately

## Usage Examples

### Library Usage

```python
from beige_book import TranscriptionService, create_file_request

# Create a request
request = create_file_request(
    filename="audio.mp3",
    model="base",
    format="json",
    output_path="output.json"
)

# Process it
service = TranscriptionService()
response = service.process(request)

# Handle response
if response.success:
    for result in response.results:
        print(f"Transcribed: {result.filename}")
```

### REST API Integration

```python
from flask import Flask, request, jsonify
from beige_book import TranscriptionService, TranscriptionRequest

app = Flask(__name__)
service = TranscriptionService()

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Parse request from JSON
    transcription_request = TranscriptionRequest.from_json(
        request.get_json()
    )
    
    # Process
    response = service.process(transcription_request)
    
    # Return JSON response
    return jsonify(json.loads(response.to_json()))
```

### Feed Processing with Resumability

```python
from beige_book import create_feed_request, TranscriptionService

request = create_feed_request(
    toml_path="feeds.toml",
    model="small",
    format="sqlite",
    db_path="podcasts.db",
    limit=10,
    order="newest"
)

service = TranscriptionService()
response = service.process(request)

print(f"Processed {response.summary.processed} items")
print(f"Skipped {response.summary.skipped} already processed items")
```

## Request/Response Schema

### Request Schema

```json
{
  "input": {
    "type": "file" | "feed",
    "source": "path/to/file"
  },
  "processing": {
    "model": "tiny" | "base" | "small" | "medium" | "large",
    "verbose": true | false,
    "feed_options": {
      "limit": 10,
      "order": "newest" | "oldest",
      "max_retries": 3,
      "initial_delay": 1.0
    }
  },
  "output": {
    "format": "text" | "json" | "table" | "csv" | "toml" | "sqlite",
    "destination": "output/path",
    "database": {
      "db_path": "database.db",
      "metadata_table": "metadata",
      "segments_table": "segments"
    }
  }
}
```

### Response Schema

```json
{
  "success": true,
  "results": [
    {
      "filename": "audio.mp3",
      "file_hash": "sha256...",
      "language": "en",
      "full_text": "...",
      "segments": [...]
    }
  ],
  "errors": [
    {
      "source": "file/path",
      "error_type": "FileNotFoundError",
      "message": "...",
      "timestamp": "2024-01-20T10:30:00"
    }
  ],
  "summary": {
    "total_items": 100,
    "processed": 95,
    "skipped": 3,
    "failed": 2,
    "elapsed_time": 120.5
  }
}
```

## Benefits

1. **Clean API Boundary**: Request/response objects provide a clear contract
2. **Serializable**: All objects can be serialized to/from JSON for REST APIs
3. **Validation**: Built-in validation ensures requests are well-formed
4. **Extensible**: Easy to add new input sources, processing options, or output formats
5. **Testable**: Service layer can be easily unit tested
6. **Future-proof**: Ready for REST API, gRPC, or other interfaces

## Migration Guide

### From Direct API to Request/Response API

**Before:**
```python
transcriber = AudioTranscriber(model_name="base")
result = transcriber.transcribe_file("audio.mp3")
print(result.to_json())
```

**After:**
```python
service = TranscriptionService()
request = create_file_request("audio.mp3", model="base", format="json")
response = service.process(request)
if response.success:
    print(response.results[0].to_json())
```

The legacy API remains available for backward compatibility.
