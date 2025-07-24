# Beige Book - Audio Transcription Tool

A command-line tool, Python library, and REST API for transcribing audio files using OpenAI's Whisper model with support for multiple structured output formats including SQLite database storage. Now with RSS/podcast feed processing capabilities.

## Features

- Multiple output formats: text, JSON, CSV, TOML, table, and SQLite database
- SHA256 file hashing for integrity verification
- Structured data with timestamps and segments
- SQLite database storage with foreign key relationships
- RSS/podcast feed processing with automatic audio download
- Resumable feed processing with duplicate detection
- Feed item ordering (newest/oldest first) and limiting
- Python library API for programmatic use
- REST API with Swagger/OpenAPI documentation
- Comprehensive test suite

## Installation

1. Ensure you have Python 3.10+ installed
2. Clone the repository
3. Install dependencies:
   ```bash
   cd beige-book/
   flox activate
   uv pip install -e .
   ```

### API Dependencies

To run the REST API server, install additional dependencies:
```bash
uv pip install fastapi uvicorn
```

## Command Line Usage

### Basic Usage

```bash
transcribe /path/to/audio.wav
```

### Full Command Reference

```
usage: transcribe [-h] [--model {tiny,base,small,medium,large}]
                  [--format {text,json,table,csv,toml,sqlite}]
                  [--output OUTPUT] [--db-path DB_PATH]
                  [--metadata-table METADATA_TABLE]
                  [--segments-table SEGMENTS_TABLE]
                  [--feed] [--verbose] [--limit LIMIT]
                  [--order {newest,oldest}]
                  filename

Transcribe audio files with Whisper

positional arguments:
  filename              Audio file or TOML feed file to transcribe

options:
  -h, --help            show this help message and exit
  --model {tiny,base,small,medium,large}
                        Whisper model to use
  --format {text,json,table,csv,toml,sqlite}
                        Output format (default: text)
  --output OUTPUT, -o OUTPUT
                        Output file (default: stdout)
  --db-path DB_PATH     Path to SQLite database file (required for sqlite
                        format)
  --metadata-table METADATA_TABLE
                        Name of the metadata table (default:
                        transcription_metadata)
  --segments-table SEGMENTS_TABLE
                        Name of the segments table (default:
                        transcription_segments)
  --feed                Treat input as a TOML file containing RSS feed URLs
  --verbose, -v         Enable verbose logging
  --limit LIMIT         Maximum number of feed items to process per feed
  --order {newest,oldest}
                        Process feed items from newest or oldest first
                        (default: newest)
```

### Output Formats

#### 1. Text Format (Default)
Returns only the transcribed text without timestamps.

```bash
transcribe audio.wav
```

#### 2. JSON Format
Structured output with metadata, timestamps, and segments.

```bash
transcribe audio.wav --format json
```

Output structure:
```json
{
  "filename": "audio.wav",
  "file_hash": "sha256_hash_here",
  "language": "en",
  "segments": [
    {
      "start": "00:00:00.000",
      "end": "00:00:04.500",
      "text": "Transcribed text here",
      "duration": 4.5
    }
  ],
  "full_text": "Complete transcribed text..."
}
```

#### 3. Table Format
ASCII table with formatted columns for easy reading.

```bash
transcribe audio.wav --format table
```

#### 4. CSV Format
Comma-separated values with proper escaping.

```bash
transcribe audio.wav --format csv
```

#### 5. TOML Format
TOML structured data format.

```bash
transcribe audio.wav --format toml
```

#### 6. SQLite Database Format
Store transcriptions in a SQLite database with proper normalization.

```bash
transcribe audio.wav --format sqlite --db-path transcriptions.db
```

With custom table names:
```bash
transcribe audio.wav --format sqlite --db-path my.db \
  --metadata-table my_metadata \
  --segments-table my_segments
```

### Model Selection

Choose different Whisper models based on your speed/accuracy needs:

```bash
transcribe audio.wav --model tiny    # Fastest, least accurate (default)
transcribe audio.wav --model base    # Good balance
transcribe audio.wav --model small   # Better accuracy
transcribe audio.wav --model medium  # High accuracy
transcribe audio.wav --model large   # Best accuracy, slowest
```

### Saving Output

Save transcription to a file:

```bash
transcribe audio.wav --format json --output result.json
transcribe audio.wav --format csv -o data.csv
```

## RSS Feed Processing

The tool can process RSS/podcast feeds from a TOML configuration file, automatically downloading and transcribing audio files.

### Feed Configuration

Create a TOML file with RSS feed URLs:

```toml
[feeds]
rss = [
    "https://feeds.example.com/podcast1.xml",
    "https://feeds.megaphone.fm/ESP9520742908",
    "https://feeds.example.com/podcast2.xml"
]
```

### Basic Feed Processing

Process all items from feeds:

```bash
uv run transcribe feeds.toml --feed
```

### Limiting Items

Process only the 5 newest episodes from each feed:

```bash
uv run transcribe feeds.toml --feed --limit 5
```

### Processing Order

Process oldest episodes first:

```bash
uv run transcribe feeds.toml --feed --order oldest
```

Process 10 oldest episodes from each feed:

```bash
uv run transcribe feeds.toml --feed --limit 10 --order oldest
```

### Resumable Processing

When outputting to a file or database, the tool tracks which items have been processed to avoid duplicates:

```bash
# Resumable with database
uv run transcribe feeds.toml --feed --format sqlite --db-path podcasts.db

# Resumable with file output
uv run transcribe feeds.toml --feed --format json --output transcriptions.json

# Non-resumable (stdout)
uv run transcribe feeds.toml --feed --format text
```

If processing is interrupted, simply run the same command again and it will skip already-processed items.

### Feed Metadata in Output

When processing feeds, additional metadata is included in the output:

- **JSON/TOML formats**: Includes `feed_metadata` object with feed URL, item ID, title, and publication date
- **Text/CSV/Table formats**: Includes feed information in headers/comments

Example JSON output with feed metadata:

```json
{
  "filename": "downloaded_episode.mp3",
  "file_hash": "sha256_hash_here",
  "language": "en",
  "segments": [...],
  "full_text": "...",
  "feed_metadata": {
    "feed_url": "https://feeds.example.com/podcast.xml",
    "item_id": "unique-episode-id",
    "title": "Episode Title",
    "audio_url": "https://example.com/episode.mp3",
    "published": "2025-07-19T12:00:00"
  }
}
```

## Library Usage

### Basic Example

```python
from beige_book import AudioTranscriber

# Create transcriber
transcriber = AudioTranscriber(model_name="tiny")

# Transcribe file
result = transcriber.transcribe_file("audio.wav")

# Access data
print(f"Language: {result.language}")
print(f"Text: {result.full_text}")

# Export to different formats
json_output = result.to_json()
csv_output = result.to_csv()
```

### Database Storage

```python
from beige_book import AudioTranscriber, TranscriptionDatabase

# Transcribe
transcriber = AudioTranscriber()
result = transcriber.transcribe_file("audio.wav")

# Store in database
db = TranscriptionDatabase("transcriptions.db")
db.create_tables()
transcription_id = db.save_transcription(result, model_name="tiny")

# Retrieve later
data = db.get_transcription(transcription_id)
```

### Feed Processing

```python
from beige_book import FeedParser, AudioDownloader, AudioTranscriber, TranscriptionDatabase

# Parse feeds
parser = FeedParser()
feed_items = parser.parse_feed("https://feeds.example.com/podcast.xml")

# Download and transcribe
downloader = AudioDownloader()
transcriber = AudioTranscriber(model_name="tiny")
db = TranscriptionDatabase("podcasts.db")
db.create_tables()

for item in feed_items[:5]:  # Process first 5 items
    # Check if already processed
    if db.check_feed_item_exists(item.feed_url, item.item_id):
        continue

    # Download audio
    temp_path, file_hash = downloader.download_to_temp(item.audio_url)

    try:
        # Transcribe
        result = transcriber.transcribe_file(temp_path)

        # Save with feed metadata
        db.save_transcription(
            result,
            model_name="tiny",
            feed_url=item.feed_url,
            feed_item_id=item.item_id,
            feed_item_title=item.title,
            feed_item_published=item.published.isoformat() if item.published else None
        )
    finally:
        downloader.cleanup_temp_file(temp_path)
```

See `examples/` directory for more usage examples.

## Protocol Buffers Support

The library includes Protocol Buffers support for efficient binary serialization of transcription results.

### Generating Protobuf Python Classes

To regenerate the Python protobuf classes from the `.proto` file:

```bash
# Using grpcio-tools (already installed)
uv run python -m grpc_tools.protoc -I=. --python_out=. beige_book/transcription.proto

# Or if you have protoc installed via flox
flox install protobuf
protoc -I=beige_book --python_out=beige_book beige_book/transcription.proto

# Note: If you encounter protobuf version mismatches in flox environments,
# you may need to comment out the version validation in the generated file
```

### Protocol Buffers Integration

TranscriptionResult now uses Protocol Buffers internally for efficient storage and serialization, while maintaining the same API:

```python
from beige_book import AudioTranscriber, TranscriptionResult

# Use standard transcriber - it now creates protobuf-based results
transcriber = AudioTranscriber(model_name="tiny")
result = transcriber.transcribe_file("audio.wav")

# Direct protobuf serialization (very efficient)
proto_bytes = result.to_protobuf_bytes()  # Binary format
restored = TranscriptionResult.from_protobuf_bytes(proto_bytes)

# Base64 encoding for network/text transmission
encoded = result.to_protobuf_base64()
restored = TranscriptionResult.from_protobuf_base64(encoded)

# All other formats still work exactly the same
json_str = result.to_json()
toml_str = result.to_toml()
csv_str = result.to_csv()
table_str = result.to_table()

# Create results programmatically
result = TranscriptionResult()
result.filename = "my_audio.wav"
result.file_hash = "hash123"
result.language = "en"
result.full_text = "Hello world"
result.add_segment(0.0, 2.0, "Hello world")
```

### Benefits of Protobuf

- **Compact**: Binary format is typically 3-5x smaller than JSON
- **Fast**: Efficient serialization/deserialization
- **Type-safe**: Strong typing with generated classes
- **Language-agnostic**: Can be used with any language that supports protobuf
- **Zero overhead**: TranscriptionResult now uses protobuf internally, so there's no conversion cost

**Note**: The API remains unchanged - existing code will continue to work. The only difference is that TranscriptionResult now stores data in a protobuf structure internally for better performance and smaller memory footprint.

See `examples/protobuf_usage.py` for complete examples.

## REST API

The tool includes a REST API server with Swagger/OpenAPI documentation.

### Starting the API Server

```bash
# Basic usage
uv run python run_api.py

# With auto-reload for development
uv run python run_api.py --reload

# Custom host and port
uv run python run_api.py --host 0.0.0.0 --port 8080

# With debug logging
uv run python run_api.py --log-level debug
```

### API Documentation

Once the server is running, access the documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

### API Usage Example

Transcribe an audio file with curl:

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "type": "file",
      "source": "/path/to/audio.wav"
    },
    "processing": {
      "model": "medium",
      "verbose": false
    },
    "output": {
      "format": "json"
    }
  }'
```

Process RSS feeds:

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "type": "feed",
      "source": "/path/to/feeds.toml"
    },
    "processing": {
      "model": "base",
      "verbose": true,
      "feed_options": {
        "limit": 10,
        "order": "newest"
      }
    },
    "output": {
      "format": "sqlite",
      "database": {
        "db_path": "/path/to/podcasts.db"
      }
    }
  }'
```

## Running Tests

The project includes a comprehensive test suite using pytest.

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_transcriber.py
pytest tests/test_database.py

# Run specific test
pytest tests/test_transcriber.py::TestAudioTranscriber::test_file_hash

# Run with coverage (requires pytest-cov)
pytest --cov=beige_book
```

### Test Structure

- `tests/test_transcriber.py` - Tests for core transcription functionality
- `tests/test_database.py` - Tests for SQLite database operations
- Test audio file: `../../resources/audio/harvard.wav` - Shared test audio file with known content (located in the parent project's resources directory)

The tests verify:
- File hash calculation
- Transcription accuracy
- All output formats (JSON, CSV, TOML, table)
- Database operations (CRUD, foreign keys, cascading deletes)
- Error handling

## Database Schema

When using SQLite storage, the following schema is created:

### Metadata Table
- `id` - Primary key
- `filename` - Name of transcribed file
- `file_hash` - SHA256 hash of the file
- `language` - Detected language code
- `full_text` - Complete transcription text
- `model_name` - Whisper model used
- `feed_url` - RSS feed URL (for feed items)
- `feed_item_id` - Unique ID of feed item
- `feed_item_title` - Title of feed item
- `feed_item_published` - Publication date of feed item
- `created_at` - Timestamp

The table includes a unique index on `(feed_url, feed_item_id)` to prevent duplicate processing of feed items.

### Segments Table
- `id` - Primary key
- `transcription_id` - Foreign key to metadata
- `segment_index` - Order of segment
- `start_time` - Start time in seconds
- `end_time` - End time in seconds
- `duration` - Duration in seconds
- `text` - Segment text

## Troubleshooting

### "File not found" error when file exists
Ensure `ffmpeg` is installed and accessible in your PATH.

### FP16 Warning
The warning "FP16 is not supported on CPU; using FP32 instead" is normal and doesn't affect quality.

### Memory issues with large files
Try using a smaller model or processing the audio in chunks.

### Poor transcription quality
- Use a larger model (`--model medium` or `--model large`)
- Ensure audio quality is good (clear speech, minimal background noise)
