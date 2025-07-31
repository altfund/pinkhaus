# pinkhaus-models

Shared database models and schemas for pinkhaus projects. This package provides SQLAlchemy models and database utilities for storing and retrieving podcast transcriptions.

## Installation

### Using uv (Recommended)
```bash
# From your project directory (e.g., grant or beige-book)
uv add -e ../pinkhaus-models
```

### Using pip
```bash
# From your project directory
pip install -e ../pinkhaus-models
```

### Using in pyproject.toml
```toml
[project]
dependencies = [
    "pinkhaus-models @ file:///path/to/pinkhaus-models",
]
```

## Models

### TranscriptionMetadata
Stores metadata about transcribed audio files and podcast episodes.

**Fields:**
- `id`: Primary key (Integer)
- `filename`: Name of the transcribed file (String)
- `file_hash`: SHA256 hash of the audio file (String)
- `language`: Detected language code, e.g., "en" (String)
- `full_text`: Complete transcription text (Text)
- `model_name`: Whisper model used for transcription (String)
- `feed_url`: RSS feed URL if from a podcast (String, optional)
- `feed_item_id`: Unique ID of the feed item (String, optional)
- `feed_item_title`: Title of the podcast episode (String, optional)
- `feed_item_published`: Publication date (String, optional)
- `feed_item_audio_url`: Direct URL to audio file (String, optional)
- `created_at`: Timestamp of creation (DateTime)

### TranscriptionSegment
Stores individual segments of a transcription with timestamps.

**Fields:**
- `id`: Primary key (Integer)
- `transcription_id`: Foreign key to TranscriptionMetadata (Integer)
- `segment_index`: Order of segment in transcription (Integer)
- `start_time`: Start time in seconds (Float)
- `end_time`: End time in seconds (Float)
- `duration`: Duration in seconds (Float)
- `text`: Segment text (Text)

### TranscriptionResult
In-memory model for working with complete transcriptions.

**Properties:**
- `filename`: Audio file name
- `file_hash`: SHA256 hash
- `language`: Language code
- `segments`: List of Segment objects
- `full_text`: Complete transcribed text
- `feed_metadata`: Optional metadata for podcast episodes

**Methods:**
- `add_segment(start, end, text)`: Add a new segment
- `to_json()`: Export as JSON string
- `to_csv()`: Export as CSV string
- `to_toml()`: Export as TOML string
- `to_table()`: Export as formatted ASCII table
- `to_protobuf_bytes()`: Serialize to Protocol Buffer bytes
- `to_protobuf_base64()`: Serialize to base64-encoded Protocol Buffer
- `from_protobuf_bytes(data)`: Deserialize from Protocol Buffer bytes
- `from_protobuf_base64(data)`: Deserialize from base64-encoded Protocol Buffer

### Segment
Individual transcription segment.

**Properties:**
- `start`: Start time in seconds
- `end`: End time in seconds
- `text`: Segment text
- `duration`: Calculated duration

## Database Operations

### TranscriptionDatabase
Main database interface for storing and retrieving transcriptions.

```python
from pinkhaus_models import TranscriptionDatabase, TranscriptionResult

# Initialize database
db = TranscriptionDatabase("transcriptions.db")

# Create tables if they don't exist
db.create_tables()

# Save a transcription
result = TranscriptionResult()
result.filename = "episode.mp3"
result.file_hash = "abc123"
result.language = "en"
result.full_text = "This is the transcription"
result.add_segment(0.0, 5.0, "This is the transcription")

transcription_id = db.save_transcription(
    result,
    model_name="base",
    feed_url="https://example.com/feed.xml",
    feed_item_id="episode-1",
    feed_item_title="Episode 1: Introduction"
)

# Retrieve transcription
transcription = db.get_transcription(transcription_id)
print(f"Text: {transcription['full_text']}")
print(f"Segments: {len(transcription['segments'])}")

# Get all transcriptions
all_transcriptions = db.get_all_transcriptions()
for t in all_transcriptions:
    print(f"{t.filename}: {t.language}")

# Search transcriptions
results = db.search_transcriptions("keyword")
for r in results:
    print(f"Found in {r.filename}: {r.full_text[:100]}...")

# Check if feed item exists (for duplicate detection)
exists = db.check_feed_item_exists(
    "https://example.com/feed.xml",
    "episode-1"
)

# Get transcriptions by date range
from datetime import datetime, timedelta
yesterday = datetime.now() - timedelta(days=1)
recent = db.get_transcriptions_after(yesterday)

# Get transcription by ID with segments
full_data = db.get_transcription_with_segments(transcription_id)
for segment in full_data['segments']:
    print(f"{segment['start_time']}-{segment['end_time']}: {segment['text']}")
```

## Usage Examples

### Basic Usage
```python
from pinkhaus_models import TranscriptionDatabase, TranscriptionMetadata

# Open database
db = TranscriptionDatabase("podcasts.db")
db.create_tables()

# Query all transcriptions
transcriptions = db.get_all_transcriptions()
for t in transcriptions:
    print(f"{t.filename} ({t.language}): {t.created_at}")

# Search for specific content
results = db.search_transcriptions("artificial intelligence")
print(f"Found {len(results)} transcriptions mentioning AI")
```

### Working with Feed Data
```python
# Save transcription with podcast metadata
db.save_transcription(
    result,
    model_name="medium",
    feed_url="https://example.com/podcast.xml",
    feed_item_id="ep-123",
    feed_item_title="Episode 123: Future of Tech",
    feed_item_published="2025-07-01T10:00:00Z",
    feed_item_audio_url="https://example.com/ep123.mp3"
)

# Check for duplicates before processing
if not db.check_feed_item_exists(feed_url, item_id):
    # Process the episode
    pass
```

### Advanced Queries
```python
# Custom queries using SQLAlchemy
from pinkhaus_models import TranscriptionMetadata
from sqlalchemy import and_

# Get transcriptions from a specific feed
feed_transcriptions = db.session.query(TranscriptionMetadata).filter(
    TranscriptionMetadata.feed_url == "https://example.com/feed.xml"
).all()

# Get transcriptions in a specific language
english_transcriptions = db.session.query(TranscriptionMetadata).filter(
    TranscriptionMetadata.language == "en"
).order_by(TranscriptionMetadata.created_at.desc()).all()
```

## Database Schema

### Tables Created

1. **transcription_metadata** (or custom name via metadata_table parameter)
   - Primary storage for transcription information
   - Includes unique index on (feed_url, feed_item_id) for duplicate prevention
   
2. **transcription_segments** (or custom name via segments_table parameter)
   - Stores time-stamped segments
   - Foreign key relationship to metadata table with CASCADE delete

### Relationships
- One-to-many relationship between metadata and segments
- Segments are automatically deleted when parent transcription is deleted

## Integration with Other Projects

This package is designed to work seamlessly with:
- **beige-book**: For transcribing audio and storing results
- **grant**: For RAG queries over stored transcriptions

Both projects use these models to ensure consistent data storage and retrieval.