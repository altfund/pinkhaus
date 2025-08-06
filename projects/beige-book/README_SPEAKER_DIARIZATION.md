# Speaker Diarization and Identity Tracking

This document explains how to use pyannote-audio for speaker diarization (identifying "who speaks when") and speaker identity tracking (recognizing recurring speakers) in podcast transcriptions.

## Quick Test

**Run the test now that you have your HF_TOKEN set:**

```bash
# Make sure your token is set
echo $HF_TOKEN  # Should show your token

# Run the harvard audio test
python tests/test_harvard_diarization.py
```

This test uses the harvard.wav file and creates a complete database with speaker information.

## Current Status

✅ **pyannote-audio is now fully integrated and working with Python 3.11!**

The project has been configured to use Python 3.11 to avoid dependency issues with newer Python versions. Speaker diarization is available in two modes:

1. **Real Mode**: Full pyannote-audio speaker diarization (requires HF token)
2. **Mock Mode**: For testing and development (no token needed)

## What's Been Implemented

### 1. Speaker Diarizer Module (`beige_book/speaker_diarizer.py`)
- `SpeakerDiarizer` class for speaker identification
- Mock diarization for testing without pyannote
- Alignment of speaker segments with transcription segments
- Support for Hugging Face model loading

### 2. Extended Data Models
- Updated protobuf definitions to include speaker information
- Added `speaker` and `confidence` fields to segments
- Added `num_speakers` and `has_speaker_labels` to results

### 3. Enhanced Transcriber
- Added `enable_diarization` parameter to `transcribe_file()`
- Automatic fallback to mock mode if pyannote unavailable
- Integration with existing transcription pipeline

### 4. Updated Output Formats
- JSON: Includes speaker labels and confidence scores
- CSV: Added speaker column when diarization is enabled
- Table: Shows speaker information in formatted output

## Usage

### Basic Usage (Mock Mode)

```python
from beige_book.transcriber import AudioTranscriber

# Initialize transcriber
transcriber = AudioTranscriber(model_name="tiny")

# Transcribe with mock speaker diarization
result = transcriber.transcribe_file(
    "podcast.wav",
    enable_diarization=True
)

# Output will include speaker labels
print(result.to_json())
```

### With Real pyannote-audio (When Available)

```python
# Set your Hugging Face token
import os
os.environ["HF_TOKEN"] = "your-token-here"

# Transcribe with real speaker diarization
result = transcriber.transcribe_file(
    "podcast.wav",
    enable_diarization=True,
    hf_token=os.getenv("HF_TOKEN")
)
```

### Using the Standalone Diarizer

```python
from beige_book.speaker_diarizer import SpeakerDiarizer

# Initialize diarizer
diarizer = SpeakerDiarizer(auth_token="your-hf-token")

# Perform diarization
diarization_result = diarizer.diarize_file("podcast.wav")

# Access speaker segments
for segment in diarization_result.segments:
    print(f"{segment.speaker}: {segment.start:.2f}s - {segment.end:.2f}s")
```

## Installation Requirements

The project now uses Python 3.11, which fully supports pyannote-audio. Everything is already installed and ready to use!

### Quick Start

1. **Activate the environment:**
   ```bash
   flox activate
   source .venv/bin/activate
   ```

2. **Set up Hugging Face token (for real diarization):**
   ```bash
   # Create account at https://huggingface.co
   # Accept model conditions at https://huggingface.co/pyannote/speaker-diarization-3.1
   # Generate token at https://huggingface.co/settings/tokens
   export HF_TOKEN='your-token-here'
   ```

3. **Run transcription with diarization:**
   ```bash
   # Using the CLI (once implemented)
   transcribe podcast.wav --enable-diarization
   
   # Or using Python
   python demos/demo_diarization.py
   ```

## Hugging Face Setup

To use real speaker diarization models:

1. **Create account** at https://huggingface.co

2. **Accept model conditions** at https://huggingface.co/pyannote/speaker-diarization-3.1
   - Click "Agree and access repository" button
   - This is required before the model can be downloaded

3. **Generate access token** at https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it (e.g., "pyannote-diarization")
   - Select "read" permission (that's all you need)
   - Copy the token (starts with `hf_...`)

4. **Set token as environment variable**: 
   ```bash
   export HF_TOKEN='hf_your_token_here'
   ```

**Token Permissions Required**: Only 'read' permission is needed to download and use the models.

## Example Output

### Without Speaker Diarization
```json
{
  "segments": [
    {
      "start": "00:00:00.000",
      "end": "00:00:05.230",
      "text": "Welcome to our podcast!",
      "duration": 5.23
    }
  ]
}
```

### With Speaker Diarization
```json
{
  "segments": [
    {
      "start": "00:00:00.000",
      "end": "00:00:05.230",
      "text": "Welcome to our podcast!",
      "duration": 5.23,
      "speaker": "SPEAKER_0",
      "confidence": 0.95
    }
  ],
  "num_speakers": 2,
  "has_speaker_labels": true
}
```

## Future Improvements

1. **Speaker Recognition**: Identify recurring speakers across episodes
2. **Speaker Naming**: Allow manual labeling of speakers (Host, Guest, etc.)
3. **Overlap Handling**: Better handling of overlapping speech
4. **Real-time Processing**: Stream-based diarization for live podcasts
5. **Custom Models**: Fine-tune models for specific podcast domains

## Troubleshooting

### "pyannote-audio is not installed" Error
- This is expected with Python 3.13
- The system will automatically fall back to mock diarization
- To use real diarization, see installation options above

### "No HF_TOKEN found" Error
- Set your Hugging Face token: `export HF_TOKEN=your-token-here`
- Or pass it directly: `transcribe_file(..., hf_token="your-token")`

### Performance Issues
- Speaker diarization is computationally intensive
- Use GPU if available: pyannote will automatically detect CUDA
- Consider processing in batches for multiple files

## Speaker Identity Tracking (NEW!)

Beyond just identifying different speakers in a single recording, the system can now track and recognize speakers across multiple recordings.

### How It Works

1. **Voice Embeddings**: Extracts numerical "fingerprints" of each speaker's voice
2. **Speaker Profiles**: Stores persistent profiles for recurring speakers
3. **Automatic Matching**: Compares new speakers against known profiles
4. **Confidence Scoring**: Uses cosine similarity (threshold: 0.85) for matching

### Database Schema

The system adds four new tables:

- `speaker_profiles`: Persistent speaker identities with metadata
- `speaker_embeddings`: Voice fingerprints for recognition
- `speaker_occurrences`: Links temporary labels to profiles
- `speaker_metadata`: Additional speaker information

### Basic Usage

```python
from beige_book.transcriber import AudioTranscriber
from beige_book.database import TranscriptionDatabase

# Enable both diarization and speaker identification
transcriber = AudioTranscriber(model_name="tiny")
result = transcriber.transcribe_file(
    "episode_001.mp3",
    enable_diarization=True,
    enable_speaker_identification=True,  # NEW!
    feed_url="https://podcast.example.com/feed.rss"
)

# Save to database (triggers automatic speaker matching)
db = TranscriptionDatabase("podcast.db")
db.create_tables()
db.create_speaker_identity_tables()  # Create identity tables
trans_id = db.save_transcription(result)
```

### Pre-registering Known Speakers

For better accuracy, pre-register known speakers:

```python
# Create profile for the host
host_id = db.create_speaker_profile(
    display_name="John Doe",
    feed_url="https://podcast.example.com/feed.rss",
    canonical_label="HOST"
)

# Add reference voice embedding (from intro/outro)
from beige_book.voice_embeddings import VoiceEmbeddingExtractor, serialize_embedding

extractor = VoiceEmbeddingExtractor()
embedding, quality = extractor.extract_embedding_from_file(
    "host_intro.wav",
    start_time=0.0,
    end_time=10.0
)

db.add_speaker_embedding(
    profile_id=host_id,
    embedding=serialize_embedding(embedding),
    embedding_dimension=256,
    quality_score=quality
)
```

### Querying Speaker Data

```python
# Get all speakers for a podcast
profiles = db.get_speaker_profiles_for_feed("https://podcast.example.com/feed.rss")
for profile in profiles:
    print(f"{profile['display_name']}: {profile['total_appearances']} episodes")

# Get all statements by a specific speaker
statements = db.get_speaker_statements(
    profile_id=host_id,
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Find when two speakers appeared together
from beige_book.database import TranscriptionDatabase

with db._get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT t.filename, t.created_at
        FROM transcription_metadata t
        JOIN speaker_occurrences so1 ON so1.transcription_id = t.id
        JOIN speaker_occurrences so2 ON so2.transcription_id = t.id
        WHERE so1.profile_id = ? AND so2.profile_id = ?
    """, (host_id, guest_id))
    
    co_appearances = cursor.fetchall()
```

### Managing Profiles

```python
from beige_book.speaker_matcher import SpeakerMatcher

matcher = SpeakerMatcher(db)

# Manually verify a speaker match
db.link_speaker_occurrence(
    transcription_id=trans_id,
    temporary_label="SPEAKER_0",
    profile_id=host_id,
    confidence=1.0,
    is_verified=True
)

# Merge duplicate profiles
matcher.merge_speaker_profiles(
    profile_id_keep=host_id,
    profile_id_merge=duplicate_id
)
```

### Configuration

Environment variables:
- `SPEAKER_EMBEDDING_METHOD`: 'speechbrain' (default), 'pyannote', or 'mock'
- `SPEAKER_MATCHING_THRESHOLD`: Similarity threshold (default: 0.85)
- `SPEAKER_MIN_DURATION`: Minimum speech duration for embedding (default: 3.0 seconds)

### Use Cases

1. **Podcast Analytics**: Track speaker participation over time
2. **Content Search**: Find all episodes where specific people spoke
3. **Speaker Statistics**: Analyze speaking time, frequency, co-appearances
4. **Automated Show Notes**: Generate speaker-attributed summaries
5. **Compliance**: Track speaker consent and appearances

### Technical Details

- **Embedding Dimension**: 256-dimensional vectors
- **Similarity Metric**: Cosine similarity (normalized 0-1)
- **Storage**: Embeddings stored as BLOB in SQLite
- **Scoping**: Speakers are scoped to feeds to avoid cross-contamination

### Limitations

1. **Minimum Duration**: Needs ~3 seconds of speech for reliable embedding
2. **Voice Variability**: Illness, recording quality affect accuracy
3. **Similar Voices**: Family members or similar voices may be confused
4. **Storage**: Each embedding is ~1KB (256 floats × 4 bytes)

### Future Enhancements

- [ ] Web UI for profile management
- [ ] Bulk speaker verification interface  
- [ ] Export speaker timelines
- [ ] Cross-feed speaker matching (optional)
- [ ] Voice change detection (illness, age)