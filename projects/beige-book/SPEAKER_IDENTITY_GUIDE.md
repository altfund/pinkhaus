# Speaker Identity Tracking - Quick Reference Guide

## Overview

Speaker identity tracking allows you to recognize and track recurring speakers across multiple podcast episodes. This builds on top of speaker diarization to provide persistent speaker profiles.

## Key Features

- 🎯 **Voice Fingerprinting**: Extract unique voice embeddings for each speaker
- 👥 **Persistent Profiles**: Maintain speaker identities across episodes
- 🔍 **Automatic Recognition**: Match speakers using voice similarity
- 📊 **Query Capabilities**: Search statements by speaker over time
- 🏷️ **Role Labels**: Assign canonical roles (HOST, GUEST, etc.)

## Quick Start

```python
from beige_book.transcriber import AudioTranscriber
from beige_book.database import TranscriptionDatabase

# Setup
db = TranscriptionDatabase("podcast.db")
db.create_tables()
db.create_speaker_identity_tables()

# Transcribe with speaker identification
transcriber = AudioTranscriber(model_name="tiny")
result = transcriber.transcribe_file(
    "episode.mp3",
    enable_diarization=True,
    enable_speaker_identification=True,
    feed_url="https://podcast.com/feed.rss"
)

# Save (automatically identifies speakers)
db.save_transcription(result, feed_url="https://podcast.com/feed.rss")

# Query speakers
profiles = db.get_speaker_profiles_for_feed("https://podcast.com/feed.rss")
for profile in profiles:
    print(f"{profile['display_name']}: {profile['total_appearances']} episodes")
```

## Documentation

- **[README.md](README.md)** - Main documentation with feature overview
- **[README_SPEAKER_DIARIZATION.md](README_SPEAKER_DIARIZATION.md)** - Detailed speaker diarization and identity docs
- **[API_SPEAKER_IDENTITY.md](API_SPEAKER_IDENTITY.md)** - Complete API reference
- **[examples/speaker_identity_quickstart.py](examples/speaker_identity_quickstart.py)** - Quick start examples
- **[examples/speaker_identity_examples.py](examples/speaker_identity_examples.py)** - Comprehensive examples
- **[tests/test_speaker_identity.py](tests/test_speaker_identity.py)** - Unit tests

## Database Schema

Four new tables support speaker identity:

1. **speaker_profiles** - Persistent speaker identities
2. **speaker_embeddings** - Voice fingerprints
3. **speaker_occurrences** - Links temporary labels to profiles
4. **speaker_metadata** - Additional speaker information

## Configuration

Environment variables:
- `SPEAKER_EMBEDDING_METHOD`: 'speechbrain' (default), 'pyannote', or 'mock'
- `SPEAKER_MATCHING_THRESHOLD`: Similarity threshold (default: 0.85)
- `HF_TOKEN`: HuggingFace token for models

## Common Use Cases

### 1. Track Podcast Hosts
Pre-register known hosts for accurate identification across episodes.

### 2. Guest Analytics
Analyze guest appearances, speaking time, and contributions.

### 3. Content Search
Find all statements by a specific speaker on particular topics.

### 4. Speaker Timeline
Track when speakers appeared together or separately.

## Tips

- Pre-register known speakers with voice samples for best accuracy
- Process episodes chronologically for better cross-episode tracking
- Use canonical labels (HOST, GUEST) for easier querying
- Minimum 3 seconds of speech needed for reliable voice embedding
- Voice similarity threshold of 0.85 works well for most cases

## Requirements

- Python 3.11 (for pyannote compatibility)
- HuggingFace token (for speaker diarization)
- SpeechBrain or PyAnnote (for voice embeddings)
- ~1GB disk space for models

## Performance

- Voice embedding extraction: ~1-2 seconds per speaker
- Database queries: <100ms for most operations
- Storage: ~1KB per voice embedding
- Accuracy: 85-95% for speaker recognition (varies by audio quality)