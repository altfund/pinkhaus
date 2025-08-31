# Speaker Identity and Voice Fingerprinting Design

## Overview

This document outlines the design for persistent speaker identification across recordings, enabling tracking of speakers over time within podcasts/feeds.

## Core Concepts

1. **Speaker Profile**: A persistent identity representing a real person
2. **Voice Embedding**: A numerical representation of a speaker's voice characteristics (256-dimensional vector)
3. **Speaker Occurrence**: An instance of a speaker in a specific recording
4. **Confidence Threshold**: Minimum similarity score to match speakers

## Database Schema

### New Tables

```sql
-- Global speaker profiles (per feed/podcast)
CREATE TABLE speaker_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feed_url TEXT,  -- Scoped to specific podcast/feed
    display_name TEXT NOT NULL,  -- e.g., "Joe Rogan", "Guest: Elon Musk"
    canonical_label TEXT,  -- e.g., "HOST", "GUEST_1", "REGULAR_COHOST"
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    total_appearances INTEGER DEFAULT 0,
    total_duration_seconds REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(feed_url, display_name)
);

-- Voice embeddings for speaker identification
CREATE TABLE speaker_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,  -- Serialized numpy array (256 dims)
    source_transcription_id INTEGER,  -- Which recording this came from
    source_segment_indices TEXT,  -- JSON array of segment indices used
    quality_score REAL,  -- Confidence in this embedding
    extraction_method TEXT,  -- 'pyannote', 'speechbrain', etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (profile_id) REFERENCES speaker_profiles(id),
    FOREIGN KEY (source_transcription_id) REFERENCES transcription_metadata(id)
);

-- Links temporary speaker labels to permanent profiles
CREATE TABLE speaker_occurrences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transcription_id INTEGER NOT NULL,
    temporary_label TEXT NOT NULL,  -- e.g., "SPEAKER_0" from diarization
    profile_id INTEGER,  -- Matched permanent speaker
    confidence REAL,  -- How confident we are in this match
    is_verified BOOLEAN DEFAULT 0,  -- Human-verified match
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transcription_id) REFERENCES transcription_metadata(id),
    FOREIGN KEY (profile_id) REFERENCES speaker_profiles(id),
    UNIQUE(transcription_id, temporary_label)
);

-- Speaker profile metadata (optional enrichment)
CREATE TABLE speaker_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT,
    FOREIGN KEY (profile_id) REFERENCES speaker_profiles(id),
    UNIQUE(profile_id, key)
);
```

### Updates to Existing Tables

```sql
-- Add to segments table (already has speaker_id for temporary speakers)
ALTER TABLE transcription_segments ADD COLUMN profile_id INTEGER;
ALTER TABLE transcription_segments ADD FOREIGN KEY (profile_id) REFERENCES speaker_profiles(id);
```

## Implementation Architecture

### 1. Voice Embedding Extraction

```python
class VoiceEmbeddingExtractor:
    """Extract voice embeddings using multiple methods"""
    
    def __init__(self, method='speechbrain'):
        self.method = method
        self.model = self._load_model()
    
    def extract_embedding(self, audio_path: str, segments: List[Segment]) -> np.ndarray:
        """Extract embedding from audio segments"""
        # Concatenate audio from all segments for this speaker
        # Extract 256-dimensional embedding
        # Return normalized embedding
    
    def extract_embeddings_for_transcription(self, audio_path: str, 
                                           transcription: TranscriptionResult) -> Dict[str, np.ndarray]:
        """Extract embeddings for all speakers in a transcription"""
        # Group segments by speaker
        # Extract embedding for each speaker
        # Return mapping of speaker_label -> embedding
```

### 2. Speaker Matching Service

```python
class SpeakerMatcher:
    """Match temporary speakers to persistent profiles"""
    
    def __init__(self, db: TranscriptionDatabase, threshold: float = 0.85):
        self.db = db
        self.threshold = threshold  # Cosine similarity threshold
    
    def match_speaker(self, embedding: np.ndarray, feed_url: str) -> Optional[SpeakerProfile]:
        """Find best matching speaker profile for an embedding"""
        # Get all embeddings for this feed
        # Calculate cosine similarity
        # Return best match if above threshold
    
    def identify_speakers_in_transcription(self, transcription_id: int, 
                                         embeddings: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Match all speakers in a transcription to profiles"""
        # For each temporary speaker label
        # Try to match to existing profile
        # Create new profile if no match
        # Return mapping of temp_label -> profile_id
```

### 3. Enhanced Database Methods

```python
class TranscriptionDatabase:
    # ... existing methods ...
    
    def create_speaker_profile(self, feed_url: str, display_name: str, 
                             canonical_label: Optional[str] = None) -> int:
        """Create a new persistent speaker profile"""
    
    def add_speaker_embedding(self, profile_id: int, embedding: np.ndarray,
                            source_transcription_id: int, **kwargs) -> int:
        """Add a voice embedding to a speaker profile"""
    
    def link_speaker_occurrence(self, transcription_id: int, temp_label: str,
                              profile_id: int, confidence: float) -> int:
        """Link a temporary speaker to a permanent profile"""
    
    def get_speaker_history(self, profile_id: int, limit: int = 100) -> List[Dict]:
        """Get all appearances of a speaker across recordings"""
    
    def get_speaker_statements(self, profile_id: int, 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> List[Dict]:
        """Get all statements made by a speaker with timestamps"""
    
    def merge_speaker_profiles(self, profile_id1: int, profile_id2: int) -> int:
        """Merge two profiles (for fixing misidentifications)"""
```

## Workflow

### Processing New Transcription

1. **Transcribe with diarization** (existing)
2. **Extract embeddings** for each speaker
3. **Match speakers** to existing profiles:
   - Query embeddings for the feed
   - Calculate similarities
   - Match if above threshold
4. **Create new profiles** for unmatched speakers
5. **Link occurrences** in database
6. **Update segment profile_ids**

### Query Examples

```sql
-- Get all statements by a specific person
SELECT t.filename, t.created_at, s.start_time, s.text
FROM transcription_segments s
JOIN transcription_metadata t ON s.transcription_id = t.id
WHERE s.profile_id = ?
ORDER BY t.created_at, s.start_time;

-- Get speaker statistics for a podcast
SELECT sp.display_name, sp.total_appearances, sp.total_duration_seconds
FROM speaker_profiles sp
WHERE sp.feed_url = ?
ORDER BY sp.total_appearances DESC;

-- Find when two speakers appeared together
SELECT DISTINCT t.filename, t.created_at
FROM transcription_metadata t
JOIN speaker_occurrences so1 ON so1.transcription_id = t.id
JOIN speaker_occurrences so2 ON so2.transcription_id = t.id
WHERE so1.profile_id = ? AND so2.profile_id = ?;
```

## Configuration

```python
# In settings or environment
SPEAKER_MATCHING_THRESHOLD = 0.85  # Cosine similarity threshold
SPEAKER_EMBEDDING_METHOD = 'speechbrain'  # or 'pyannote', 'resemblyzer'
SPEAKER_MIN_DURATION = 3.0  # Minimum seconds of speech for embedding
SPEAKER_PROFILE_SCOPE = 'feed'  # or 'global' for cross-feed matching
```

## Benefits

1. **Track speakers over time**: See how opinions/statements evolve
2. **Automatic speaker identification**: No manual labeling needed
3. **Search by speaker**: Find all content from specific people
4. **Speaker analytics**: Frequency, duration, co-appearances
5. **Corrections possible**: Merge profiles, verify matches

## Implementation Priority

1. **Phase 1**: Basic embedding extraction and storage
2. **Phase 2**: Automatic matching with confidence scores
3. **Phase 3**: UI for profile management and corrections
4. **Phase 4**: Advanced analytics and timeline views