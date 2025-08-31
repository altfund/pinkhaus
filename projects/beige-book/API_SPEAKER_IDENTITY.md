# Speaker Identity API Reference

This document provides detailed API documentation for all speaker identity tracking methods and classes in the beige-book library.

## Table of Contents

1. [Database Methods](#database-methods)
2. [Voice Embeddings](#voice-embeddings)
3. [Speaker Matching](#speaker-matching)
4. [Transcriber Integration](#transcriber-integration)

## Database Methods

### TranscriptionDatabase

The `TranscriptionDatabase` class has been extended with speaker identity methods.

#### create_speaker_identity_tables()

Creates the database tables required for speaker identity tracking.

```python
def create_speaker_identity_tables(
    self,
    profiles_table: str = "speaker_profiles",
    embeddings_table: str = "speaker_embeddings", 
    occurrences_table: str = "speaker_occurrences",
    profile_metadata_table: str = "speaker_metadata",
    segments_table: str = "transcription_segments",
) -> None
```

**Parameters:**
- `profiles_table` (str): Name for the speaker profiles table
- `embeddings_table` (str): Name for the voice embeddings table
- `occurrences_table` (str): Name for the speaker occurrences table
- `profile_metadata_table` (str): Name for the speaker metadata table
- `segments_table` (str): Name of existing segments table to reference

**Example:**
```python
db = TranscriptionDatabase("podcast.db")
db.create_tables()
db.create_speaker_identity_tables()
```

#### create_speaker_profile()

Creates or retrieves a speaker profile.

```python
def create_speaker_profile(
    self,
    display_name: str,
    feed_url: Optional[str] = None,
    canonical_label: Optional[str] = None,
    is_active: bool = True
) -> int
```

**Parameters:**
- `display_name` (str): Display name for the speaker
- `feed_url` (str, optional): RSS feed URL to scope the speaker to
- `canonical_label` (str, optional): Canonical role (HOST, COHOST, GUEST, etc.)
- `is_active` (bool): Whether the profile is active

**Returns:**
- `int`: Profile ID

**Example:**
```python
host_id = db.create_speaker_profile(
    display_name="John Doe",
    feed_url="https://podcast.example.com/feed.rss",
    canonical_label="HOST"
)
```

#### add_speaker_embedding()

Adds a voice embedding to a speaker profile.

```python
def add_speaker_embedding(
    self,
    profile_id: int,
    embedding: bytes,
    embedding_dimension: int,
    quality_score: float = 1.0,
    extraction_method: str = "unknown",
    audio_source: Optional[str] = None
) -> int
```

**Parameters:**
- `profile_id` (int): Speaker profile ID
- `embedding` (bytes): Serialized embedding vector
- `embedding_dimension` (int): Dimension of the embedding (typically 256)
- `quality_score` (float): Quality score of the embedding (0-1)
- `extraction_method` (str): Method used to extract embedding
- `audio_source` (str, optional): Source audio file path

**Returns:**
- `int`: Embedding ID

**Example:**
```python
from beige_book.voice_embeddings import VoiceEmbeddingExtractor, serialize_embedding

extractor = VoiceEmbeddingExtractor()
embedding, quality = extractor.extract_embedding_from_file("host_intro.wav")

embedding_id = db.add_speaker_embedding(
    profile_id=host_id,
    embedding=serialize_embedding(embedding),
    embedding_dimension=256,
    quality_score=quality,
    extraction_method="speechbrain",
    audio_source="host_intro.wav"
)
```

#### link_speaker_occurrence()

Links a temporary speaker label to a persistent profile.

```python
def link_speaker_occurrence(
    self,
    transcription_id: int,
    temporary_label: str,
    profile_id: int,
    confidence: float,
    is_verified: bool = False
) -> int
```

**Parameters:**
- `transcription_id` (int): Transcription ID
- `temporary_label` (str): Temporary label (e.g., "SPEAKER_0")
- `profile_id` (int): Speaker profile ID to link to
- `confidence` (float): Confidence score (0-1)
- `is_verified` (bool): Whether manually verified

**Returns:**
- `int`: Occurrence ID

#### get_speaker_profiles_for_feed()

Get all speaker profiles for a specific feed.

```python
def get_speaker_profiles_for_feed(
    self,
    feed_url: str,
    include_inactive: bool = False
) -> List[Dict[str, Any]]
```

**Parameters:**
- `feed_url` (str): RSS feed URL
- `include_inactive` (bool): Include inactive profiles

**Returns:**
- `List[Dict]`: List of profile dictionaries with fields:
  - `id`: Profile ID
  - `display_name`: Speaker name
  - `canonical_label`: Role label
  - `total_appearances`: Number of episodes
  - `total_duration`: Total speaking time
  - `first_seen`: First appearance date
  - `last_seen`: Last appearance date

#### get_speaker_history()

Get appearance history for a speaker.

```python
def get_speaker_history(
    self,
    profile_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]
```

**Parameters:**
- `profile_id` (int): Speaker profile ID
- `start_date` (str, optional): Start date (YYYY-MM-DD)
- `end_date` (str, optional): End date (YYYY-MM-DD)
- `limit` (int): Maximum results

**Returns:**
- `List[Dict]`: List of appearances with transcription details

#### get_speaker_statements()

Get all statements made by a speaker.

```python
def get_speaker_statements(
    self,
    profile_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_duration: Optional[float] = None,
    search_text: Optional[str] = None
) -> List[Dict[str, Any]]
```

**Parameters:**
- `profile_id` (int): Speaker profile ID
- `start_date` (str, optional): Start date filter
- `end_date` (str, optional): End date filter
- `min_duration` (float, optional): Minimum segment duration
- `search_text` (str, optional): Text search filter

**Returns:**
- `List[Dict]`: List of statement dictionaries

#### get_speaker_embeddings()

Get all embeddings for a speaker profile.

```python
def get_speaker_embeddings(
    self,
    profile_id: int
) -> List[Dict[str, Any]]
```

**Parameters:**
- `profile_id` (int): Speaker profile ID

**Returns:**
- `List[Dict]`: List of embedding records

## Voice Embeddings

### VoiceEmbeddingExtractor

Extracts voice embeddings from audio files.

```python
class VoiceEmbeddingExtractor:
    def __init__(
        self, 
        method: str = "speechbrain", 
        device: Optional[str] = None
    )
```

**Parameters:**
- `method` (str): Extraction method ('speechbrain', 'pyannote', or 'mock')
- `device` (str, optional): Device to use ('cpu' or 'cuda')

#### extract_embedding_from_file()

Extract embedding from an audio file segment.

```python
def extract_embedding_from_file(
    self, 
    audio_path: str, 
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> Tuple[np.ndarray, float]
```

**Parameters:**
- `audio_path` (str): Path to audio file
- `start_time` (float, optional): Start time in seconds
- `end_time` (float, optional): End time in seconds

**Returns:**
- `Tuple[np.ndarray, float]`: (embedding vector, quality score)

#### extract_embeddings_for_speaker()

Extract embedding from multiple segments of a speaker.

```python
def extract_embeddings_for_speaker(
    self,
    audio_path: str,
    segments: List[Dict[str, any]],
    min_duration: float = 3.0
) -> Tuple[Optional[np.ndarray], float, List[int]]
```

**Parameters:**
- `audio_path` (str): Path to audio file
- `segments` (List[Dict]): Segment dictionaries with 'start_time', 'end_time'
- `min_duration` (float): Minimum total duration required

**Returns:**
- `Tuple`: (embedding, total_duration, segment_indices_used)

### Utility Functions

#### cosine_similarity()

Calculate similarity between embeddings.

```python
def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float
```

**Parameters:**
- `emb1` (np.ndarray): First embedding
- `emb2` (np.ndarray): Second embedding

**Returns:**
- `float`: Similarity score (0-1, higher is more similar)

#### serialize_embedding()

Convert numpy array to bytes for storage.

```python
def serialize_embedding(embedding: np.ndarray) -> bytes
```

#### deserialize_embedding()

Convert bytes back to numpy array.

```python
def deserialize_embedding(
    embedding_bytes: bytes, 
    dimension: int = 256
) -> np.ndarray
```

## Speaker Matching

### SpeakerMatcher

Matches speakers across recordings using voice embeddings.

```python
class SpeakerMatcher:
    def __init__(
        self,
        db: TranscriptionDatabase,
        threshold: float = 0.85,
        embedding_method: str = "speechbrain"
    )
```

**Parameters:**
- `db` (TranscriptionDatabase): Database instance
- `threshold` (float): Similarity threshold for matching
- `embedding_method` (str): Method for embedding extraction

#### find_best_match()

Find best matching profiles for an embedding.

```python
def find_best_match(
    self,
    embedding: np.ndarray,
    feed_url: Optional[str] = None,
    top_k: int = 5
) -> List[Tuple[int, float]]
```

**Parameters:**
- `embedding` (np.ndarray): Voice embedding to match
- `feed_url` (str, optional): Limit to specific feed
- `top_k` (int): Number of top matches to return

**Returns:**
- `List[Tuple[int, float]]`: List of (profile_id, similarity) tuples

#### match_speaker()

Match or create speaker profile for an embedding.

```python
def match_speaker(
    self,
    embedding: np.ndarray,
    feed_url: Optional[str] = None,
    create_if_not_found: bool = True,
    speaker_hint: Optional[str] = None
) -> Tuple[Optional[int], float]
```

**Parameters:**
- `embedding` (np.ndarray): Voice embedding
- `feed_url` (str, optional): Feed URL for scoping
- `create_if_not_found` (bool): Create new profile if no match
- `speaker_hint` (str, optional): Name hint for new profile

**Returns:**
- `Tuple[Optional[int], float]`: (profile_id, confidence)

#### identify_speakers_in_transcription()

Identify all speakers in a transcription.

```python
def identify_speakers_in_transcription(
    self,
    transcription_id: int,
    speaker_embeddings: Dict[str, Tuple[np.ndarray, float, List[int]]],
    feed_url: Optional[str] = None
) -> Dict[str, Tuple[int, float]]
```

**Parameters:**
- `transcription_id` (int): Transcription ID
- `speaker_embeddings` (Dict): Speaker embeddings from transcription
- `feed_url` (str, optional): Feed URL for scoping

**Returns:**
- `Dict[str, Tuple[int, float]]`: Mapping of labels to (profile_id, confidence)

#### merge_speaker_profiles()

Merge duplicate speaker profiles.

```python
def merge_speaker_profiles(
    self,
    profile_id_keep: int,
    profile_id_merge: int
) -> bool
```

**Parameters:**
- `profile_id_keep` (int): Profile to keep
- `profile_id_merge` (int): Profile to merge and delete

**Returns:**
- `bool`: Success status

## Transcriber Integration

### AudioTranscriber

The transcriber has been extended with speaker identification support.

#### transcribe_file()

```python
def transcribe_file(
    self,
    filepath: str,
    verbose: bool = False,
    enable_diarization: bool = False,
    hf_token: str = None,
    enable_speaker_identification: bool = False,
    feed_url: Optional[str] = None
) -> TranscriptionResult
```

**New Parameters:**
- `enable_speaker_identification` (bool): Enable speaker identity tracking
- `feed_url` (str, optional): Feed URL for speaker scoping

### TranscriptionDatabase.save_transcription()

The save method now automatically performs speaker identification.

```python
def save_transcription(
    self,
    result: TranscriptionResult,
    metadata_table: str = "transcription_metadata",
    segments_table: str = "transcription_segments",
    speakers_table: str = "speakers",
    feed_url: Optional[str] = None,
    feed_item_id: Optional[str] = None,
    feed_item_title: Optional[str] = None,
    feed_item_published: Optional[str] = None,
) -> int
```

**Behavior:**
- If `result` contains speaker embeddings, automatic identification is performed
- Speaker occurrences are linked to profiles based on voice similarity
- New profiles are created for unmatched speakers

## Environment Variables

Configure speaker identity behavior with these environment variables:

- `SPEAKER_EMBEDDING_METHOD`: Embedding extraction method ('speechbrain', 'pyannote', 'mock')
- `SPEAKER_MATCHING_THRESHOLD`: Similarity threshold for matching (0-1, default: 0.85)
- `SPEAKER_MIN_DURATION`: Minimum speech duration for embedding extraction (default: 3.0 seconds)
- `HF_TOKEN`: HuggingFace token for certain models

## Error Handling

All methods may raise:
- `sqlite3.Error`: Database errors
- `ValueError`: Invalid parameters
- `FileNotFoundError`: Audio file not found
- `RuntimeError`: Model loading or processing errors

## Complete Example

```python
from beige_book.transcriber import AudioTranscriber
from beige_book.database import TranscriptionDatabase
from beige_book.speaker_matcher import SpeakerMatcher
from beige_book.voice_embeddings import VoiceEmbeddingExtractor, serialize_embedding

# Initialize components
db = TranscriptionDatabase("podcast.db")
db.create_tables()
db.create_speaker_identity_tables()

# Pre-register known speakers
host_id = db.create_speaker_profile(
    "John Doe",
    feed_url="https://podcast.example.com/feed.rss",
    canonical_label="HOST"
)

# Add reference embedding
extractor = VoiceEmbeddingExtractor()
embedding, quality = extractor.extract_embedding_from_file("host_intro.wav")
db.add_speaker_embedding(
    host_id,
    serialize_embedding(embedding),
    256,
    quality_score=quality
)

# Transcribe with speaker identification
transcriber = AudioTranscriber(model_name="tiny")
result = transcriber.transcribe_file(
    "episode_001.mp3",
    enable_diarization=True,
    enable_speaker_identification=True,
    feed_url="https://podcast.example.com/feed.rss"
)

# Save (automatically matches speakers)
trans_id = db.save_transcription(
    result,
    feed_url="https://podcast.example.com/feed.rss"
)

# Query results
profiles = db.get_speaker_profiles_for_feed("https://podcast.example.com/feed.rss")
for profile in profiles:
    print(f"{profile['display_name']}: {profile['total_appearances']} episodes")
    
    statements = db.get_speaker_statements(profile['id'])
    for stmt in statements[:5]:
        print(f"  - {stmt['text']}")
```