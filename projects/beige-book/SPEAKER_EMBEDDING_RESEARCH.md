# Speaker Embedding Research and Implementation Guide

## Overview

This document summarizes research on speaker embedding capabilities for voice fingerprinting, speaker verification, and identification across multiple recordings.

## Current Implementation

The codebase currently has:
- **PyAnnote Audio Integration**: Basic speaker diarization (who speaks when) in `beige_book/speaker_diarizer.py`
- **Python 3.11 Requirement**: Due to dependency constraints with pyannote-audio
- **No Speaker Embedding Extraction**: Current implementation only performs diarization, not embedding extraction

## PyAnnote Speaker Embedding Capabilities

### 1. PyAnnote Embedding Model
PyAnnote provides a pre-trained embedding model (`pyannote/embedding`) based on x-vector TDNN architecture with SincNet features.

**Key Features:**
- Extracts 256-dimensional embeddings
- 2.8% EER on VoxCeleb 1 test set (without VAD or PLDA)
- Can extract embeddings from whole files or specific segments

**Implementation Example:**
```python
from pyannote.audio import Inference
from pyannote.core import Segment

# Initialize inference
inference = Inference("pyannote/embedding", window="whole")

# Extract embedding from whole file
embedding1 = inference("speaker1.wav")  # Returns (1 x D) numpy array

# Extract embedding from specific segment
excerpt = Segment(13.37, 19.81)
embedding = inference.crop("audio.wav", excerpt)
```

### 2. Comparing Embeddings
```python
from scipy.spatial.distance import cdist

# Compare embeddings using cosine distance
distance = cdist(embedding1, embedding2, metric="cosine")[0,0]
# Lower distance = more similar voices
```

## Alternative Libraries

### 1. Resemblyzer
**Pros:**
- Simple API, easy to use
- Fast execution (1000x real-time on GTX 1080)
- 256-dimensional embeddings
- Works on CPU

**Installation:**
```bash
pip install resemblyzer
```

**Usage:**
```python
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

# Load and preprocess audio
wav = preprocess_wav(Path("audio.wav"))

# Extract embedding
encoder = VoiceEncoder()
embed = encoder.embed_utterance(wav)
```

### 2. SpeechBrain
**Pros:**
- State-of-the-art ECAPA-TDNN model
- Pre-trained on VoxCeleb dataset
- PyTorch-based, GPU accelerated
- Already in project dependencies (found in uv.lock)

**Usage:**
```python
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

# Load model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

# Extract embedding
signal, fs = torchaudio.load('audio.wav')
embedding = classifier.encode_batch(signal)
```

## Recommended Implementation Approach

### 1. Extend Current Speaker Diarizer
Add embedding extraction capabilities to the existing `SpeakerDiarizer` class:

```python
class SpeakerDiarizer:
    def __init__(self, auth_token=None, device=None):
        # ... existing code ...
        self._embedding_model = None
        
    def _load_embedding_model(self):
        """Load speaker embedding model."""
        if self._embedding_model is None:
            from pyannote.audio import Inference
            self._embedding_model = Inference(
                "pyannote/embedding",
                window="whole",
                device=self.device
            )
    
    def extract_speaker_embedding(self, audio_path, segment=None):
        """Extract speaker embedding from audio."""
        self._load_embedding_model()
        
        if segment:
            return self._embedding_model.crop(audio_path, segment)
        else:
            return self._embedding_model(audio_path)
    
    def compare_speakers(self, embedding1, embedding2):
        """Compare two speaker embeddings."""
        from scipy.spatial.distance import cosine
        similarity = 1 - cosine(embedding1, embedding2)
        return similarity
```

### 2. Create Speaker Database
Store speaker embeddings for comparison across recordings:

```python
class SpeakerDatabase:
    def __init__(self, db_path="speakers.pkl"):
        self.db_path = db_path
        self.speakers = {}
        
    def add_speaker(self, name, embedding):
        """Add a speaker to the database."""
        self.speakers[name] = embedding
        self.save()
        
    def identify_speaker(self, embedding, threshold=0.7):
        """Identify a speaker from embedding."""
        best_match = None
        best_similarity = 0
        
        for name, stored_embedding in self.speakers.items():
            similarity = 1 - cosine(embedding, stored_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
                
        if best_similarity > threshold:
            return best_match, best_similarity
        return "Unknown", best_similarity
    
    def save(self):
        """Save database to disk."""
        import pickle
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.speakers, f)
            
    def load(self):
        """Load database from disk."""
        import pickle
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                self.speakers = pickle.load(f)
```

### 3. Integration with Transcription Pipeline
Enhance the transcription process to identify known speakers:

```python
def transcribe_with_speaker_identification(
    audio_path,
    speaker_database,
    transcriber,
    diarizer
):
    # Perform diarization
    diarization = diarizer.diarize_file(audio_path)
    
    # Extract embeddings for each speaker segment
    speaker_embeddings = {}
    for segment in diarization.segments:
        if segment.speaker not in speaker_embeddings:
            # Extract embedding for this speaker's segments
            embedding = diarizer.extract_speaker_embedding(
                audio_path,
                segment
            )
            speaker_embeddings[segment.speaker] = embedding
    
    # Identify speakers
    speaker_mapping = {}
    for speaker_label, embedding in speaker_embeddings.items():
        identified_name, confidence = speaker_database.identify_speaker(embedding)
        speaker_mapping[speaker_label] = identified_name
    
    # Transcribe with identified speakers
    result = transcriber.transcribe_file(audio_path)
    
    # Update segments with identified speakers
    for segment in result.segments:
        if segment.speaker in speaker_mapping:
            segment.identified_speaker = speaker_mapping[segment.speaker]
    
    return result
```

## Key Use Cases

1. **Podcast Host Identification**: Automatically identify recurring hosts across episodes
2. **Meeting Speaker Recognition**: Identify participants in recurring meetings
3. **Voice Authentication**: Verify speaker identity for security applications
4. **Content Filtering**: Extract only segments from specific speakers
5. **Speaker Analytics**: Track speaking time and patterns for known speakers

## Performance Considerations

- **Embedding Extraction**: ~10-50ms per utterance on GPU
- **Comparison**: <1ms per comparison (cosine distance)
- **Storage**: 256 floats per speaker (~1KB)
- **Accuracy**: Expect 95%+ accuracy with good quality audio

## Next Steps

1. **Choose Embedding Model**: PyAnnote (integrated) vs SpeechBrain (better performance)
2. **Implement Speaker Database**: Persistent storage for known speaker embeddings
3. **Add CLI Options**: `--identify-speakers`, `--speaker-db path/to/db`
4. **Create Management Tools**: Add/remove/list known speakers
5. **Enhance Output**: Include identified speaker names in transcription results