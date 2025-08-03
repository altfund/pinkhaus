"""
Voice embedding extraction for speaker identification.

This module provides functionality to extract voice embeddings (fingerprints)
from audio segments, enabling speaker recognition across recordings.
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torchaudio
from pathlib import Path

try:
    try:
        # Try new import path first (SpeechBrain 1.0+)
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        # Fall back to old import path
        from speechbrain.pretrained import EncoderClassifier
    HAS_SPEECHBRAIN = True
except ImportError:
    HAS_SPEECHBRAIN = False
    print("Warning: SpeechBrain not available. Install with: pip install speechbrain")

try:
    from pyannote.audio import Model as PyannoteModel
    from pyannote.audio import Inference
    HAS_PYANNOTE = True
except ImportError:
    HAS_PYANNOTE = False


class VoiceEmbeddingExtractor:
    """Extract voice embeddings using various methods."""
    
    def __init__(self, method: str = "speechbrain", device: Optional[str] = None):
        """
        Initialize the embedding extractor.
        
        Args:
            method: Extraction method ('speechbrain', 'pyannote', or 'mock')
            device: Device to use ('cpu' or 'cuda'). Auto-detected if None.
        """
        self.method = method
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the appropriate model based on method."""
        if self.method == "speechbrain" and HAS_SPEECHBRAIN:
            # Load pre-trained ECAPA-TDNN model
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
        elif self.method == "pyannote" and HAS_PYANNOTE:
            # Load PyAnnote embedding model
            # Note: Requires HuggingFace token for some models
            model = PyannoteModel.from_pretrained("pyannote/embedding")
            self.model = Inference(model, window="whole")
        elif self.method == "mock":
            # Mock mode for testing
            self.model = None
        else:
            raise ValueError(f"Method '{self.method}' not available or not installed")
    
    def extract_embedding_from_file(
        self, 
        audio_path: str, 
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Extract embedding from an audio file.
        
        Args:
            audio_path: Path to audio file
            start_time: Optional start time in seconds
            end_time: Optional end time in seconds
            
        Returns:
            Tuple of (embedding vector, quality score)
        """
        if self.method == "mock":
            # Return mock embedding for testing
            return np.random.randn(256).astype(np.float32), 1.0
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Extract segment if times provided
        if start_time is not None or end_time is not None:
            start_sample = int((start_time or 0) * sample_rate)
            end_sample = int((end_time or len(waveform[0]) / sample_rate) * sample_rate)
            waveform = waveform[:, start_sample:end_sample]
        
        # Ensure mono audio
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract embedding based on method
        if self.method == "speechbrain":
            # SpeechBrain expects shape (batch, samples)
            if waveform.dim() == 2:
                waveform = waveform.squeeze(0)
            embeddings = self.model.encode_batch(waveform.unsqueeze(0))
            embedding = embeddings.squeeze().cpu().numpy()
        
        elif self.method == "pyannote":
            # PyAnnote expects numpy array
            embedding = self.model({"waveform": waveform, "sample_rate": sample_rate})
            embedding = embedding.cpu().numpy()
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        # Calculate quality score based on audio duration
        duration = waveform.shape[-1] / sample_rate
        quality_score = min(1.0, duration / 10.0)  # Max quality at 10+ seconds
        
        return embedding.astype(np.float32), quality_score
    
    def extract_embeddings_for_speaker(
        self,
        audio_path: str,
        segments: List[Dict[str, any]],
        min_duration: float = 3.0
    ) -> Tuple[Optional[np.ndarray], float, List[int]]:
        """
        Extract embedding for a speaker from multiple segments.
        
        Args:
            audio_path: Path to audio file
            segments: List of segment dictionaries with 'start_time', 'end_time'
            min_duration: Minimum total duration required (seconds)
            
        Returns:
            Tuple of (embedding, total_duration, segment_indices_used)
        """
        # Filter segments by minimum duration
        valid_segments = []
        total_duration = 0.0
        
        for i, seg in enumerate(segments):
            duration = seg['end_time'] - seg['start_time']
            if duration >= 0.5:  # Minimum 0.5 seconds per segment
                valid_segments.append((i, seg))
                total_duration += duration
        
        if total_duration < min_duration:
            return None, total_duration, []
        
        # For mock mode, return mock embedding
        if self.method == "mock":
            indices = [i for i, _ in valid_segments]
            return np.random.randn(256).astype(np.float32), total_duration, indices
        
        # Extract embeddings from each valid segment
        embeddings = []
        weights = []
        indices_used = []
        
        for idx, seg in valid_segments[:10]:  # Limit to 10 segments
            try:
                emb, quality = self.extract_embedding_from_file(
                    audio_path, 
                    seg['start_time'],
                    seg['end_time']
                )
                embeddings.append(emb)
                weights.append(seg['end_time'] - seg['start_time'])
                indices_used.append(idx)
            except Exception as e:
                print(f"Warning: Failed to extract embedding from segment {idx}: {e}")
                continue
        
        if not embeddings:
            return None, total_duration, []
        
        # Combine embeddings using weighted average
        embeddings = np.array(embeddings)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        combined_embedding = np.average(embeddings, axis=0, weights=weights)
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
        
        return combined_embedding.astype(np.float32), total_duration, indices_used
    
    def extract_embeddings_for_transcription(
        self,
        audio_path: str,
        transcription_result: 'TranscriptionResult',
        min_duration: float = 3.0
    ) -> Dict[str, Tuple[np.ndarray, float, List[int]]]:
        """
        Extract embeddings for all speakers in a transcription.
        
        Args:
            audio_path: Path to audio file
            transcription_result: TranscriptionResult with speaker diarization
            min_duration: Minimum duration per speaker
            
        Returns:
            Dict mapping speaker_label to (embedding, duration, segment_indices)
        """
        if not transcription_result._proto.has_speaker_labels:
            return {}
        
        # Group segments by speaker
        speaker_segments = {}
        for i, seg in enumerate(transcription_result.segments):
            if hasattr(seg, 'speaker') and seg.speaker:
                if seg.speaker not in speaker_segments:
                    speaker_segments[seg.speaker] = []
                speaker_segments[seg.speaker].append({
                    'index': i,
                    'start_time': seg.start_ms / 1000.0,
                    'end_time': seg.end_ms / 1000.0,
                    'text': seg.text
                })
        
        # Extract embeddings for each speaker
        speaker_embeddings = {}
        for speaker_label, segments in speaker_segments.items():
            embedding, duration, indices = self.extract_embeddings_for_speaker(
                audio_path, segments, min_duration
            )
            if embedding is not None:
                speaker_embeddings[speaker_label] = (embedding, duration, indices)
        
        return speaker_embeddings


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding vector
        emb2: Second embedding vector
        
    Returns:
        Cosine similarity score (0-1, higher is more similar)
    """
    # Ensure normalized
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    # Calculate cosine similarity
    similarity = np.dot(emb1, emb2)
    
    # Convert to 0-1 range
    return (similarity + 1.0) / 2.0


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize numpy embedding to bytes for database storage."""
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(embedding_bytes: bytes, dimension: int = 256) -> np.ndarray:
    """Deserialize bytes back to numpy embedding."""
    return np.frombuffer(embedding_bytes, dtype=np.float32).reshape(-1)[:dimension]