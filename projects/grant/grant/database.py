# This module now uses pinkhaus-models for all database operations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from pinkhaus_models import TranscriptionDatabase, VectorStore
from pinkhaus_models import TranscriptionMetadata as _TranscriptionMetadata
from pinkhaus_models import TranscriptionSegment


# Type aliases for backward compatibility
PodcastTranscription = _TranscriptionMetadata


@dataclass
class TextChunk:
    """Text chunk with embedding for vector similarity search."""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }


class PodcastDatabase(TranscriptionDatabase):
    """Database for podcast transcriptions.
    
    This class extends TranscriptionDatabase from pinkhaus-models.
    """
    
    def __init__(self, db_path: str):
        """Initialize with database path."""
        super().__init__(db_path=db_path)
        self.db_path = db_path
    
    # Additional grant-specific methods can be added here if needed


# Re-export VectorStore from pinkhaus-models for backward compatibility
# The imported VectorStore class already has all the needed functionality
