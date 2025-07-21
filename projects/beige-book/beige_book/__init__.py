"""
Beige Book - Audio transcription library and CLI tool.
"""

from .transcriber import AudioTranscriber, TranscriptionResult, Segment
from .database import TranscriptionDatabase

__all__ = [
    "AudioTranscriber",
    "TranscriptionResult",
    "Segment",
    "TranscriptionDatabase",
]
__version__ = "0.1.0"
