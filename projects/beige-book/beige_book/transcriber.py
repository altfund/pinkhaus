"""
Transcription library - now using betterproto for better performance.
This module re-exports the betterproto implementation.
"""

# Re-export everything from the betterproto implementation
from .transcriber_betterproto import (
    AudioTranscriber,
    TranscriptionResult,
    Segment,
    create_extended_result,
)

# Make exports explicit for linter
__all__ = [
    "AudioTranscriber",
    "TranscriptionResult",
    "Segment",
    "create_extended_result",
]
