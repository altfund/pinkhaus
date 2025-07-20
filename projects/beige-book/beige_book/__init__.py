"""
Beige Book - Audio transcription library and CLI tool.
"""

from .transcriber import AudioTranscriber, TranscriptionResult, Segment
from .database import TranscriptionDatabase
from .feed_parser import FeedParser, FeedItem
from .downloader import AudioDownloader

__all__ = [
    "AudioTranscriber",
    "TranscriptionResult",
    "Segment",
    "TranscriptionDatabase",
    "FeedParser",
    "FeedItem",
    "AudioDownloader"
]
__version__ = "0.1.0"