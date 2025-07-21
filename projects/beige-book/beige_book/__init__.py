"""
Beige Book - Audio transcription library and CLI tool.
"""

from .transcriber import AudioTranscriber, TranscriptionResult, Segment
from .database import TranscriptionDatabase
from .feed_parser import FeedParser, FeedItem
from .downloader import AudioDownloader
from .models import (
    TranscriptionRequest, TranscriptionResponse,
    InputConfig, ProcessingConfig, OutputConfig,
    FeedOptions, DatabaseConfig,
    ProcessingError, ProcessingSummary,
    create_file_request, create_feed_request
)
from .service import TranscriptionService, OutputFormatter

__all__ = [
    # Original exports
    "AudioTranscriber",
    "TranscriptionResult",
    "Segment",
    "TranscriptionDatabase",
    "FeedParser",
    "FeedItem",
    "AudioDownloader",
    # New request/response models
    "TranscriptionRequest",
    "TranscriptionResponse",
    "InputConfig",
    "ProcessingConfig",
    "OutputConfig",
    "FeedOptions",
    "DatabaseConfig",
    "ProcessingError",
    "ProcessingSummary",
    # Service layer
    "TranscriptionService",
    "OutputFormatter",
    # Convenience functions
    "create_file_request",
    "create_feed_request"
]
__version__ = "0.1.0"