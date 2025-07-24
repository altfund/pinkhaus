"""
Beige Book - Audio transcription library and CLI tool.
"""

from .transcriber import AudioTranscriber, TranscriptionResult, Segment, create_extended_result
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

# Protobuf support is now built into TranscriptionResult
# Import generated protobuf classes
from .transcription_pb2 import (
    TranscriptionResult as TranscriptionResultProto,
    Segment as SegmentProto,
    FeedMetadata,
    ExtendedTranscriptionResult
)

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
    "create_feed_request",
    # Protobuf support
    "create_extended_result",
    "TranscriptionResultProto",
    "SegmentProto",
    "FeedMetadata",
    "ExtendedTranscriptionResult"
]
__version__ = "0.1.0"