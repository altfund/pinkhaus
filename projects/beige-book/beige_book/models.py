"""
Request and response models - now using betterproto for better performance.
This module re-exports the betterproto implementation for backward compatibility.
"""

# Re-export everything from the betterproto implementation
from .models_betterproto import (
    InputConfig,
    ProcessingConfig,
    OutputConfig,
    FeedOptions,
    DatabaseConfig,
    TranscriptionRequest,
    TranscriptionResponse,
    ProcessingError,
    ProcessingSummary,
    create_file_request,
    create_feed_request,
    InputType,
    Model,
    Format,
    Order,
)

# Make exports explicit for linter
__all__ = [
    "InputConfig",
    "ProcessingConfig",
    "OutputConfig",
    "FeedOptions",
    "DatabaseConfig",
    "TranscriptionRequest",
    "TranscriptionResponse",
    "ProcessingError",
    "ProcessingSummary",
    "create_file_request",
    "create_feed_request",
    "InputType",
    "Model",
    "Format",
    "Order",
]

# For complete backward compatibility, also provide string literals
from typing import Literal

# These are provided for type hints in existing code
InputTypeLiteral = Literal["file", "feed"]
ModelLiteral = Literal["tiny", "base", "small", "medium", "large"]
FormatLiteral = Literal["text", "json", "table", "csv", "toml", "sqlite"]
OrderLiteral = Literal["newest", "oldest"]
