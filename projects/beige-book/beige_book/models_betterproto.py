"""
Request and response models using betterproto.
Provides backward compatibility layer for existing code.
"""

from typing import Optional, Dict, Any
from datetime import datetime
import json

from .proto_models import (
    InputConfig as _InputConfig,
    InputConfigInputType,
    ProcessingConfig as _ProcessingConfig,
    ProcessingConfigModel,
    OutputConfig as _OutputConfig,
    OutputConfigFormat,
    FeedOptions as _FeedOptions,
    FeedOptionsOrder,
    DatabaseConfig as _DatabaseConfig,
    TranscriptionRequest as _TranscriptionRequest,
    TranscriptionResponse as _TranscriptionResponse,
    ProcessingError as _ProcessingError,
    ProcessingSummary as _ProcessingSummary,
    # Import needed for TranscriptionResponse type hints
    TranscriptionResult,  # noqa: F401
)


# Re-export enums with cleaner names
class InputType:
    FILE = InputConfigInputType.INPUT_TYPE_FILE
    FEED = InputConfigInputType.INPUT_TYPE_FEED


class Model:
    TINY = ProcessingConfigModel.MODEL_TINY
    BASE = ProcessingConfigModel.MODEL_BASE
    SMALL = ProcessingConfigModel.MODEL_SMALL
    MEDIUM = ProcessingConfigModel.MODEL_MEDIUM
    LARGE = ProcessingConfigModel.MODEL_LARGE


class Format:
    TEXT = OutputConfigFormat.FORMAT_TEXT
    JSON = OutputConfigFormat.FORMAT_JSON
    TABLE = OutputConfigFormat.FORMAT_TABLE
    CSV = OutputConfigFormat.FORMAT_CSV
    TOML = OutputConfigFormat.FORMAT_TOML
    SQLITE = OutputConfigFormat.FORMAT_SQLITE


class Order:
    NEWEST = FeedOptionsOrder.ORDER_NEWEST
    OLDEST = FeedOptionsOrder.ORDER_OLDEST


# Wrapper classes that add validation methods
class InputConfig(_InputConfig):
    """Configuration for input sources"""

    def __init__(self, **kwargs):
        """Initialize with string conversion support"""
        # Convert string type to enum if needed
        if "type" in kwargs and isinstance(kwargs["type"], str):
            type_map = {
                "file": InputConfigInputType.INPUT_TYPE_FILE,
                "feed": InputConfigInputType.INPUT_TYPE_FEED,
            }
            kwargs["type"] = type_map.get(
                kwargs["type"], InputConfigInputType.INPUT_TYPE_UNSPECIFIED
            )
        super().__init__(**kwargs)

    def validate(self):
        """Validate the input configuration"""
        if not self.source:
            raise ValueError("Input source cannot be empty")
        if self.type not in [
            InputConfigInputType.INPUT_TYPE_FILE,
            InputConfigInputType.INPUT_TYPE_FEED,
        ]:
            raise ValueError(f"Invalid input type: {self.type}")


class FeedOptions(_FeedOptions):
    """Options specific to feed processing"""

    def __init__(self, **kwargs):
        """Initialize with string conversion support"""
        # Convert string order to enum if needed
        if "order" in kwargs and isinstance(kwargs["order"], str):
            order_map = {
                "newest": FeedOptionsOrder.ORDER_NEWEST,
                "oldest": FeedOptionsOrder.ORDER_OLDEST,
            }
            kwargs["order"] = order_map.get(
                kwargs["order"], FeedOptionsOrder.ORDER_UNSPECIFIED
            )
        # Set defaults for missing fields
        kwargs.setdefault("limit", 0)
        kwargs.setdefault("max_retries", 3)
        kwargs.setdefault("initial_delay", 1.0)
        super().__init__(**kwargs)

    def validate(self):
        """Validate feed options"""
        if self.limit is not None and self.limit < 0:
            raise ValueError("Feed limit must be non-negative")
        if self.order not in [
            FeedOptionsOrder.ORDER_NEWEST,
            FeedOptionsOrder.ORDER_OLDEST,
            FeedOptionsOrder.ORDER_UNSPECIFIED,
        ]:
            raise ValueError(f"Invalid order: {self.order}")
        if self.max_retries is not None and self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
        if self.initial_delay is not None and self.initial_delay < 0:
            raise ValueError("Initial delay cannot be negative")
        if self.date_threshold:
            # Validate ISO8601 format
            try:
                from datetime import datetime

                datetime.fromisoformat(self.date_threshold.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError(f"Invalid ISO8601 date format: {self.date_threshold}")


class ProcessingConfig(_ProcessingConfig):
    """Configuration for processing options"""

    def __init__(self, **kwargs):
        """Initialize with string conversion support"""
        # Convert string model to enum if needed
        if "model" in kwargs and isinstance(kwargs["model"], str):
            model_map = {
                "tiny": ProcessingConfigModel.MODEL_TINY,
                "base": ProcessingConfigModel.MODEL_BASE,
                "small": ProcessingConfigModel.MODEL_SMALL,
                "medium": ProcessingConfigModel.MODEL_MEDIUM,
                "large": ProcessingConfigModel.MODEL_LARGE,
            }
            kwargs["model"] = model_map.get(
                kwargs["model"], ProcessingConfigModel.MODEL_TINY
            )
        super().__init__(**kwargs)

    def validate(self):
        """Validate processing configuration"""
        valid_models = [
            ProcessingConfigModel.MODEL_TINY,
            ProcessingConfigModel.MODEL_BASE,
            ProcessingConfigModel.MODEL_SMALL,
            ProcessingConfigModel.MODEL_MEDIUM,
            ProcessingConfigModel.MODEL_LARGE,
        ]
        if self.model not in valid_models:
            raise ValueError(f"Invalid model: {self.model}")
        if self.feed_options and hasattr(self.feed_options, "validate"):
            self.feed_options.validate()


class DatabaseConfig(_DatabaseConfig):
    """Database-specific configuration"""

    def validate(self):
        """Validate database configuration"""
        if not self.db_path:
            raise ValueError("Database path cannot be empty")
        if not self.metadata_table:
            raise ValueError("Metadata table name cannot be empty")
        if not self.segments_table:
            raise ValueError("Segments table name cannot be empty")


class OutputConfig(_OutputConfig):
    """Configuration for output handling"""

    def __init__(self, **kwargs):
        """Initialize with string conversion support"""
        # Convert string format to enum if needed
        if "format" in kwargs and isinstance(kwargs["format"], str):
            format_map = {
                "text": OutputConfigFormat.FORMAT_TEXT,
                "json": OutputConfigFormat.FORMAT_JSON,
                "table": OutputConfigFormat.FORMAT_TABLE,
                "csv": OutputConfigFormat.FORMAT_CSV,
                "toml": OutputConfigFormat.FORMAT_TOML,
                "sqlite": OutputConfigFormat.FORMAT_SQLITE,
            }
            kwargs["format"] = format_map.get(
                kwargs["format"], OutputConfigFormat.FORMAT_TEXT
            )
        # Set empty string as default for destination
        kwargs.setdefault("destination", "")
        super().__init__(**kwargs)

    def validate(self):
        """Validate output configuration"""
        valid_formats = [
            OutputConfigFormat.FORMAT_TEXT,
            OutputConfigFormat.FORMAT_JSON,
            OutputConfigFormat.FORMAT_TABLE,
            OutputConfigFormat.FORMAT_CSV,
            OutputConfigFormat.FORMAT_TOML,
            OutputConfigFormat.FORMAT_SQLITE,
        ]
        if self.format not in valid_formats:
            raise ValueError(f"Invalid format: {self.format}")

        # SQLite format requires database config
        if self.format == OutputConfigFormat.FORMAT_SQLITE and not self.database:
            raise ValueError("SQLite format requires database configuration")

        if self.database and hasattr(self.database, "validate"):
            self.database.validate()


class TranscriptionRequest(_TranscriptionRequest):
    """Main request object for all transcription operations"""

    def validate(self):
        """Validate the entire request"""
        if hasattr(self.input, "validate"):
            self.input.validate()
        if hasattr(self.processing, "validate"):
            self.processing.validate()
        if hasattr(self.output, "validate"):
            self.output.validate()

        # Cross-field validation
        if (
            self.input.type == InputConfigInputType.INPUT_TYPE_FILE
            and self.processing.feed_options
        ):
            if (
                (
                    self.processing.feed_options.limit is not None
                    and self.processing.feed_options.limit > 0
                )
                or self.processing.feed_options.order
                != FeedOptionsOrder.ORDER_UNSPECIFIED
            ):
                raise ValueError("Feed options provided for file input")
        if (
            self.input.type == InputConfigInputType.INPUT_TYPE_FEED
            and not self.processing.feed_options
        ):
            # Create default feed options if not provided
            self.processing.feed_options = FeedOptions()

    def to_json(self) -> str:
        """Convert request to JSON string"""
        return self.to_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TranscriptionRequest":
        """Create request from JSON string"""
        return cls().from_json(json_str)

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary"""
        return super().to_dict()


class ProcessingError(_ProcessingError):
    """Error information for failed items"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO timestamp"""
        # Use betterproto's to_dict() from parent class
        data = super().to_dict()
        if isinstance(self.timestamp, datetime):
            data["timestamp"] = self.timestamp.isoformat()
        return data


class ProcessingSummary(_ProcessingSummary):
    """Summary statistics for batch operations"""

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_items == 0:
            return 0.0
        return (self.processed / self.total_items) * 100


class TranscriptionResponse(_TranscriptionResponse):
    """Response object for transcription operations"""

    def add_error(self, source: str, error_type: str, message: str):
        """Add an error to the response"""
        error = ProcessingError(
            source=source,
            error_type=error_type,
            message=message,
            timestamp=datetime.now(),
        )
        self.errors.append(error)

    def to_json(self) -> str:
        """Convert response to JSON string"""
        # Convert results to dicts if they have to_dict method
        data = {
            "success": self.success,
            "results": [
                r.to_dict() if hasattr(r, "to_dict") else r for r in self.results
            ],
            "errors": [
                e.to_dict() if hasattr(e, "to_dict") else e for e in self.errors
            ],
            "summary": self.summary.to_dict()
            if self.summary and hasattr(self.summary, "to_dict")
            else None,
        }
        return json.dumps(data, indent=2)


# Convenience functions for creating requests
def create_file_request(
    filename: str,
    model: str = "tiny",
    format: str = "text",
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> TranscriptionRequest:
    """Create a request for transcribing a single file"""
    # Map string values to enums
    model_map = {
        "tiny": ProcessingConfigModel.MODEL_TINY,
        "base": ProcessingConfigModel.MODEL_BASE,
        "small": ProcessingConfigModel.MODEL_SMALL,
        "medium": ProcessingConfigModel.MODEL_MEDIUM,
        "large": ProcessingConfigModel.MODEL_LARGE,
    }
    format_map = {
        "text": OutputConfigFormat.FORMAT_TEXT,
        "json": OutputConfigFormat.FORMAT_JSON,
        "table": OutputConfigFormat.FORMAT_TABLE,
        "csv": OutputConfigFormat.FORMAT_CSV,
        "toml": OutputConfigFormat.FORMAT_TOML,
        "sqlite": OutputConfigFormat.FORMAT_SQLITE,
    }

    return TranscriptionRequest(
        input=InputConfig(type=InputConfigInputType.INPUT_TYPE_FILE, source=filename),
        processing=ProcessingConfig(
            model=model_map.get(model, ProcessingConfigModel.MODEL_TINY),
            verbose=verbose,
        ),
        output=OutputConfig(
            format=format_map.get(format, OutputConfigFormat.FORMAT_TEXT),
            destination=output_path or "",
        ),
    )


def create_feed_request(
    toml_path: str,
    model: str = "tiny",
    format: str = "text",
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    order: str = "newest",
    verbose: bool = False,
    db_path: Optional[str] = None,
    date_threshold: Optional[str] = None,
) -> TranscriptionRequest:
    """Create a request for processing RSS feeds"""
    # Map string values to enums
    model_map = {
        "tiny": ProcessingConfigModel.MODEL_TINY,
        "base": ProcessingConfigModel.MODEL_BASE,
        "small": ProcessingConfigModel.MODEL_SMALL,
        "medium": ProcessingConfigModel.MODEL_MEDIUM,
        "large": ProcessingConfigModel.MODEL_LARGE,
    }
    format_map = {
        "text": OutputConfigFormat.FORMAT_TEXT,
        "json": OutputConfigFormat.FORMAT_JSON,
        "table": OutputConfigFormat.FORMAT_TABLE,
        "csv": OutputConfigFormat.FORMAT_CSV,
        "toml": OutputConfigFormat.FORMAT_TOML,
        "sqlite": OutputConfigFormat.FORMAT_SQLITE,
    }
    order_map = {
        "newest": FeedOptionsOrder.ORDER_NEWEST,
        "oldest": FeedOptionsOrder.ORDER_OLDEST,
    }

    database_config = None
    if db_path or format == "sqlite":
        database_config = DatabaseConfig(db_path=db_path or "beige_book_feeds.db")

    return TranscriptionRequest(
        input=InputConfig(type=InputConfigInputType.INPUT_TYPE_FEED, source=toml_path),
        processing=ProcessingConfig(
            model=model_map.get(model, ProcessingConfigModel.MODEL_TINY),
            verbose=verbose,
            feed_options=FeedOptions(
                limit=limit or 0,
                order=order_map.get(order, FeedOptionsOrder.ORDER_NEWEST),
                max_retries=3,
                initial_delay=1.0,
                date_threshold=date_threshold or "",
            ),
        ),
        output=OutputConfig(
            format=format_map.get(format, OutputConfigFormat.FORMAT_TEXT),
            destination=output_path or "",
            database=database_config,
        ),
    )
