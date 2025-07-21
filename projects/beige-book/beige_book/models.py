"""
Request and response models for the transcription service.

These models provide a clean boundary between different interfaces (CLI, REST API, etc.)
and the core transcription functionality.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
import json


# Input Configuration
@dataclass
class InputConfig:
    """Configuration for input sources"""
    type: Literal["file", "feed"]  # Input type
    source: str  # File path or TOML path

    def validate(self):
        """Validate the input configuration"""
        if not self.source:
            raise ValueError("Input source cannot be empty")
        if self.type not in ["file", "feed"]:
            raise ValueError(f"Invalid input type: {self.type}")


# Processing Configuration
@dataclass
class FeedOptions:
    """Options specific to feed processing"""
    limit: Optional[int] = None
    order: Literal["newest", "oldest"] = "newest"
    max_retries: int = 3
    initial_delay: float = 1.0

    def validate(self):
        """Validate feed options"""
        if self.limit is not None and self.limit < 1:
            raise ValueError("Feed limit must be positive")
        if self.order not in ["newest", "oldest"]:
            raise ValueError(f"Invalid order: {self.order}")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
        if self.initial_delay < 0:
            raise ValueError("Initial delay cannot be negative")


@dataclass
class ProcessingConfig:
    """Configuration for processing options"""
    model: Literal["tiny", "base", "small", "medium", "large"] = "tiny"
    verbose: bool = False
    feed_options: Optional[FeedOptions] = None

    def validate(self):
        """Validate processing configuration"""
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if self.model not in valid_models:
            raise ValueError(f"Invalid model: {self.model}. Must be one of {valid_models}")
        if self.feed_options:
            self.feed_options.validate()


# Output Configuration
@dataclass
class DatabaseConfig:
    """Database-specific configuration"""
    db_path: str
    metadata_table: str = "transcription_metadata"
    segments_table: str = "transcription_segments"

    def validate(self):
        """Validate database configuration"""
        if not self.db_path:
            raise ValueError("Database path cannot be empty")
        if not self.metadata_table:
            raise ValueError("Metadata table name cannot be empty")
        if not self.segments_table:
            raise ValueError("Segments table name cannot be empty")


@dataclass
class OutputConfig:
    """Configuration for output handling"""
    format: Literal["text", "json", "table", "csv", "toml", "sqlite"] = "text"
    destination: Optional[str] = None  # File path or None for stdout
    database: Optional[DatabaseConfig] = None

    def validate(self):
        """Validate output configuration"""
        valid_formats = ["text", "json", "table", "csv", "toml", "sqlite"]
        if self.format not in valid_formats:
            raise ValueError(f"Invalid format: {self.format}. Must be one of {valid_formats}")

        # SQLite format requires database config
        if self.format == "sqlite" and not self.database:
            raise ValueError("SQLite format requires database configuration")

        if self.database:
            self.database.validate()


# Main Request Object
@dataclass
class TranscriptionRequest:
    """Main request object for all transcription operations"""
    input: InputConfig
    processing: ProcessingConfig
    output: OutputConfig

    def validate(self):
        """Validate the entire request"""
        self.input.validate()
        self.processing.validate()
        self.output.validate()

        # Cross-field validation
        if self.input.type == "file" and self.processing.feed_options:
            raise ValueError("Feed options provided for file input")
        if self.input.type == "feed" and not self.processing.feed_options:
            # Create default feed options if not provided
            self.processing.feed_options = FeedOptions()

    def to_json(self) -> str:
        """Convert request to JSON string"""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'TranscriptionRequest':
        """Create request from JSON string"""
        data = json.loads(json_str)

        # Reconstruct nested objects
        input_config = InputConfig(**data['input'])

        feed_options = None
        if data['processing'].get('feed_options'):
            feed_options = FeedOptions(**data['processing']['feed_options'])

        processing_config = ProcessingConfig(
            model=data['processing']['model'],
            verbose=data['processing']['verbose'],
            feed_options=feed_options
        )

        database_config = None
        if data['output'].get('database'):
            database_config = DatabaseConfig(**data['output']['database'])

        output_config = OutputConfig(
            format=data['output']['format'],
            destination=data['output'].get('destination'),
            database=database_config
        )

        return cls(input=input_config, processing=processing_config, output=output_config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary"""
        return asdict(self)


# Response Objects
@dataclass
class ProcessingError:
    """Error information for failed items"""
    source: str  # File path or feed URL
    error_type: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO timestamp"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ProcessingSummary:
    """Summary statistics for batch operations"""
    total_items: int
    processed: int
    skipped: int
    failed: int
    elapsed_time: float

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_items == 0:
            return 0.0
        return (self.processed / self.total_items) * 100


@dataclass
class TranscriptionResponse:
    """Response object for transcription operations"""
    success: bool
    results: List[Any] = field(default_factory=list)  # List of TranscriptionResult objects
    errors: List[ProcessingError] = field(default_factory=list)
    summary: Optional[ProcessingSummary] = None

    def add_error(self, source: str, error_type: str, message: str):
        """Add an error to the response"""
        self.errors.append(ProcessingError(
            source=source,
            error_type=error_type,
            message=message
        ))

    def to_json(self) -> str:
        """Convert response to JSON string"""
        # Convert results to dicts if they have to_dict method
        data = {
            'success': self.success,
            'results': [r.to_dict() if hasattr(r, 'to_dict') else r for r in self.results],
            'errors': [e.to_dict() for e in self.errors],
            'summary': asdict(self.summary) if self.summary else None
        }
        return json.dumps(data, indent=2)


# Convenience functions for creating requests
def create_file_request(
    filename: str,
    model: str = "tiny",
    format: str = "text",
    output_path: Optional[str] = None,
    verbose: bool = False
) -> TranscriptionRequest:
    """Create a request for transcribing a single file"""
    return TranscriptionRequest(
        input=InputConfig(type="file", source=filename),
        processing=ProcessingConfig(model=model, verbose=verbose),
        output=OutputConfig(format=format, destination=output_path)
    )


def create_feed_request(
    toml_path: str,
    model: str = "tiny",
    format: str = "text",
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    order: str = "newest",
    verbose: bool = False,
    db_path: Optional[str] = None
) -> TranscriptionRequest:
    """Create a request for processing RSS feeds"""
    database_config = None
    if db_path or format == "sqlite":
        database_config = DatabaseConfig(db_path=db_path or "beige_book_feeds.db")

    return TranscriptionRequest(
        input=InputConfig(type="feed", source=toml_path),
        processing=ProcessingConfig(
            model=model,
            verbose=verbose,
            feed_options=FeedOptions(limit=limit, order=order)
        ),
        output=OutputConfig(
            format=format,
            destination=output_path,
            database=database_config
        )
    )
