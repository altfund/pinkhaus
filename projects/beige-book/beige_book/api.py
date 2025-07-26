"""
FastAPI application for the Beige Book transcription service.

Provides a REST API interface to the same functionality as the CLI.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from datetime import datetime
import logging

from .models import (
    TranscriptionRequest,
    InputConfig,
    ProcessingConfig,
    OutputConfig,
    FeedOptions,
    DatabaseConfig,
)
from .service import TranscriptionService, OutputFormatter


# Configure logging
logger = logging.getLogger(__name__)


# Pydantic models for API request/response
class FeedOptionsAPI(BaseModel):
    """Options specific to feed processing"""

    limit: Optional[int] = Field(
        None, description="Maximum number of feed items to process per feed"
    )
    order: Literal["newest", "oldest"] = Field(
        "newest", description="Process feed items from newest or oldest first"
    )
    max_retries: int = Field(
        3, description="Maximum number of retries for failed downloads"
    )
    initial_delay: float = Field(
        1.0, description="Initial delay in seconds between retries"
    )

    @validator("limit")
    def validate_limit(cls, v):
        if v is not None and v < 1:
            raise ValueError("Feed limit must be positive")
        return v

    @validator("max_retries")
    def validate_max_retries(cls, v):
        if v < 0:
            raise ValueError("Max retries cannot be negative")
        return v

    @validator("initial_delay")
    def validate_initial_delay(cls, v):
        if v < 0:
            raise ValueError("Initial delay cannot be negative")
        return v


class DatabaseConfigAPI(BaseModel):
    """Database-specific configuration"""

    db_path: str = Field(..., description="Path to SQLite database file")
    metadata_table: str = Field(
        "transcription_metadata", description="Name of the metadata table"
    )
    segments_table: str = Field(
        "transcription_segments", description="Name of the segments table"
    )

    @validator("db_path")
    def validate_db_path(cls, v):
        if not v:
            raise ValueError("Database path cannot be empty")
        return v


class InputConfigAPI(BaseModel):
    """Configuration for input sources"""

    type: Literal["file", "feed"] = Field(
        ..., description="Input type: single file or RSS feed"
    )
    source: str = Field(
        ..., description="File path for audio file or TOML path for feeds"
    )

    @validator("source")
    def validate_source(cls, v):
        if not v:
            raise ValueError("Input source cannot be empty")
        return v


class ProcessingConfigAPI(BaseModel):
    """Configuration for processing options"""

    model: Literal["tiny", "base", "small", "medium", "large"] = Field(
        "tiny", description="Whisper model to use for transcription"
    )
    verbose: bool = Field(False, description="Enable verbose logging")
    feed_options: Optional[FeedOptionsAPI] = Field(
        None,
        description="Options for feed processing (required when input type is 'feed')",
    )


class OutputConfigAPI(BaseModel):
    """Configuration for output handling"""

    format: Literal["text", "json", "table", "csv", "toml", "sqlite"] = Field(
        "text", description="Output format for transcription results"
    )
    destination: Optional[str] = Field(
        None, description="File path for output (None for response body)"
    )
    database: Optional[DatabaseConfigAPI] = Field(
        None, description="Database configuration (required for sqlite format)"
    )

    @validator("database", always=True)
    def validate_database(cls, v, values):
        if values.get("format") == "sqlite" and not v:
            raise ValueError("SQLite format requires database configuration")
        return v


class TranscriptionRequestAPI(BaseModel):
    """Main request object for all transcription operations"""

    input: InputConfigAPI = Field(..., description="Input configuration")
    processing: ProcessingConfigAPI = Field(..., description="Processing configuration")
    output: OutputConfigAPI = Field(..., description="Output configuration")

    class Config:
        schema_extra = {
            "example": {
                "input": {"type": "file", "source": "/path/to/audio.mp3"},
                "processing": {"model": "tiny", "verbose": False},
                "output": {"format": "json", "destination": None},
            }
        }

    @validator("processing")
    def validate_processing(cls, v, values):
        input_config = values.get("input")
        if input_config:
            if input_config.type == "file" and v.feed_options:
                raise ValueError("Feed options provided for file input")
            if input_config.type == "feed" and not v.feed_options:
                # Set default feed options
                v.feed_options = FeedOptionsAPI()
        return v

    def to_internal_request(self) -> TranscriptionRequest:
        """Convert API request to internal request model"""
        # Convert feed options if present
        feed_options = None
        if self.processing.feed_options:
            feed_options = FeedOptions(
                limit=self.processing.feed_options.limit,
                order=self.processing.feed_options.order,
                max_retries=self.processing.feed_options.max_retries,
                initial_delay=self.processing.feed_options.initial_delay,
            )

        # Convert database config if present
        database_config = None
        if self.output.database:
            database_config = DatabaseConfig(
                db_path=self.output.database.db_path,
                metadata_table=self.output.database.metadata_table,
                segments_table=self.output.database.segments_table,
            )

        return TranscriptionRequest(
            input=InputConfig(type=self.input.type, source=self.input.source),
            processing=ProcessingConfig(
                model=self.processing.model,
                verbose=self.processing.verbose,
                feed_options=feed_options,
            ),
            output=OutputConfig(
                format=self.output.format,
                destination=self.output.destination,
                database=database_config,
            ),
        )


# Create FastAPI app
app = FastAPI(
    title="Beige Book Transcription API",
    description="API for transcribing audio files and RSS feed audio content using Whisper",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.post(
    "/transcribe",
    summary="Transcribe audio content",
    description="Process audio files or RSS feeds and return transcriptions in various formats",
    response_description="Transcription results in the requested format",
    responses={
        200: {
            "description": "Successful transcription",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "results": [{"text": "Transcribed content...", "segments": []}],
                        "errors": [],
                        "summary": {
                            "total_items": 1,
                            "processed": 1,
                            "skipped": 0,
                            "failed": 0,
                            "elapsed_time": 5.2,
                        },
                    }
                },
                "text/plain": {"example": "Transcribed text content..."},
                "text/csv": {"example": 'start,end,text\n0.0,5.2,"Hello world"'},
            },
        },
        400: {"description": "Invalid request"},
        500: {"description": "Internal server error"},
    },
    tags=["transcription"],
)
async def transcribe(request: TranscriptionRequestAPI):
    """
    Transcribe audio content from a file or RSS feeds.

    This endpoint accepts the same request structure as the CLI and returns results
    in the format specified in the output configuration.

    ## Input Types
    - **file**: Transcribe a single audio file
    - **feed**: Process RSS feeds from a TOML configuration file

    ## Output Formats
    - **text**: Plain text transcription
    - **json**: Structured JSON with segments and metadata
    - **table**: Human-readable table format
    - **csv**: CSV format with timestamps
    - **toml**: TOML format
    - **sqlite**: Save to SQLite database (returns success status)

    ## Examples

    ### Transcribe a single file to JSON:
    ```json
    {
        "input": {"type": "file", "source": "/path/to/audio.mp3"},
        "processing": {"model": "tiny", "verbose": false},
        "output": {"format": "json"}
    }
    ```

    ### Process RSS feeds and save to database:
    ```json
    {
        "input": {"type": "feed", "source": "/path/to/feeds.toml"},
        "processing": {
            "model": "base",
            "verbose": true,
            "feed_options": {"limit": 10, "order": "newest"}
        },
        "output": {
            "format": "sqlite",
            "database": {
                "db_path": "/path/to/database.db",
                "metadata_table": "transcriptions",
                "segments_table": "segments"
            }
        }
    }
    ```
    """
    try:
        # Convert API request to internal request
        internal_request = request.to_internal_request()

        # Validate the request
        internal_request.validate()

        # Create service and process request
        service = TranscriptionService()
        response = service.process(internal_request)

        # Handle different output formats
        if request.output.format == "json":
            return JSONResponse(
                content=response.to_json(), media_type="application/json"
            )

        elif request.output.format == "sqlite":
            # For SQLite, just return success status
            if response.success:
                return JSONResponse(
                    content={
                        "success": True,
                        "message": f"Saved to database: {request.output.database.db_path}",
                        "summary": response.summary.__dict__
                        if response.summary
                        else None,
                    }
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "success": False,
                        "errors": [e.to_dict() for e in response.errors],
                    },
                )

        else:
            # Format results for other output types
            if response.success and response.results:
                formatted = OutputFormatter.format_results(
                    response.results,
                    request.output.format,
                    include_feed_metadata=(request.input.type == "feed"),
                )

                # Determine content type based on format
                content_types = {
                    "text": "text/plain",
                    "table": "text/plain",
                    "csv": "text/csv",
                    "toml": "application/toml",
                }
                content_type = content_types.get(request.output.format, "text/plain")

                return Response(content=formatted, media_type=content_type)
            else:
                # Handle errors
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "success": False,
                        "errors": [e.to_dict() for e in response.errors],
                    },
                )

    except ValueError as e:
        # Validation errors
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during transcription: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


@app.get("/", summary="API Information", tags=["info"])
async def root():
    """Get basic API information"""
    return {
        "name": "Beige Book Transcription API",
        "version": "1.0.0",
        "description": "API for transcribing audio files and RSS feed audio content",
        "endpoints": {
            "transcribe": "/transcribe",
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
    }


@app.get("/health", summary="Health Check", tags=["info"])
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Add example requests for documentation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title="Beige Book Transcription API",
        version="1.0.0",
        description="API for transcribing audio files and RSS feed audio content using Whisper",
        routes=app.routes,
    )

    # Add additional examples
    if "paths" in openapi_schema:
        if "/transcribe" in openapi_schema["paths"]:
            if "post" in openapi_schema["paths"]["/transcribe"]:
                openapi_schema["paths"]["/transcribe"]["post"]["requestBody"][
                    "content"
                ]["application/json"]["examples"] = {
                    "single_file": {
                        "summary": "Transcribe single audio file",
                        "value": {
                            "input": {"type": "file", "source": "/path/to/audio.mp3"},
                            "processing": {"model": "tiny", "verbose": False},
                            "output": {"format": "json"},
                        },
                    },
                    "rss_feeds": {
                        "summary": "Process RSS feeds",
                        "value": {
                            "input": {"type": "feed", "source": "/path/to/feeds.toml"},
                            "processing": {
                                "model": "base",
                                "verbose": True,
                                "feed_options": {"limit": 10, "order": "newest"},
                            },
                            "output": {"format": "text"},
                        },
                    },
                    "save_to_database": {
                        "summary": "Save transcriptions to SQLite",
                        "value": {
                            "input": {"type": "file", "source": "/path/to/audio.wav"},
                            "processing": {"model": "small", "verbose": False},
                            "output": {
                                "format": "sqlite",
                                "database": {
                                    "db_path": "/path/to/transcriptions.db",
                                    "metadata_table": "metadata",
                                    "segments_table": "segments",
                                },
                            },
                        },
                    },
                }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
