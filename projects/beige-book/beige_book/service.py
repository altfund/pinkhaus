"""
Transcription service that processes requests and returns responses.

This service provides a clean interface for all transcription operations,
abstracting away the details of file handling, feed processing, and output formatting.
"""

import time
import logging
import os
from typing import List, Optional
from datetime import datetime

from .transcriber import AudioTranscriber, TranscriptionResult
from pinkhaus_models import TranscriptionDatabase
from .models import (
    TranscriptionRequest,
    TranscriptionResponse,
    ProcessingSummary,
)
from .proto_models import (
    InputConfigInputType,
    ProcessingConfigModel,
    OutputConfigFormat,
    FeedOptionsOrder,
    TranscriptionResult,
)
from .feed_parser import FeedParser, FeedItem
from .downloader import AudioDownloader


logger = logging.getLogger(__name__)


class TranscriptionService:
    """Main service for handling transcription requests"""

    # Model enum to string mapping
    MODEL_NAME_MAP = {
        ProcessingConfigModel.MODEL_TINY: "tiny",
        ProcessingConfigModel.MODEL_BASE: "base",
        ProcessingConfigModel.MODEL_SMALL: "small",
        ProcessingConfigModel.MODEL_MEDIUM: "medium",
        ProcessingConfigModel.MODEL_LARGE: "large",
    }

    def __init__(self):
        """Initialize the transcription service"""
        self.transcriber = None
        self.database = None
        self.feed_parser = FeedParser()
        self.downloader = AudioDownloader()

    def process(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """
        Process a transcription request.

        Args:
            request: The transcription request to process

        Returns:
            TranscriptionResponse with results or errors
        """
        try:
            # Validate request
            request.validate()

            # Initialize transcriber with requested model
            # Convert enum to string for AudioTranscriber
            model_name = self.MODEL_NAME_MAP.get(request.processing.model, "tiny")
            self.transcriber = AudioTranscriber(model_name=model_name)

            # Process based on input type
            if request.input.type == InputConfigInputType.INPUT_TYPE_FILE:
                return self._process_file(request)
            else:
                return self._process_feeds(request)

        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            response = TranscriptionResponse(success=False)
            response.add_error(
                source=request.input.source, error_type=type(e).__name__, message=str(e)
            )
            return response

    def _process_file(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """Process a single audio file"""
        start_time = time.time()
        response = TranscriptionResponse(success=True)

        try:
            # Expand path (handle ~ and relative paths)
            file_path = os.path.expanduser(request.input.source)
            file_path = os.path.abspath(file_path)

            # Transcribe the file
            result = self.transcriber.transcribe_file(
                file_path, verbose=request.processing.verbose
            )
            response.results.append(result)

            # Handle output
            self._handle_output(request, [result], response)

            # Add summary
            response.summary = ProcessingSummary(
                total_items=1,
                processed=1,
                skipped=0,
                failed=0,
                elapsed_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"File processing failed: {e}")
            response.success = False
            response.add_error(
                source=request.input.source, error_type=type(e).__name__, message=str(e)
            )
            response.summary = ProcessingSummary(
                total_items=1,
                processed=0,
                skipped=0,
                failed=1,
                elapsed_time=time.time() - start_time,
            )

        return response

    def _process_feeds(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """Process RSS feeds from TOML file"""
        start_time = time.time()
        response = TranscriptionResponse(success=True)

        # Setup database if needed for resumability
        db = None
        is_resumable = self._is_resumable(request)
        if is_resumable:
            db_path = (
                request.output.database.db_path
                if request.output.database
                else "beige_book_feeds.db"
            )
            db = TranscriptionDatabase(db_path)
            db.create_tables(
                request.output.database.metadata_table
                if request.output.database
                else "transcription_metadata",
                request.output.database.segments_table
                if request.output.database
                else "transcription_segments",
            )
            self.database = db

        # Parse feeds
        try:
            feed_items_dict = self.feed_parser.parse_all_feeds(
                request.input.source,
                max_retries=request.processing.feed_options.max_retries,
            )
        except Exception as e:
            response.success = False
            response.add_error(
                source=request.input.source,
                error_type=type(e).__name__,
                message=f"Failed to parse feeds: {str(e)}",
            )
            return response

        # Process items
        total_items = 0
        processed = 0
        skipped = 0
        failed = 0

        for feed_url, items in feed_items_dict.items():
            # Sort and limit items
            sorted_items = self._sort_and_limit_items(
                items, request.processing.feed_options
            )
            total_items += len(sorted_items)

            for item in sorted_items:
                try:
                    # Check if already processed
                    if (
                        is_resumable
                        and db
                        and db.check_feed_item_exists(
                            item.feed_url,
                            item.item_id,
                            request.output.database.metadata_table
                            if request.output.database
                            else "transcription_metadata",
                        )
                    ):
                        logger.info(f"Skipping already processed: {item.title}")
                        skipped += 1
                        continue

                    # Process item
                    result = self._process_feed_item(item, request)
                    if result:
                        response.results.append(result)
                        processed += 1
                    else:
                        failed += 1

                except Exception as e:
                    logger.error(f"Failed to process {item.title}: {e}")
                    response.add_error(
                        source=item.audio_url,
                        error_type=type(e).__name__,
                        message=str(e),
                    )
                    failed += 1

        # Set summary
        response.summary = ProcessingSummary(
            total_items=total_items,
            processed=processed,
            skipped=skipped,
            failed=failed,
            elapsed_time=time.time() - start_time,
        )

        response.success = failed == 0
        return response

    def _process_feed_item(
        self, item: FeedItem, request: TranscriptionRequest
    ) -> Optional[TranscriptionResult]:
        """Process a single feed item"""
        logger.info(f"Processing: {item.title}")

        # Download audio
        temp_path, file_hash = self.downloader.download_with_retry(
            item.audio_url,
            max_retries=request.processing.feed_options.max_retries,
            initial_delay=request.processing.feed_options.initial_delay,
        )

        try:
            # Transcribe
            result = self.transcriber.transcribe_file(
                temp_path,
                verbose=(request.processing.verbose and not request.output.destination),
            )

            # Save to database if configured
            if self.database:
                self._save_to_database(result, item, request)

            # Add feed metadata to result for output formatting
            result.feed_metadata = {
                "feed_url": item.feed_url,
                "item_id": item.item_id,
                "title": item.title,
                "audio_url": item.audio_url,
                "published": item.published.isoformat() if item.published else None,
            }

            return result

        finally:
            # Clean up temp file
            self.downloader.cleanup_temp_file(temp_path)

    def _save_to_database(
        self, result: TranscriptionResult, item: FeedItem, request: TranscriptionRequest
    ):
        """Save transcription to database with feed metadata"""
        db_config = request.output.database
        # Convert enum to string for database
        model_name = self.MODEL_NAME_MAP.get(request.processing.model, "tiny")
        self.database.save_transcription(
            result,
            model_name=model_name,
            metadata_table=db_config.metadata_table
            if db_config
            else "transcription_metadata",
            segments_table=db_config.segments_table
            if db_config
            else "transcription_segments",
            feed_url=item.feed_url,
            feed_item_id=item.item_id,
            feed_item_title=item.title,
            feed_item_published=item.published.isoformat() if item.published else None,
        )

    def _sort_and_limit_items(
        self, items: List[FeedItem], feed_options
    ) -> List[FeedItem]:
        """Sort and limit feed items based on options"""
        # Sort by publication date
        if feed_options.order == FeedOptionsOrder.ORDER_NEWEST:
            sorted_items = sorted(
                items, key=lambda x: x.published or datetime.min, reverse=True
            )
        else:
            sorted_items = sorted(items, key=lambda x: x.published or datetime.min)

        # Apply limit
        if feed_options.limit:
            sorted_items = sorted_items[: feed_options.limit]

        return sorted_items

    def _is_resumable(self, request: TranscriptionRequest) -> bool:
        """Check if the request supports resumability"""
        resumable_formats = {
            OutputConfigFormat.FORMAT_TEXT,
            OutputConfigFormat.FORMAT_JSON,
            OutputConfigFormat.FORMAT_TABLE,
            OutputConfigFormat.FORMAT_CSV,
            OutputConfigFormat.FORMAT_TOML,
            OutputConfigFormat.FORMAT_SQLITE,
        }
        return request.output.format in resumable_formats and (
            request.output.database or request.output.destination
        )

    def _handle_output(
        self,
        request: TranscriptionRequest,
        results: List[TranscriptionResult],
        response: TranscriptionResponse,
    ):
        """Handle output formatting and writing"""
        if request.output.format == OutputConfigFormat.FORMAT_SQLITE:
            # For SQLite, results are already saved during processing
            if not self.database:
                # Single file to database
                db_config = request.output.database
                db = TranscriptionDatabase(db_config.db_path)
                db.create_tables(db_config.metadata_table, db_config.segments_table)

                for result in results:
                    db.save_transcription(
                        result,
                        model_name=self.MODEL_NAME_MAP.get(
                            request.processing.model, "tiny"
                        ),
                        metadata_table=db_config.metadata_table,
                        segments_table=db_config.segments_table,
                    )
        else:
            # Format output for other formats
            # This would be handled by the interface layer (CLI, REST, etc.)
            # The service just returns the results
            pass


class OutputFormatter:
    """Helper class for formatting output based on request"""

    @staticmethod
    def format_results(
        results: List[TranscriptionResult],
        format: str,
        include_feed_metadata: bool = False,
    ) -> str:
        """
        Format multiple results into a single output string.

        Args:
            results: List of transcription results
            format: Output format
            include_feed_metadata: Whether to include feed metadata in output

        Returns:
            Formatted output string
        """
        if format == "text":
            outputs = []
            for result in results:
                if include_feed_metadata and hasattr(result, "feed_metadata"):
                    header = OutputFormatter._format_feed_header(result.feed_metadata)
                    outputs.append(header + result.full_text)
                else:
                    outputs.append(result.full_text)
            return "\n\n".join(outputs)

        elif format == "json":
            import json

            all_results = []
            for result in results:
                data = result.to_dict()
                if include_feed_metadata and hasattr(result, "feed_metadata"):
                    data["feed_metadata"] = result.feed_metadata
                all_results.append(data)
            return json.dumps(all_results, indent=2, ensure_ascii=False)

        else:
            # For other formats, concatenate individual formatted results
            outputs = []
            for result in results:
                if include_feed_metadata and hasattr(result, "feed_metadata"):
                    header = OutputFormatter._format_feed_comment(
                        result.feed_metadata, format
                    )
                    outputs.append(header + result.format(format))
                else:
                    outputs.append(result.format(format))
            return "\n\n".join(outputs)

    @staticmethod
    def _format_feed_header(feed_metadata: dict) -> str:
        """Format feed metadata as text header"""
        header = f"Feed: {feed_metadata['feed_url']}\n"
        header += f"Title: {feed_metadata['title']}\n"
        if feed_metadata.get("published"):
            header += f"Published: {feed_metadata['published']}\n"
        header += f"Audio URL: {feed_metadata['audio_url']}\n"
        header += "-" * 80 + "\n\n"
        return header

    @staticmethod
    def _format_feed_comment(feed_metadata: dict, format: str) -> str:
        """Format feed metadata as comment for various formats"""
        comment_char = "#" if format in ["csv", "table"] else "//"
        header = f"{comment_char} Feed: {feed_metadata['feed_url']}\n"
        header += f"{comment_char} Title: {feed_metadata['title']}\n"
        if feed_metadata.get("published"):
            header += f"{comment_char} Published: {feed_metadata['published']}\n"
        return header
