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
)
from .feed_parser import FeedParser, FeedItem
from .downloader import AudioDownloader
from .blog_processor import BlogProcessor


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
        self.blog_processor = BlogProcessor()

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

        # Initialize feed refresh tracking
        last_feed_refresh = time.time()
        # Get refresh interval - default to 10 minutes if not specified
        refresh_interval_minutes = 10
        if hasattr(request.processing.feed_options, 'refresh_interval_minutes') and request.processing.feed_options.refresh_interval_minutes:
            refresh_interval_minutes = request.processing.feed_options.refresh_interval_minutes
        feed_refresh_interval = refresh_interval_minutes * 60
        # Parse feeds initially
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

        # Prepare items for processing with refresh capability
        def prepare_feed_items():
            """Prepare feed items from current feed_items_dict"""
            prepared = {}
            item_count = 0
            for feed_url, items in feed_items_dict.items():
                # Sort and limit items
                sorted_items = self._sort_and_limit_items(
                    items, request.processing.feed_options
                )
                if sorted_items:
                    prepared[feed_url] = sorted_items
                    item_count += len(sorted_items)
            return prepared, item_count

        feed_items_prepared, total_items = prepare_feed_items()

        # Process items in round-robin or sequential mode
        if request.processing.feed_options.round_robin:
            # Round-robin mode: process newest from each feed before moving to next
            logger.info("Processing feeds in round-robin mode")
            round_index = 0

            while any(items for items in feed_items_prepared.values()):
                # Check if we need to refresh feeds
                current_time = time.time()
                if current_time - last_feed_refresh > feed_refresh_interval:
                    logger.info("Refreshing feeds to check for new items...")
                    try:
                        # Re-parse all feeds
                        new_feed_items_dict = self.feed_parser.parse_all_feeds(
                            request.input.source,
                            max_retries=request.processing.feed_options.max_retries,
                        )

                        # Merge new items into existing structure
                        for feed_url, new_items in new_feed_items_dict.items():
                            if feed_url in feed_items_dict:
                                # Get current item IDs to avoid duplicates
                                existing_ids = {item.item_id for item in feed_items_dict[feed_url]}
                                # Add truly new items
                                for new_item in new_items:
                                    if new_item.item_id not in existing_ids:
                                        feed_items_dict[feed_url].append(new_item)
                                        logger.info(f"Found new item in {feed_url}: {new_item.title}")
                            else:
                                # New feed added to TOML file
                                feed_items_dict[feed_url] = new_items
                                logger.info(f"Found new feed: {feed_url} with {len(new_items)} items")

                        # Re-prepare items with new data
                        feed_items_prepared, new_total = prepare_feed_items()
                        if new_total > total_items:
                            logger.info(f"Found {new_total - total_items} new items after refresh")
                            total_items = new_total
                        last_feed_refresh = current_time
                    except Exception as e:
                        logger.warning(f"Failed to refresh feeds: {e}")

                feeds_processed_this_round = 0

                for feed_url, items in list(feed_items_prepared.items()):
                    if not items:
                        continue

                    # Take the first item from this feed
                    item = items[0]
                    feeds_processed_this_round += 1

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
                        else:
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
                            source=item.audio_url or item.link or item.item_id,
                            error_type=type(e).__name__,
                            message=str(e),
                        )
                        failed += 1

                    # Remove processed item
                    feed_items_prepared[feed_url] = items[1:]
                    if not feed_items_prepared[feed_url]:
                        del feed_items_prepared[feed_url]

                round_index += 1
                if feeds_processed_this_round > 0:
                    logger.info(
                        f"Completed round {round_index}, processed {feeds_processed_this_round} feeds"
                    )
        else:
            # Sequential mode: process all from one feed before moving to next
            items_processed_since_refresh = 0

            for feed_url, sorted_items in feed_items_prepared.items():
                for item in sorted_items:
                    # Check if we need to refresh feeds
                    current_time = time.time()
                    if current_time - last_feed_refresh > feed_refresh_interval:
                        logger.info("Refreshing feeds to check for new items...")
                        try:
                            # Re-parse all feeds
                            new_feed_items_dict = self.feed_parser.parse_all_feeds(
                                request.input.source,
                                max_retries=request.processing.feed_options.max_retries,
                            )

                            # Update feed_items_dict with new items
                            for url, new_items in new_feed_items_dict.items():
                                if url in feed_items_dict:
                                    existing_ids = {item.item_id for item in feed_items_dict[url]}
                                    for new_item in new_items:
                                        if new_item.item_id not in existing_ids:
                                            feed_items_dict[url].append(new_item)
                                            # If this is the current feed, add to sorted_items
                                            if url == feed_url:
                                                sorted_new = self._sort_and_limit_items(
                                                    [new_item], request.processing.feed_options
                                                )
                                                if sorted_new:
                                                    sorted_items.extend(sorted_new)
                                                    logger.info(f"Added new item to current feed: {new_item.title}")
                                else:
                                    feed_items_dict[url] = new_items

                            last_feed_refresh = current_time
                        except Exception as e:
                            logger.warning(f"Failed to refresh feeds: {e}")

                        items_processed_since_refresh = 0
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
                            source=item.audio_url or item.link or item.item_id,
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
        logger.info(f"Processing: {item.title} (type: {item.feed_type})")

        if item.feed_type == "blog":
            # Process blog content
            if not item.content:
                logger.warning(f"Blog item has no content: {item.title}")
                return None
            # Process blog content into transcription format
            result = self.blog_processor.process_blog_content(
                content=item.content,
                filename=item.link or item.item_id,
                title=item.title,
            )

            # Save to database if configured
            if self.database:
                transcription_id = self._save_to_database(result, item, request)
                self._validate_database_save(transcription_id, item, request)

            # Add feed metadata to result for output formatting
            result.feed_metadata = {
                "feed_url": item.feed_url,
                "item_id": item.item_id,
                "title": item.title,
                "link": item.link,
                "published": item.published.isoformat() if item.published else None,
            }

            return result
        else:
            # Process podcast audio
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
                    verbose=(
                        request.processing.verbose and not request.output.destination
                    ),
                )

                # Save to database if configured
                if self.database:
                    transcription_id = self._save_to_database(result, item, request)
                    self._validate_database_save(transcription_id, item, request)

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
    ) -> int:
        """Save transcription to database with feed metadata"""
        db_config = request.output.database
        # Convert enum to string for database
        model_name = self.MODEL_NAME_MAP.get(request.processing.model, "tiny")
        transcription_id = self.database.save_transcription(
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
        return transcription_id

    def _validate_database_save(
        self, transcription_id: int, item: FeedItem, request: TranscriptionRequest
    ):
        """Validate that data was properly saved to the database"""
        if not transcription_id:
            logger.warning(
                f"Failed to save transcription for {item.title} - no ID returned"
            )
            return

        db_config = request.output.database
        metadata_table = (
            db_config.metadata_table if db_config else "transcription_metadata"
        )
        segments_table = (
            db_config.segments_table if db_config else "transcription_segments"
        )

        try:
            # Check metadata was saved
            metadata = self.database.get_transcription_metadata(
                transcription_id, metadata_table
            )
            if not metadata:
                logger.warning(
                    f"WARNING: No metadata found for transcription ID {transcription_id} ({item.title})"
                )
                return

            # Check segments were saved
            segments = self.database.get_segments_for_transcription(
                transcription_id, segments_table
            )
            if not segments:
                logger.warning(
                    f"WARNING: No segments found for transcription ID {transcription_id} ({item.title})"
                )
                return

            # Validate data integrity
            if not metadata.full_text:
                logger.warning(
                    f"WARNING: Empty full_text for transcription ID {transcription_id} ({item.title})"
                )

            if metadata.feed_item_id != item.item_id:
                logger.warning(
                    f"WARNING: Mismatched item ID for transcription ID {transcription_id} "
                    f"(expected: {item.item_id}, got: {metadata.feed_item_id})"
                )

            # Log success for blog items (since they're new)
            if item.feed_type == "blog":
                logger.info(
                    f"✓ Blog post saved: ID={transcription_id}, title='{item.title}', "
                    f"segments={len(segments)}, text_length={len(metadata.full_text)}"
                )
            else:
                logger.debug(
                    f"✓ Podcast saved: ID={transcription_id}, title='{item.title}', "
                    f"segments={len(segments)}"
                )

        except Exception as e:
            logger.warning(
                f"WARNING: Could not validate database save for {item.title}: {e}"
            )

    def _sort_and_limit_items(
        self, items: List[FeedItem], feed_options
    ) -> List[FeedItem]:
        """Sort and limit feed items based on options"""
        # Filter by date threshold if specified
        if feed_options.date_threshold:
            try:
                threshold_date = datetime.fromisoformat(
                    feed_options.date_threshold.replace("Z", "+00:00")
                )
                # Filter items published after the threshold
                items = [
                    item
                    for item in items
                    if item.published and item.published > threshold_date
                ]
                logger.info(
                    f"Filtered to {len(items)} items after {feed_options.date_threshold}"
                )
            except ValueError as e:
                logger.warning(f"Invalid date threshold format: {e}")

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
