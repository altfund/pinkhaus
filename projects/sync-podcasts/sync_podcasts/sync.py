"""Core synchronization logic for podcast pipeline."""

import logging
import os
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import fcntl

from beige_book import TranscriptionService
from beige_book.models import (
    TranscriptionRequest,
    InputConfig,
    ProcessingConfig,
    OutputConfig,
    FeedOptions,
    DatabaseConfig,
)
from beige_book.feed_parser import FeedParser
from grant import RAGPipeline, OllamaClient
from pinkhaus_models import TranscriptionDatabase

from .validate_feed import validate_feeds_toml

logger = logging.getLogger(__name__)


@dataclass
class SyncConfig:
    """Configuration for podcast synchronization."""

    feeds_path: str = "./feeds.toml"
    db_path: str = "./podcasts.db"
    vector_store_path: str = "./grant_chroma_db"
    model: str = "tiny"
    round_robin: bool = True
    daemon: bool = False
    date_threshold: Optional[str] = None
    days_back: Optional[int] = None
    ollama_base_url: str = "http://localhost:11434"
    verbose: bool = False
    max_failures: int = 3  # Maximum retry attempts before marking as permanently failed
    stale_minutes: int = 30  # Minutes before considering a processing task stale
    skip_validation: bool = False  # Skip feed validation on startup (for testing)


class ProcessLock:
    """File-based lock to prevent multiple instances working on the same feeds file."""

    def __init__(self, feeds_path: str):
        # Create a lock file based on the feeds file path
        import hashlib

        feeds_abs_path = os.path.abspath(feeds_path)
        path_hash = hashlib.md5(feeds_abs_path.encode()).hexdigest()[:8]
        self.lock_file = f"/tmp/sync_podcasts_{path_hash}.lock"
        self.lock_fd = None
        self.feeds_path = feeds_abs_path

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns True if successful."""
        try:
            self.lock_fd = open(self.lock_file, "w")
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Write our PID, hostname, and feeds file
            self.lock_fd.write(f"{os.getpid()}@{socket.gethostname()}\n")
            self.lock_fd.write(f"Feeds: {self.feeds_path}\n")
            self.lock_fd.flush()
            return True
        except IOError:
            if self.lock_fd:
                self.lock_fd.close()
                self.lock_fd = None
            return False

    def release(self):
        """Release the lock."""
        if self.lock_fd:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            self.lock_fd.close()
            self.lock_fd = None
            try:
                os.unlink(self.lock_file)
            except OSError:
                pass


class PodcastSyncer:
    """Orchestrates podcast transcription and indexing."""

    def __init__(self, config: SyncConfig = SyncConfig()):
        self.config = config
        self.transcription_service = TranscriptionService()
        self.ollama_client = OllamaClient(base_url=config.ollama_base_url)
        self.db = TranscriptionDatabase(config.db_path)
        self.db.create_tables()
        self.process_lock = ProcessLock(config.feeds_path)
        self.hostname = socket.gethostname()
        self.pid = os.getpid()

        # Validate feeds before starting (unless skipped for testing)
        if not config.skip_validation:
            self._validate_feeds()

        # Determine date threshold
        if config.days_back:
            self.date_threshold = (
                datetime.now() - timedelta(days=config.days_back)
            ).isoformat()
        elif config.date_threshold:
            self.date_threshold = config.date_threshold
        else:
            # Default to last 30 days
            self.date_threshold = (datetime.now() - timedelta(days=30)).isoformat()

        logger.info(f"Syncing podcasts published after: {self.date_threshold}")

        # Show startup report
        self._show_startup_report()

    def _validate_feeds(self):
        """Validate all feeds in the TOML file before processing."""
        logger.info(f"Validating feeds from: {self.config.feeds_path}")

        # Validate all feeds
        results = validate_feeds_toml(self.config.feeds_path, move_invalid=False)

        # Check for TOML parsing error
        if "_error" in results:
            logger.error(f"Error parsing feeds file: {results['_error'].error_message}")
            raise SystemExit(1)

        # Check for invalid feeds
        invalid_feeds = [
            (url, result) for url, result in results.items() if not result.is_valid
        ]

        if invalid_feeds:
            logger.error("=" * 70)
            logger.error("INVALID FEEDS DETECTED")
            logger.error("=" * 70)

            for feed_url, result in invalid_feeds:
                logger.error(f"✗ {feed_url}")
                logger.error(f"  Error: {result.error_message}")
                if result.feed_title:
                    logger.error(f"  Title: {result.feed_title}")

            logger.error("=" * 70)
            logger.error(
                f"Found {len(invalid_feeds)} invalid feed(s) in {self.config.feeds_path}"
            )
            logger.error(
                "Please fix or remove these feeds before running sync-podcasts."
            )
            logger.error(
                "You can use the 'validate-feed' tool to check and fix feed issues:"
            )
            logger.error(f"  validate-feed {self.config.feeds_path} --move-invalid")
            logger.error("=" * 70)

            raise SystemExit(1)

        # All feeds valid
        total_feeds = len(results)
        logger.info(f"All {total_feeds} feed(s) validated successfully")

    def _show_startup_report(self):
        """Show a report of unprocessed and failed items on startup."""
        logger.info("=" * 70)
        logger.info("STARTUP REPORT")
        logger.info("=" * 70)

        # Show failed items summary
        failed_summary = self.db.get_failed_items_summary()
        if failed_summary:
            logger.warning("Failed items found in 'failed_items' table:")
            for feed_info in failed_summary:
                logger.warning(
                    f"  - {feed_info['feed_url']}: "
                    f"{feed_info['failed_count']} failed items "
                    f"(max failures: {feed_info['max_failures']})"
                )
        else:
            logger.info("No failed items found")

        # Check for unprocessed items
        try:
            feed_parser = FeedParser()
            feed_urls = feed_parser.parse_toml_feeds(self.config.feeds_path)

            total_unprocessed = 0
            feed_stats = []

            for feed_url in feed_urls:
                try:
                    # Parse the feed
                    items = feed_parser.parse_feed(feed_url)

                    # Apply date threshold
                    if self.date_threshold:
                        threshold_date = datetime.fromisoformat(
                            self.date_threshold.replace("Z", "+00:00")
                        )
                        items = [
                            item
                            for item in items
                            if item.published and item.published > threshold_date
                        ]

                    # Count how many are not in the database
                    unprocessed_count = 0
                    for item in items:
                        if not self.db.check_feed_item_exists(
                            feed_url, item.item_id, "transcription_metadata"
                        ):
                            # Also check if it's not permanently failed
                            failed_info = self.db.get_failed_item(
                                feed_url, item.item_id
                            )
                            if (
                                not failed_info
                                or failed_info["failure_count"]
                                < self.config.max_failures
                            ):
                                unprocessed_count += 1

                    if unprocessed_count > 0:
                        feed_stats.append((feed_url, unprocessed_count, len(items)))
                        total_unprocessed += unprocessed_count

                except Exception as e:
                    logger.error(f"Error checking feed {feed_url}: {e}")

            if feed_stats:
                logger.warning(
                    "Unprocessed items found (will be stored in 'transcription_metadata' table):"
                )
                for feed_url, unprocessed, total in feed_stats:
                    logger.warning(
                        f"  - {feed_url}: {unprocessed}/{total} items to process"
                    )
                logger.warning(f"Total unprocessed items: {total_unprocessed}")
            else:
                logger.info("All items up to date!")

        except Exception as e:
            logger.error(f"Error generating startup report: {e}")

        logger.info("=" * 70)

    def _check_and_clean_stale_processing(self):
        """Check for stale processing items and clean them up."""
        stale_items = self.db.get_stale_processing_items(self.config.stale_minutes)
        for item in stale_items:
            logger.warning(
                f"Found stale processing item: {item['feed_item_title']} "
                f"(started at {item['started_at']} by PID {item['pid']}@{item['hostname']})"
            )
            # Clear the stale state so it can be retried
            self.db.clear_processing_state(item["feed_url"], item["feed_item_id"])

    def _should_skip_item(self, feed_url: str, feed_item_id: str) -> tuple[bool, str]:
        """Check if an item should be skipped. Returns (should_skip, reason)."""
        # Check if already processed
        if self.db.check_feed_item_exists(feed_url, feed_item_id):
            return True, "already processed"

        # Check if permanently failed
        failed_info = self.db.get_failed_item(feed_url, feed_item_id)
        if failed_info and failed_info["failure_count"] >= self.config.max_failures:
            return (
                True,
                f"permanently failed after {failed_info['failure_count']} attempts",
            )

        return False, ""

    def process_one_podcast(self) -> bool:
        """Process a single podcast. Returns True if a podcast was processed."""
        # First, clean up any stale processing items
        self._check_and_clean_stale_processing()

        # Create transcription request with our custom processor
        request = TranscriptionRequest(
            input=InputConfig(type="feed", source=self.config.feeds_path),
            processing=ProcessingConfig(
                model=self.config.model,
                verbose=self.config.verbose,
                feed_options=FeedOptions(
                    limit=1,  # Process only one podcast
                    order="newest",
                    date_threshold=self.date_threshold,
                    round_robin=self.config.round_robin,
                ),
            ),
            output=OutputConfig(
                format="sqlite",
                database=DatabaseConfig(
                    db_path=self.config.db_path,
                    metadata_table="transcription_metadata",
                    segments_table="transcription_segments",
                ),
            ),
        )

        # We need to intercept the processing to add our state tracking
        # For now, let's process normally and enhance beige-book later
        response = self.transcription_service.process(request)

        # Check if we processed anything
        if (
            not response.success
            or not response.summary
            or response.summary.processed == 0
        ):
            # Check if there were failures
            if response.errors:
                for error in response.errors:
                    logger.error(f"Processing error: {error.message}")
                    # TODO: Extract feed_url and feed_item_id from error context
                    # This would require enhancing beige-book to provide this info

            logger.info("No new podcasts to process")
            return False

        logger.info(f"Processed {response.summary.processed} podcast(s)")

        # Index the new transcription immediately
        try:
            rag_pipeline = RAGPipeline(
                ollama_client=self.ollama_client,
                db_path=self.config.db_path,
                vector_store_path=self.config.vector_store_path,
            )

            # Index all new transcriptions (should be just one)
            rag_pipeline.index_all_transcriptions(batch_size=1)
            logger.info("Successfully indexed transcription")
            return True

        except Exception as e:
            logger.error(f"Failed to index transcription: {e}")
            # TODO: Record indexing failure
            return False

    def run(self):
        """Run the synchronization process."""
        # Try to acquire lock
        if not self.process_lock.acquire():
            logger.error(
                f"Another instance is already processing feeds from: {self.config.feeds_path}\n"
                f"Lock file: {self.process_lock.lock_file}"
            )
            return

        try:
            if self.config.daemon:
                self._run_daemon()
            else:
                self._run_once()
        finally:
            self.process_lock.release()

    def _run_once(self):
        """Process all available podcasts one at a time."""
        processed_count = 0

        while True:
            if self.process_one_podcast():
                processed_count += 1
            else:
                # No more podcasts to process
                break

        if processed_count > 0:
            logger.info(
                f"Sync completed successfully! Processed {processed_count} podcast(s)."
            )
            print("\nYou can now query the podcasts using:")
            print(
                f'  flox activate -- uv run python -m grant ask "your question here" --vector-store {self.config.vector_store_path}'
            )
        else:
            logger.info("No new podcasts to process.")

    def _run_daemon(self):
        """Run in daemon mode with exponential backoff."""
        logger.info(f"Starting in daemon mode (PID: {self.pid}@{self.hostname})...")
        sleep_time = 60  # Start with 1 minute
        max_sleep_time = 3600  # Max 1 hour

        while True:
            try:
                if self.process_one_podcast():
                    # Successfully processed a podcast, reset sleep time
                    sleep_time = 60
                else:
                    # No podcast processed, exponential backoff
                    logger.info(
                        f"No new podcasts found. Sleeping for {sleep_time} seconds..."
                    )
                    time.sleep(sleep_time)
                    sleep_time = min(sleep_time * 2, max_sleep_time)
            except KeyboardInterrupt:
                logger.info("Daemon mode interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in daemon mode: {e}")
                time.sleep(60)  # Sleep 1 minute on error
