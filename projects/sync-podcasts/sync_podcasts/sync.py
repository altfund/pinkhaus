"""Core synchronization logic for podcast pipeline."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from beige_book import TranscriptionService
from beige_book.models import (
    TranscriptionRequest,
    InputConfig,
    ProcessingConfig,
    OutputConfig,
    FeedOptions,
    DatabaseConfig,
)
from grant import RAGPipeline, OllamaClient

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


class PodcastSyncer:
    """Orchestrates podcast transcription and indexing."""

    def __init__(self, config: SyncConfig = SyncConfig()):
        self.config = config
        self.transcription_service = TranscriptionService()
        self.ollama_client = OllamaClient(base_url=config.ollama_base_url)

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

    def process_one_podcast(self) -> bool:
        """Process a single podcast. Returns True if a podcast was processed."""
        # Create transcription request
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

        # Process the request
        response = self.transcription_service.process(request)

        # Check if we processed anything
        if (
            not response.success
            or not response.summary
            or response.summary.processed == 0
        ):
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
            return False

    def run(self):
        """Run the synchronization process."""
        if self.config.daemon:
            self._run_daemon()
        else:
            self._run_once()

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
        logger.info("Starting in daemon mode...")
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
