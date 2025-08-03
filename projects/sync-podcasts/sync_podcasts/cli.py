"""CLI interface for sync-podcasts."""

import argparse
import logging
import sys
from pathlib import Path

from .sync import PodcastSyncer, SyncConfig


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Main entry point for sync-podcasts CLI."""
    parser = argparse.ArgumentParser(
        description="Sync podcast transcriptions to vector database"
    )
    parser.add_argument(
        "--since",
        "--date-threshold",
        dest="date_threshold",
        help="Only process episodes published after this date (ISO8601 format)",
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Process episodes from the last N days (alternative to --since)",
    )
    parser.add_argument(
        "--feeds",
        default="./resources/fc/feeds.toml",
        help="Path to TOML file containing RSS feed URLs (default: ./resources/fc/feeds.toml)",
    )
    parser.add_argument(
        "--db",
        default="./resources/fc/fc.db",
        help="Path to beige-book database (default: ./resources/fc/fc.db)",
    )
    parser.add_argument(
        "--vector-store",
        default="./projects/grant/grant_chroma_db",
        help="Path to Grant vector store directory (default: ./projects/grant/grant_chroma_db)",
    )
    parser.add_argument(
        "--model",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use for transcription (default: tiny)",
    )
    parser.add_argument(
        "--round-robin",
        action="store_true",
        help="Process feeds in round-robin mode (newest episode from each feed before moving to next)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in daemon mode (continuous processing with exponential backoff)",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Check if feeds file exists
    if not Path(args.feeds).exists():
        print(f"Error: Feeds file not found: {args.feeds}", file=sys.stderr)
        sys.exit(1)

    # Create configuration
    config = SyncConfig(
        feeds_path=args.feeds,
        db_path=args.db,
        vector_store_path=args.vector_store,
        model=args.model,
        round_robin=args.round_robin,
        daemon=args.daemon,
        date_threshold=args.date_threshold,
        days_back=args.days,
        ollama_base_url=args.ollama_base_url,
        verbose=args.verbose,
    )

    # Run the syncer
    try:
        syncer = PodcastSyncer(config)
        syncer.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
