"""CLI interface for sync-podcasts."""

import argparse
import logging
import sys
from pathlib import Path

from .sync import PodcastSyncer, SyncConfig
from .validate_feed import FeedValidator, validate_feeds_toml


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


def validate_feed_main():
    """Entry point for validate-feed command."""
    parser = argparse.ArgumentParser(
        description="Validate RSS feeds for podcast compatibility"
    )
    parser.add_argument(
        "input",
        help="URL of RSS feed or path to TOML file containing feed URLs",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--move-invalid",
        action="store_true",
        help="Move invalid feeds to invalid-feeds.toml (only for TOML input)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Check if input is a file or URL
    if Path(args.input).exists() and args.input.endswith('.toml'):
        # Validate all feeds in TOML file
        print(f"Validating feeds from: {args.input}")
        results = validate_feeds_toml(args.input, move_invalid=args.move_invalid)
        
        # Check for TOML parsing error
        if "_error" in results:
            print(f"✗ {results['_error'].error_message}", file=sys.stderr)
            sys.exit(1)
        
        # Display results for each feed
        total_feeds = len(results)
        valid_feeds = sum(1 for r in results.values() if r.is_valid)
        
        print(f"\nValidating {total_feeds} feed(s)...\n")
        
        for feed_url, result in results.items():
            if result.is_valid:
                print(f"✓ {feed_url}")
                print(f"  Title: {result.feed_title}")
                print(f"  Type: {result.feed_type}")
                print(f"  Audio episodes: {result.audio_entries}")
            else:
                print(f"✗ {feed_url}")
                print(f"  Error: {result.error_message}")
                if result.feed_title:
                    print(f"  Title: {result.feed_title}")
                if result.total_entries > 0:
                    print(f"  Total entries: {result.total_entries}")
            print()
        
        # Summary
        print(f"Summary: {valid_feeds}/{total_feeds} valid feeds")
        
        # If we moved invalid feeds, note that
        if args.move_invalid and valid_feeds < total_feeds:
            invalid_count = total_feeds - valid_feeds
            print(f"\n{invalid_count} invalid feed(s) moved to invalid-feeds.toml")
            print(f"Original feeds.toml updated with {valid_feeds} valid feed(s)")
        
        # Exit with error if any feeds are invalid
        if valid_feeds < total_feeds:
            sys.exit(1)
        else:
            sys.exit(0)
    else:
        # Validate single URL
        if args.move_invalid:
            print("Warning: --move-invalid is only applicable for TOML files, ignoring this option", file=sys.stderr)
        
        validator = FeedValidator()
        result = validator.validate_feed(args.input)
        
        if result.is_valid:
            print(f"✓ Valid {result.feed_type} feed")
            print(f"Title: {result.feed_title}")
            if result.feed_description:
                print(f"Description: {result.feed_description[:100]}{'...' if len(result.feed_description) > 100 else ''}")
            print(f"Total entries: {result.total_entries}")
            print(f"Audio entries: {result.audio_entries}")
            sys.exit(0)
        else:
            print(f"✗ Invalid feed: {result.error_message}", file=sys.stderr)
            if result.feed_title:
                print(f"Feed title: {result.feed_title}", file=sys.stderr)
            if result.total_entries > 0:
                print(f"Total entries found: {result.total_entries}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
