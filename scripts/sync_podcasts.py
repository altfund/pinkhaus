#!/usr/bin/env python3
"""
Sync podcast transcriptions from RSS feeds to vector database.

This script:
1. Fetches and transcribes podcast episodes after a given date threshold
2. Indexes the transcriptions into Grant's vector database for RAG queries
"""

import argparse
import subprocess
import sys
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path


# Setup logging
logger = logging.getLogger(__name__)


def run_command(cmd: list[str], description: str, capture_output: bool = False) -> tuple[int, str]:
    """Run a command and return its exit code and output."""
    logger.info(f"{description}...")
    logger.debug(f"Running: {' '.join(cmd)}")

    try:
        if capture_output:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return result.returncode, result.stdout
        else:
            result = subprocess.run(cmd, check=True)
            return result.returncode, ""
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if capture_output and e.stdout:
            logger.debug(f"Output: {e.stdout}")
        return e.returncode, ""


def process_one_podcast(args, date_threshold: str) -> bool:
    """Process a single podcast. Returns True if a podcast was processed."""
    # Build beige-book command for one podcast
    beige_book_cmd = [
        "flox", "activate", "--",
        "uv", "run", "python", "-m", "beige_book",
        "transcribe",
        "--feed", args.feeds,
        "--format", "sqlite",
        "--db-path", args.db,
        "--model", args.model,
        "--date-threshold", date_threshold,
        "--limit", "1",  # Process only one podcast
    ]

    if args.round_robin:
        beige_book_cmd.append("--round-robin")
        
    if args.verbose:
        beige_book_cmd.append("--verbose")

    # Run beige-book transcribe for one podcast
    exit_code, output = run_command(
        beige_book_cmd,
        f"Fetching and transcribing one podcast after {date_threshold}",
        capture_output=True
    )

    if exit_code != 0:
        return False

    # Check if we actually processed something
    if "Processed: 0" in output or "Total items: 0" in output:
        logger.info("No new podcasts to process")
        return False

    # Run grant index immediately
    grant_cmd = [
        "flox", "activate", "--",
        "uv", "run", "python", "-m", "grant",
        "index",
        "--db", args.db,
        "--vector-store", args.vector_store,
    ]

    exit_code, _ = run_command(
        grant_cmd,
        "Indexing transcription into vector database"
    )

    if exit_code != 0:
        logger.error("Failed to index transcription")
        return False

    return True


def main():
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
        default="feeds.toml",
        help="Path to TOML file containing RSS feed URLs (default: feeds.toml)",
    )
    parser.add_argument(
        "--db",
        default="./resources/fc/fc.db",
        help="Path to beige-book database (default: beige_book_feeds.db)",
    )
    parser.add_argument(
        "--vector-store",
        default="./projects/grant/grant_chroma_db",
        help="Path to Grant vector store directory (default: ./grant_chroma_db)",
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
        "--dry-run",
        action="store_true",
        help="Show commands that would be run without executing them",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in daemon mode (continuous processing with exponential backoff)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Determine date threshold
    if args.days:
        date_threshold = (datetime.now() - timedelta(days=args.days)).isoformat()
    elif args.date_threshold:
        date_threshold = args.date_threshold
    else:
        # Default to last 30 days
        date_threshold = (datetime.now() - timedelta(days=30)).isoformat()

    logger.info(f"Syncing podcasts published after: {date_threshold}")

    # Check if feeds file exists
    if not Path(args.feeds).exists():
        logger.error(f"Feeds file not found: {args.feeds}")
        return 1

    if args.dry_run:
        # Show what would be run
        example_cmd = [
            "flox", "activate", "--",
            "uv", "run", "python", "-m", "beige_book",
            "transcribe",
            "--feed", args.feeds,
            "--format", "sqlite",
            "--db-path", args.db,
            "--model", args.model,
            "--date-threshold", date_threshold,
            "--limit", "1",
        ]
        if args.round_robin:
            example_cmd.append("--round-robin")
        if args.verbose:
            example_cmd.append("--verbose")
            
        print("\nDry run mode - commands that would be executed:")
        print(f"\n1. Fetch and transcribe one podcast:")
        print(f"   {' '.join(example_cmd)}")
        print(f"\n2. Index transcription:")
        print(f"   flox activate -- uv run python -m grant index --db {args.db} --vector-store {args.vector_store}")
        if args.daemon:
            print("\n3. Loop continuously with exponential backoff when no new podcasts are found")
        return 0

    if args.daemon:
        # Daemon mode - run continuously
        logger.info("Starting in daemon mode...")
        sleep_time = 60  # Start with 1 minute
        max_sleep_time = 3600  # Max 1 hour
        
        while True:
            try:
                if process_one_podcast(args, date_threshold):
                    # Successfully processed a podcast, reset sleep time
                    sleep_time = 60
                else:
                    # No podcast processed, exponential backoff
                    logger.info(f"No new podcasts found. Sleeping for {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    sleep_time = min(sleep_time * 2, max_sleep_time)
            except KeyboardInterrupt:
                logger.info("Daemon mode interrupted by user")
                return 0
            except Exception as e:
                logger.error(f"Unexpected error in daemon mode: {e}")
                time.sleep(60)  # Sleep 1 minute on error
    else:
        # Non-daemon mode - process all available podcasts one at a time
        processed_count = 0
        while True:
            if process_one_podcast(args, date_threshold):
                processed_count += 1
            else:
                # No more podcasts to process
                break
        
        if processed_count > 0:
            logger.info(f"\nSync completed successfully! Processed {processed_count} podcast(s).")
            print(f"\nYou can now query the podcasts using:")
            print(f"  flox activate -- uv run python -m grant ask \"your question here\" --vector-store {args.vector_store}")
        else:
            logger.info("No new podcasts to process.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
