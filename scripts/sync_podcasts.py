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
from datetime import datetime, timedelta
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return its exit code."""
    print(f"\n{description}...")
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        return e.returncode


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

    args = parser.parse_args()

    # Determine date threshold
    if args.days:
        date_threshold = (datetime.now() - timedelta(days=args.days)).isoformat()
    elif args.date_threshold:
        date_threshold = args.date_threshold
    else:
        # Default to last 30 days
        date_threshold = (datetime.now() - timedelta(days=30)).isoformat()

    print(f"Syncing podcasts published after: {date_threshold}")

    # Check if feeds file exists
    if not Path(args.feeds).exists():
        print(f"Error: Feeds file not found: {args.feeds}")
        return 1

    # Build beige-book command
    beige_book_cmd = [
        "flox", "activate", "--",
        "uv", "run", "python", "-m", "beige_book",
        "transcribe",
        "--feed", args.feeds,
        "--format", "sqlite",
        "--db-path", args.db,
        "--model", args.model,
        "--date-threshold", date_threshold,
    ]

    if args.verbose:
        beige_book_cmd.append("--verbose")

    # Build grant command
    grant_cmd = [
        "flox", "activate", "--",
        "uv", "run", "python", "-m", "grant",
        "index",
        "--db", args.db,
        "--vector-store", args.vector_store,
    ]

    if args.dry_run:
        print("\nDry run mode - commands that would be executed:")
        print(f"\n1. Fetch and transcribe podcasts:")
        print(f"   {' '.join(beige_book_cmd)}")
        print(f"\n2. Index transcriptions:")
        print(f"   {' '.join(grant_cmd)}")
        return 0

    # Run beige-book transcribe
    exit_code = run_command(
        beige_book_cmd,
        f"Fetching and transcribing podcasts after {date_threshold}"
    )

    if exit_code != 0:
        print(f"Error: beige-book transcribe failed with exit code {exit_code}")
        return exit_code

    # Run grant index
    exit_code = run_command(
        grant_cmd,
        "Indexing transcriptions into vector database"
    )

    if exit_code != 0:
        print(f"Error: grant index failed with exit code {exit_code}")
        return exit_code

    print("\nSync completed successfully!")
    print(f"\nYou can now query the podcasts using:")
    print(f"  flox activate -- uv run python -m grant ask \"your question here\" --vector-store {args.vector_store}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
