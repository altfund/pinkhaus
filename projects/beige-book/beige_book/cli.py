#!/usr/bin/env python3
"""
Command-line interface for the Beige Book transcription tool.
"""

import sys
import argparse
import os
import traceback
import logging
from datetime import datetime
from .models import (
    TranscriptionRequest, InputConfig, ProcessingConfig, OutputConfig,
    FeedOptions, DatabaseConfig
)
from .service import TranscriptionService, OutputFormatter


def valid_path(path):
    """Validate that the path exists and is a file"""
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"Path '{path}' does not exist")
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"Path '{path}' is not a file")
    return path


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def args_to_request(args) -> TranscriptionRequest:
    """Convert CLI arguments to TranscriptionRequest"""
    # Determine input type
    input_config = InputConfig(
        type="feed" if args.feed else "file",
        source=args.filename
    )

    # Create feed options if processing feeds
    feed_options = None
    if args.feed:
        feed_options = FeedOptions(
            limit=args.limit,
            order=args.order
        )

    # Create processing config
    processing_config = ProcessingConfig(
        model=args.model,
        verbose=args.verbose,
        feed_options=feed_options
    )

    # Create database config if needed
    database_config = None
    if args.db_path or args.format == "sqlite":
        database_config = DatabaseConfig(
            db_path=args.db_path or "beige_book.db",
            metadata_table=args.metadata_table,
            segments_table=args.segments_table
        )

    # Create output config
    output_config = OutputConfig(
        format=args.format,
        destination=args.output,
        database=database_config
    )

    return TranscriptionRequest(
        input=input_config,
        processing=processing_config,
        output=output_config
    )


def process_single_file(args):
    """Process a single audio file using the service"""
    service = TranscriptionService()

    # Convert args to request
    request = args_to_request(args)

    # Process the request
    response = service.process(request)

    # Handle output
    if response.success and response.results:
        result = response.results[0]

        if args.format == "sqlite":
            # Already saved by service
            print(f"Saved to database: {args.db_path or 'beige_book.db'}")
        else:
            # Format and output
            formatted = OutputFormatter.format_results(
                response.results,
                args.format,
                include_feed_metadata=False
            )

            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(formatted)
                print(f"Output written to: {args.output}")
            else:
                print(formatted)
    else:
        # Handle errors
        for error in response.errors:
            print(f"Error: {error.message}", file=sys.stderr)
        sys.exit(1)


def process_feeds(args, is_resumable: bool):
    """Process RSS feeds using the service"""
    service = TranscriptionService()

    # Convert args to request
    request = args_to_request(args)

    # Process the request
    response = service.process(request)

    # Handle output
    if args.format == "sqlite":
        # Results already saved by service
        if response.success:
            print(f"\nProcessing complete:")
            if response.summary:
                print(f"  Total items: {response.summary.total_items}")
                print(f"  Processed: {response.summary.processed}")
                print(f"  Skipped (already processed): {response.summary.skipped}")
                print(f"  Failed: {response.summary.failed}")
                print(f"  Time elapsed: {response.summary.elapsed_time:.2f}s")
    else:
        # Format and output results
        if response.results:
            formatted = OutputFormatter.format_results(
                response.results,
                args.format,
                include_feed_metadata=True
            )

            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(formatted)
                print(f"\nOutput written to: {args.output}")
            else:
                print(formatted)

        # Print summary
        print(f"\nProcessing complete:")
        if response.summary:
            print(f"  Total items: {response.summary.total_items}")
            print(f"  Processed: {response.summary.processed}")
            print(f"  Skipped (already processed): {response.summary.skipped}")
            print(f"  Failed: {response.summary.failed}")
            print(f"  Time elapsed: {response.summary.elapsed_time:.2f}s")

    # Print errors if any
    if response.errors:
        print("\nErrors encountered:")
        for error in response.errors:
            print(f"  - {error.source}: {error.message}")

    # Exit with error code if not successful
    if not response.success:
        sys.exit(1)


def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description="Transcribe audio files with Whisper")
    parser.add_argument("filename",
                        type=valid_path,
                        help="Audio file or TOML feed file to transcribe")
    parser.add_argument(
        "--model",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use"
    )
    parser.add_argument(
        "--format",
        default="text",
        choices=["text", "json", "table", "csv", "toml", "sqlite"],
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file (default: stdout)"
    )

    # Database-specific arguments
    parser.add_argument(
        "--db-path",
        help="Path to SQLite database file (required for sqlite format)"
    )
    parser.add_argument(
        "--metadata-table",
        default="transcription_metadata",
        help="Name of the metadata table (default: transcription_metadata)"
    )
    parser.add_argument(
        "--segments-table",
        default="transcription_segments",
        help="Name of the segments table (default: transcription_segments)"
    )

    # Feed-specific arguments
    parser.add_argument(
        "--feed",
        action="store_true",
        help="Treat input as a TOML file containing RSS feed URLs"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of feed items to process per feed"
    )
    parser.add_argument(
        "--order",
        choices=["newest", "oldest"],
        default="newest",
        help="Process feed items from newest or oldest first (default: newest)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Validate database arguments
    if args.format == "sqlite" and not args.db_path:
        parser.error("--db-path is required when using sqlite format")

    # Check if resumability is needed
    resumable_formats = {"text", "json", "table", "csv", "toml", "sqlite"}
    is_resumable = (args.format in resumable_formats and
                   (args.db_path or args.output))

    try:
        if args.feed:
            # Process RSS feeds from TOML file
            process_feeds(args, is_resumable)
        else:
            # Process single audio file
            process_single_file(args)

    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found", file=sys.stderr)
        if args.format == "text":  # Only show traceback in text mode
            traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.format == "text":  # Only show traceback in text mode
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
