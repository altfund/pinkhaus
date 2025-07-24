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
from .transcriber import AudioTranscriber
from .database import TranscriptionDatabase
from .feed_parser import FeedParser
from .downloader import AudioDownloader


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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def process_single_file(args):
    """Process a single audio file"""
    # Create transcriber with specified model
    transcriber = AudioTranscriber(model_name=args.model)

    # Transcribe the file (verbose only for text format)
    result = transcriber.transcribe_file(args.filename, verbose=(args.format == "text"))

    # Handle different output formats
    if args.format == "sqlite":
        # Save to SQLite database
        db = TranscriptionDatabase(args.db_path)
        db.create_tables(args.metadata_table, args.segments_table)
        transcription_id = db.save_transcription(
            result,
            model_name=args.model,
            metadata_table=args.metadata_table,
            segments_table=args.segments_table,
        )
        print(f"Transcription saved to database with ID: {transcription_id}")
        print(f"Database: {args.db_path}")
        print(f"Metadata table: {args.metadata_table}")
        print(f"Segments table: {args.segments_table}")
    else:
        # Format output for other formats
        output = result.format(args.format)

        # Write to file or stdout
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Transcription saved to {args.output}")
        else:
            print(output)


def process_feeds(args, is_resumable: bool):
    """Process RSS feeds from TOML file"""
    feed_parser = FeedParser()
    downloader = AudioDownloader()
    transcriber = AudioTranscriber(model_name=args.model)

    # Setup database if needed
    db = None
    if args.db_path or args.format == "sqlite":
        db_path = args.db_path or "beige_book_feeds.db"
        db = TranscriptionDatabase(db_path)
        db.create_tables(args.metadata_table, args.segments_table)

    # Parse feeds from TOML
    logging.info(f"Parsing feeds from: {args.filename}")
    feed_items_dict = feed_parser.parse_all_feeds(args.filename)

    # Calculate total items considering limit
    if args.limit:
        total_items = sum(
            min(len(items), args.limit) for items in feed_items_dict.values()
        )
    else:
        total_items = sum(len(items) for items in feed_items_dict.values())

    processed = 0
    skipped = 0

    for feed_url, items in feed_items_dict.items():
        logging.info(f"Processing feed: {feed_url} with {len(items)} items")

        # Sort items based on order preference
        if args.order == "newest":
            # Sort by published date descending (newest first)
            sorted_items = sorted(
                items, key=lambda x: x.published or datetime.min, reverse=True
            )
        else:
            # Sort by published date ascending (oldest first)
            sorted_items = sorted(items, key=lambda x: x.published or datetime.min)

        # Apply limit if specified
        if args.limit:
            sorted_items = sorted_items[: args.limit]
            logging.info(f"Limiting to {args.limit} items per feed")

        for item in sorted_items:
            # Check if already processed (only if resumable)
            if (
                is_resumable
                and db
                and db.check_feed_item_exists(
                    item.feed_url, item.item_id, args.metadata_table
                )
            ):
                logging.info(f"Skipping already processed: {item.title}")
                skipped += 1
                continue

            try:
                # Download audio file
                logging.info(f"Processing: {item.title}")
                temp_path, file_hash = downloader.download_with_retry(item.audio_url)

                try:
                    # Transcribe the downloaded file
                    result = transcriber.transcribe_file(
                        temp_path, verbose=(args.format == "text" and not args.output)
                    )

                    # Save or output based on format
                    if args.format == "sqlite" or (is_resumable and db):
                        # Save to database with feed metadata
                        transcription_id = db.save_transcription(
                            result,
                            model_name=args.model,
                            metadata_table=args.metadata_table,
                            segments_table=args.segments_table,
                            feed_url=item.feed_url,
                            feed_item_id=item.item_id,
                            feed_item_title=item.title,
                            feed_item_published=item.published.isoformat()
                            if item.published
                            else None,
                        )
                        logging.info(f"Saved to database with ID: {transcription_id}")
                    else:
                        # Format output with feed metadata
                        output = format_with_feed_metadata(result, item, args.format)

                        if args.output:
                            # Append to output file
                            mode = "a" if processed > 0 else "w"
                            with open(args.output, mode, encoding="utf-8") as f:
                                if processed > 0:
                                    f.write("\n\n")  # Separator between items
                                f.write(output)
                        else:
                            print(output)
                            if processed < total_items - 1:
                                print("\n" + "=" * 80 + "\n")  # Separator

                    processed += 1

                finally:
                    # Clean up temp file
                    downloader.cleanup_temp_file(temp_path)

            except Exception as e:
                logging.error(f"Failed to process {item.title}: {e}")
                continue

    # Summary
    print("\nProcessing complete:")
    print(f"  Total items: {total_items}")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already processed): {skipped}")
    print(f"  Failed: {total_items - processed - skipped}")


def format_with_feed_metadata(result, feed_item, format_type: str) -> str:
    """Format transcription output with feed metadata"""
    if format_type == "text":
        # For text format, add header with feed info
        header = f"Feed: {feed_item.feed_url}\n"
        header += f"Title: {feed_item.title}\n"
        if feed_item.published:
            header += f"Published: {feed_item.published}\n"
        header += f"Audio URL: {feed_item.audio_url}\n"
        header += "-" * 80 + "\n\n"
        return header + result.full_text

    elif format_type == "json":
        # Add feed metadata to JSON output
        import json

        data = result.to_dict()
        data["feed_metadata"] = feed_item.to_dict()
        return json.dumps(data, indent=2, ensure_ascii=False)

    elif format_type == "toml":
        # Add feed metadata to TOML output
        output = "[feed_metadata]\n"
        output += f'feed_url = "{feed_item.feed_url}"\n'
        output += f'item_id = "{feed_item.item_id}"\n'
        output += f'title = "{feed_item.title}"\n'
        output += f'audio_url = "{feed_item.audio_url}"\n'
        if feed_item.published:
            output += f'published = "{feed_item.published.isoformat()}"\n'
        output += "\n" + result.to_toml()
        return output

    else:
        # For other formats, prepend a comment with feed info
        header = f"# Feed: {feed_item.feed_url}\n"
        header += f"# Title: {feed_item.title}\n"
        if feed_item.published:
            header += f"# Published: {feed_item.published}\n"
        return header + result.format(format_type)


def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description="Transcribe audio files with Whisper")
    parser.add_argument(
        "filename", type=valid_path, help="Audio file or TOML feed file to transcribe"
    )
    parser.add_argument(
        "--model",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use",
    )
    parser.add_argument(
        "--format",
        default="text",
        choices=["text", "json", "table", "csv", "toml", "sqlite"],
        help="Output format (default: text)",
    )
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    # Database-specific arguments
    parser.add_argument(
        "--db-path", help="Path to SQLite database file (required for sqlite format)"
    )
    parser.add_argument(
        "--metadata-table",
        default="transcription_metadata",
        help="Name of the metadata table (default: transcription_metadata)",
    )
    parser.add_argument(
        "--segments-table",
        default="transcription_segments",
        help="Name of the segments table (default: transcription_segments)",
    )

    # Feed-specific arguments
    parser.add_argument(
        "--feed",
        action="store_true",
        help="Treat input as a TOML file containing RSS feed URLs",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum number of feed items to process per feed"
    )
    parser.add_argument(
        "--order",
        choices=["newest", "oldest"],
        default="newest",
        help="Process feed items from newest or oldest first (default: newest)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Validate database arguments
    if args.format == "sqlite" and not args.db_path:
        parser.error("--db-path is required when using sqlite format")

    # Check if resumability is needed
    resumable_formats = {"text", "json", "table", "csv", "toml", "sqlite"}
    is_resumable = args.format in resumable_formats and (args.db_path or args.output)

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
