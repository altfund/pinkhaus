#!/usr/bin/env python3
"""
Command-line interface for the Beige Book transcription tool.
"""

import sys
import argparse
import os
import traceback
from .transcriber import AudioTranscriber
from .database import TranscriptionDatabase


def valid_path(path):
    """Validate that the path exists and is a file"""
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"Path '{path}' does not exist")
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"Path '{path}' is not a file")
    return path


def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description="Transcribe audio files with Whisper")
    parser.add_argument("filename",
                        type=valid_path,
                        help="Audio file to transcribe")
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
    
    args = parser.parse_args()
    
    # Validate database arguments
    if args.format == "sqlite" and not args.db_path:
        parser.error("--db-path is required when using sqlite format")
    
    try:
        # Create transcriber with specified model
        transcriber = AudioTranscriber(model_name=args.model)
        
        # Transcribe the file (verbose only for text format)
        result = transcriber.transcribe_file(
            args.filename, 
            verbose=(args.format == "text")
        )
        
        # Handle different output formats
        if args.format == "sqlite":
            # Save to SQLite database
            db = TranscriptionDatabase(args.db_path)
            db.create_tables(args.metadata_table, args.segments_table)
            transcription_id = db.save_transcription(
                result, 
                model_name=args.model,
                metadata_table=args.metadata_table,
                segments_table=args.segments_table
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
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(output)
                print(f"Transcription saved to {args.output}")
            else:
                print(output)
            
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