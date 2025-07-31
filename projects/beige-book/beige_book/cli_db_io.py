#!/usr/bin/env python3
"""
Command-line interface for database import/export operations.

Usage:
    # Export all transcriptions to JSON
    python -m beige_book.cli_db_io export --db transcriptions.db --output backup.json
    
    # Export to TOML
    python -m beige_book.cli_db_io export --db transcriptions.db --output backup.toml --format toml
    
    # Export single transcription
    python -m beige_book.cli_db_io export --db transcriptions.db --output single.json --id 42
    
    # Import from JSON
    python -m beige_book.cli_db_io import --db new.db --input backup.json
    
    # Import from TOML
    python -m beige_book.cli_db_io import --db new.db --input backup.toml
"""

import argparse
import sys
from pathlib import Path
from .database_io import DatabaseIO
from pinkhaus_models import TranscriptionDatabase


def export_command(args):
    """Handle export command."""
    # Check database exists
    if not Path(args.db).exists():
        print(f"Error: Database file '{args.db}' not found", file=sys.stderr)
        return 1
    
    io = DatabaseIO(args.db)
    
    # Determine format from filename if not specified
    if not args.format:
        if args.output.endswith('.toml'):
            args.format = 'toml'
        else:
            args.format = 'json'
    
    try:
        if args.id:
            # Export single transcription
            if args.format == 'json':
                success = io.export_transcription_to_json(args.id, args.output)
            else:
                success = io.export_transcription_to_toml(args.id, args.output)
            
            if success:
                print(f"Exported transcription {args.id} to {args.output}")
            else:
                print(f"Error: Transcription {args.id} not found", file=sys.stderr)
                return 1
        else:
            # Export all transcriptions
            if args.format == 'json':
                count = io.export_all_to_json(args.output)
            else:
                count = io.export_all_to_toml(args.output)
            
            print(f"Exported {count} transcription(s) to {args.output}")
    
    except Exception as e:
        print(f"Error during export: {e}", file=sys.stderr)
        return 1
    
    return 0


def import_command(args):
    """Handle import command."""
    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        return 1
    
    # Create database if it doesn't exist
    if not Path(args.db).exists():
        print(f"Creating new database: {args.db}")
        db = TranscriptionDatabase(args.db)
        db.create_tables()
    
    io = DatabaseIO(args.db)
    
    # Determine format from filename
    if args.input.endswith('.toml'):
        format_type = 'toml'
    else:
        format_type = 'json'
    
    try:
        if format_type == 'json':
            result = io.import_from_json(args.input, skip_duplicates=not args.force)
        else:
            result = io.import_from_toml(args.input, skip_duplicates=not args.force)
        
        print(f"Imported {result['imported']} transcription(s)")
        if result['skipped'] > 0:
            print(f"Skipped {result['skipped']} duplicate transcription(s)")
            if not args.force:
                print("Use --force to import duplicates")
    
    except Exception as e:
        print(f"Error during import: {e}", file=sys.stderr)
        return 1
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Database import/export utility for transcriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export transcriptions from database')
    export_parser.add_argument('--db', required=True, help='Database file path')
    export_parser.add_argument('--output', '-o', required=True, help='Output file path')
    export_parser.add_argument('--format', choices=['json', 'toml'], 
                             help='Output format (default: auto-detect from filename)')
    export_parser.add_argument('--id', type=int, help='Export single transcription by ID')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import transcriptions to database')
    import_parser.add_argument('--db', required=True, help='Database file path')
    import_parser.add_argument('--input', '-i', required=True, help='Input file path')
    import_parser.add_argument('--force', action='store_true', 
                             help='Import duplicates (default: skip)')
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'export':
        return export_command(args)
    elif args.command == 'import':
        return import_command(args)


if __name__ == '__main__':
    sys.exit(main())