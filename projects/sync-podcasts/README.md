# Sync Podcasts

Orchestrates the podcast transcription pipeline by using beige-book to fetch/transcribe podcasts and grant to index them into a vector database.

This is a proper Python package that imports and uses beige-book and grant as libraries, providing a cleaner and more maintainable approach than shell scripts.

## Features

- **One-at-a-time processing**: Downloads and transcribes one podcast, then immediately indexes it
- **Round-robin mode**: Processes the newest episode from each feed before moving to older episodes
- **Daemon mode**: Runs continuously with exponential backoff when no new podcasts are found
- **Smart date filtering**: Only processes podcasts after a specified date threshold
- **Integrated pipeline**: Uses beige-book and grant as Python libraries for better performance
- **Robust processing**: Handles interruptions gracefully with state tracking and failure management

### Robustness Features

- **Per-Feeds-File Locking**: Prevents multiple instances from processing the same feeds file while allowing concurrent processing of different feeds files
- **Dynamic Feeds Reloading**: Reads feeds.toml fresh on each processing cycle, allowing live updates
- **Failed Item Tracking**: Records failed processing attempts with error details
- **Automatic Retry Limits**: Skips items after configurable failure threshold (default: 3 attempts)
- **Processing State Tracking**: Tracks current processing state (downloading, transcribing, indexing)
- **Stale Process Detection**: Automatically cleans up stale processing states after timeout
- **Interruption Recovery**: Can resume processing after unexpected interruptions
- **Startup Report**: Shows unprocessed items per feed and failed items summary on startup

## Installation

```bash
cd projects/sync-podcasts
flox activate
```

This will:
- Install Python 3.13
- Create a virtual environment
- Install all dependencies including beige-book and grant

## Usage

```bash
# Process last 7 days of podcasts
flox activate -- sync-podcasts --days 7

# Process with round-robin using large model
flox activate -- sync-podcasts --since 2024-01-01 --model large --round-robin

# Run in daemon mode with round-robin
flox activate -- sync-podcasts --days 30 --round-robin --daemon

# Daemon mode with verbose logging
flox activate -- sync-podcasts --since 2025-01-01 --daemon --verbose

# Run multiple instances with different feeds files
flox activate -- sync-podcasts --feeds ./tech-podcasts.toml --db ./tech.db --daemon
flox activate -- sync-podcasts --feeds ./news-podcasts.toml --db ./news.db --daemon
```

## Options

- `--since` / `--date-threshold`: Only process episodes published after this date (ISO8601 format)
- `--days`: Process episodes from the last N days (alternative to --since)
- `--feeds`: Path to TOML file containing RSS feed URLs (default: ./resources/fc/feeds.toml)
- `--db`: Path to database (default: ./resources/fc/fc.db)
- `--vector-store`: Path to Grant vector store directory (default: ./projects/grant/grant_chroma_db)
- `--model`: Whisper model to use (tiny, base, small, medium, large; default: tiny)
- `--round-robin`: Process feeds in round-robin mode
- `--daemon`: Run in daemon mode (continuous processing with exponential backoff)
- `--ollama-base-url`: Ollama API base URL (default: http://localhost:11434)
- `--verbose` / `-v`: Enable verbose logging

## Daemon Mode

When `--daemon` is enabled, the service runs continuously:
- Processes one podcast at a time, immediately indexing after transcription
- Uses exponential backoff when no new podcasts are found
- Starts with a 1-minute sleep, doubling up to a maximum of 1 hour
- Resets to 1-minute sleep when new podcasts are found
- Can be stopped with Ctrl+C

## Round-Robin Mode

When `--round-robin` is enabled:
- Downloads the newest episode from each podcast feed before moving to older episodes
- Ensures you get the most recent content from all feeds
- Prevents one feed from monopolizing the download queue

## Development

```bash
# Run tests
flox activate -- just test

# Format code
flox activate -- just fmt

# Lint code
flox activate -- just lint
```

## Architecture

This package uses:
- `beige-book.TranscriptionService` for podcast fetching and transcription
- `grant.RAGPipeline` for indexing into the vector database
- `pinkhaus-models.TranscriptionDatabase` for state tracking and failure management
- Native Python integration instead of subprocess calls for better error handling and performance

### Database Tables

The robustness features use additional database tables:

1. **failed_items**: Tracks failed processing attempts
   - Records feed URL, item ID, error type, and message
   - Increments failure count on repeated failures
   - Items with 3+ failures are permanently skipped

2. **processing_state**: Tracks current processing state
   - Records which items are being processed
   - Includes PID and hostname for distributed systems
   - Automatically cleaned up if stale (>30 minutes)

### Process Lock

A file-based lock prevents multiple instances from processing the same feeds file. The lock file is created at `/tmp/sync_podcasts_<hash>.lock` where `<hash>` is derived from the feeds file path. This allows:

- Multiple instances to run concurrently with different feeds files
- The lock file contains the PID, hostname, and feeds file path
- You can update the feeds.toml while the program is running - changes will be picked up on the next processing cycle

### Startup Report

On startup, sync-podcasts displays a comprehensive report showing:

1. **Failed Items Summary**: Items that failed processing, grouped by feed
   - Shows failure count and maximum retry attempts
   - Items with 3+ failures are permanently skipped

2. **Unprocessed Items Count**: Number of new items to process per feed
   - Only counts items within the date threshold
   - Excludes permanently failed items
   - Shows as "feed_url: X/Y items to process" where X is unprocessed and Y is total

Example startup report:
```
======================================================================
STARTUP REPORT
======================================================================
Failed items found in 'failed_items' table:
  - https://example.com/feed1.rss: 2 failed items (max failures: 2)
Unprocessed items found (will be stored in 'transcription_metadata' table):
  - https://example.com/feed1.rss: 5/10 items to process
  - https://example.com/feed2.rss: 3/8 items to process
Total unprocessed items: 8
======================================================================
```