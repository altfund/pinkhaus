# Sync Podcasts

Orchestrates the podcast transcription pipeline by using beige-book to fetch/transcribe podcasts and grant to index them into a vector database.

This is a proper Python package that imports and uses beige-book and grant as libraries, providing a cleaner and more maintainable approach than shell scripts.

## Features

- **One-at-a-time processing**: Downloads and transcribes one podcast, then immediately indexes it
- **Round-robin mode**: Processes the newest episode from each feed before moving to older episodes
- **Daemon mode**: Runs continuously with exponential backoff when no new podcasts are found
- **Smart date filtering**: Only processes podcasts after a specified date threshold
- **Integrated pipeline**: Uses beige-book and grant as Python libraries for better performance

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
- Native Python integration instead of subprocess calls for better error handling and performance