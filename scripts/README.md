# Pinkhaus Scripts

## sync_podcasts.py

Orchestrates the podcast transcription pipeline by running beige-book to fetch/transcribe podcasts and then grant to index them into a vector database.

The script processes podcasts one at a time, immediately indexing each one after transcription. This ensures the vector database is always up-to-date and allows for incremental processing.

**Important**: This script must be run from the pinkhaus root directory (not from the scripts directory).

### Usage

```bash
# From the pinkhaus root directory:
python scripts/sync_podcasts.py [options]
```

### Options

- `--since` / `--date-threshold`: Only process episodes published after this date (ISO8601 format)
- `--days`: Process episodes from the last N days (alternative to --since)
- `--feeds`: Path to TOML file containing RSS feed URLs (default: feeds.toml)
- `--db`: Path to beige-book database (default: ./resources/fc/fc.db)
- `--vector-store`: Path to Grant vector store directory (default: ./projects/grant/grant_chroma_db)
- `--model`: Whisper model to use for transcription (tiny, base, small, medium, large; default: tiny)
- `--round-robin`: Process feeds in round-robin mode (newest episode from each feed before moving to next)
- `--daemon`: Run in daemon mode (continuous processing with exponential backoff)
- `--dry-run`: Show commands that would be run without executing them
- `--verbose` / `-v`: Enable verbose logging

### Daemon Mode

When `--daemon` is enabled, the script runs continuously, checking for new podcasts and processing them as they become available. This is ideal for keeping your podcast database always up-to-date.

Features:
- Processes one podcast at a time, immediately indexing after transcription
- Uses exponential backoff when no new podcasts are found
- Starts with a 1-minute sleep, doubling up to a maximum of 1 hour
- Resets to 1-minute sleep when new podcasts are found
- Can be stopped with Ctrl+C

### Round-Robin Mode

When `--round-robin` is enabled, the script will download the newest episode from each podcast feed before moving to the next episode. This ensures you get the most recent content from all feeds rather than downloading the entire history of one feed before moving to the next.

This is particularly useful when:
- You have many podcast feeds
- You want to stay up-to-date with all feeds
- You have limited bandwidth or processing time

### One-at-a-Time Processing

The script processes podcasts individually:
1. Downloads and transcribes one podcast
2. Immediately indexes it into the vector database
3. Moves to the next podcast

This approach ensures:
- The vector database is always up-to-date
- Interruptions don't lose progress
- Memory usage stays low
- You can query recently processed podcasts immediately

### Examples

```bash
# Process last 7 days of podcasts
python scripts/sync_podcasts.py --days 7

# Process with round-robin using large model
python scripts/sync_podcasts.py --since 2024-01-01 --model large --round-robin

# Run in daemon mode with round-robin
python scripts/sync_podcasts.py --days 30 --round-robin --daemon

# Daemon mode with verbose logging
python scripts/sync_podcasts.py --since 2025-01-01 --daemon --verbose

# Dry run to see what would be executed
python scripts/sync_podcasts.py --days 30 --dry-run
```