# Voice Matching Tool

The `beige-book-match-voice` tool allows you to check if voices in a new audio file match any speakers already in your database. This is useful for:

- Verifying if you've appeared on a podcast before
- Finding all episodes where a specific person has spoken
- Cross-referencing speakers across different podcasts
- Building a speaker appearance database

## Prerequisites

1. **Existing Database**: You need a database with speaker profiles created using `beige-book` with `--diarize --speaker-profiles` flags
2. **HF_TOKEN**: Set your Hugging Face token as an environment variable
3. **Audio File**: The audio file you want to analyze

## Basic Usage

```bash
# Match voices in your audio against an existing database
beige-book-match-voice /path/to/your-audio.wav /path/to/database.db
```

## Options

- `--threshold FLOAT`: Similarity threshold for matching (0.0-1.0, default: 0.85)
  - Higher values (0.9+) = more strict matching, fewer false positives
  - Lower values (0.7-0.8) = more lenient, may catch variations in voice
  
- `--embedding-method {speechbrain,pyannote,mock}`: Voice embedding method (default: speechbrain)
  - Must match the method used when building the database
  
- `--min-duration FLOAT`: Minimum speech duration in seconds (default: 3.0)
  - Shorter segments may not have enough voice data for reliable matching
  
- `--format {summary,detailed,json}`: Output format (default: summary)
  - `summary`: Concise overview with grouped results
  - `detailed`: Full list of all appearances
  - `json`: Machine-readable JSON output
  
- `--output FILE`: Save results to file instead of printing
- `--quiet`: Suppress progress messages

## Examples

### Basic Voice Matching

```bash
# Check if your voice appears in a podcast database
beige-book-match-voice my-interview.mp3 podcast-archive.db
```

Output:
```
=== Voice Matching Results ===

Total speakers detected: 2
Speakers matched: 1
Total appearances: 5
Unique podcasts: 3
Unique feeds: 2

SPEAKER_00:
  Profile: John Doe
  Confidence: 0.923
  Duration: 145.3s
  Found in 5 recordings:
    - https://podcast1.com/feed.rss: 3 episodes
      • episode-042.mp3
      • episode-038.mp3
      • episode-021.mp3
    - https://podcast2.com/feed.rss: 2 episodes
      • interview-015.mp3
      • panel-discussion.mp3

SPEAKER_01:
  Profile: Unknown
  Confidence: 0.000
  Duration: 89.7s
  No matches found in database
```

### Detailed Output

```bash
# See all appearances with full metadata
beige-book-match-voice my-interview.mp3 podcast-archive.db --format detailed
```

### JSON Output for Processing

```bash
# Get structured data for further processing
beige-book-match-voice my-interview.mp3 podcast-archive.db --format json > matches.json
```

### Adjusting Match Sensitivity

```bash
# More strict matching (reduce false positives)
beige-book-match-voice audio.wav database.db --threshold 0.95

# More lenient matching (catch voice variations)
beige-book-match-voice audio.wav database.db --threshold 0.75
```

## Building a Voice Database

Before you can match voices, you need a database with speaker profiles:

```bash
# Process a single file with speaker profiling
beige-book podcast-episode.mp3 --db-path voices.db --format sqlite --diarize --speaker-profiles

# Process an RSS feed
beige-book feeds.toml --db-path voices.db --format sqlite --feed --diarize --speaker-profiles
```

## How It Works

1. **Speaker Diarization**: The tool first identifies different speakers in your audio
2. **Voice Embedding**: Extracts numerical "fingerprints" for each speaker's voice
3. **Database Matching**: Compares embeddings against all profiles in the database
4. **Result Aggregation**: Groups matches by podcast/feed and calculates confidence

## Accuracy Considerations

- **Voice Quality**: Clear audio with minimal background noise works best
- **Speech Duration**: Speakers need at least 3 seconds of speech for reliable matching
- **Voice Variations**: Illness, aging, or recording quality can affect matching
- **False Positives**: Very similar voices might be confused (adjust threshold)

## Privacy Notes

- Voice embeddings are numerical representations, not actual audio
- The tool only compares against speakers already in your database
- No data is sent to external services (all processing is local)

## Troubleshooting

### "No matches found" but you expect matches
- Check if the database actually contains the expected speakers
- Try lowering the threshold to 0.75 or 0.80
- Ensure you're using the same embedding method as the database

### "HF_TOKEN environment variable is required"
- Set your token: `export HF_TOKEN='hf_...'`
- Ensure you've accepted the pyannote model licenses

### High false positive rate
- Increase the threshold to 0.90 or 0.95
- Check if audio quality is consistent between files

## Advanced Usage

### Batch Processing

```bash
# Check multiple audio files
for audio in interviews/*.mp3; do
    echo "=== Checking $audio ==="
    beige-book-match-voice "$audio" database.db --format summary
    echo
done
```

### Building a Cross-Podcast Speaker Index

```python
import json
import subprocess
from pathlib import Path

# Process all podcasts
for feed in Path("feeds").glob("*.toml"):
    subprocess.run([
        "beige-book", str(feed), 
        "--db-path", "speaker-index.db",
        "--format", "sqlite",
        "--feed",
        "--diarize",
        "--speaker-profiles"
    ])

# Now you can match any audio against this comprehensive index
result = subprocess.run([
    "beige-book-match-voice", 
    "new-interview.mp3",
    "speaker-index.db",
    "--format", "json"
], capture_output=True, text=True)

matches = json.loads(result.stdout)
```