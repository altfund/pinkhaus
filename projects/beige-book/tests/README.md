# Tests Directory

This directory contains test scripts for the beige-book transcription tool.

## Available Tests

### test_harvard_diarization.py
Comprehensive test using the harvard.wav audio file that demonstrates:
- Audio transcription with Whisper
- Speaker diarization (real or mock)
- SQLite database creation with enhanced schema
- Multiple output formats (JSON, CSV)
- Speaker statistics and analysis

**Usage:**
```bash
# From project root
python tests/test_harvard_diarization.py
```

### test_simple.py
Basic unit tests for the transcription functionality.

### test_transcriber.py
Unit tests for the AudioTranscriber class.

### test_database.py
Tests for database operations and schema.

## Running All Tests

```bash
# From project root
pytest
```