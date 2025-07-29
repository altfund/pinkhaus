# Speaker Diarization Integration

This document explains how to use pyannote-audio for speaker diarization (identifying "who speaks when") in podcast transcriptions.

## Current Status

Due to dependency conflicts with Python 3.13 (specifically the `sentencepiece` package), the full pyannote-audio integration cannot be installed directly. However, the infrastructure is ready and can be used in two ways:

1. **Mock Mode**: For testing and development (works with Python 3.13)
2. **Full Mode**: With pyannote-audio installed (requires Python 3.11 or resolution of dependency issues)

## What's Been Implemented

### 1. Speaker Diarizer Module (`beige_book/speaker_diarizer.py`)
- `SpeakerDiarizer` class for speaker identification
- Mock diarization for testing without pyannote
- Alignment of speaker segments with transcription segments
- Support for Hugging Face model loading

### 2. Extended Data Models
- Updated protobuf definitions to include speaker information
- Added `speaker` and `confidence` fields to segments
- Added `num_speakers` and `has_speaker_labels` to results

### 3. Enhanced Transcriber
- Added `enable_diarization` parameter to `transcribe_file()`
- Automatic fallback to mock mode if pyannote unavailable
- Integration with existing transcription pipeline

### 4. Updated Output Formats
- JSON: Includes speaker labels and confidence scores
- CSV: Added speaker column when diarization is enabled
- Table: Shows speaker information in formatted output

## Usage

### Basic Usage (Mock Mode)

```python
from beige_book.transcriber import AudioTranscriber

# Initialize transcriber
transcriber = AudioTranscriber(model_name="tiny")

# Transcribe with mock speaker diarization
result = transcriber.transcribe_file(
    "podcast.wav",
    enable_diarization=True
)

# Output will include speaker labels
print(result.to_json())
```

### With Real pyannote-audio (When Available)

```python
# Set your Hugging Face token
import os
os.environ["HF_TOKEN"] = "your-token-here"

# Transcribe with real speaker diarization
result = transcriber.transcribe_file(
    "podcast.wav",
    enable_diarization=True,
    hf_token=os.getenv("HF_TOKEN")
)
```

### Using the Standalone Diarizer

```python
from beige_book.speaker_diarizer import SpeakerDiarizer

# Initialize diarizer
diarizer = SpeakerDiarizer(auth_token="your-hf-token")

# Perform diarization
diarization_result = diarizer.diarize_file("podcast.wav")

# Access speaker segments
for segment in diarization_result.segments:
    print(f"{segment.speaker}: {segment.start:.2f}s - {segment.end:.2f}s")
```

## Installation Requirements

### For Mock Mode (Works Now)
No additional requirements - uses existing dependencies.

### For Full pyannote-audio Support

1. **Option 1**: Use Python 3.11
   ```bash
   # Create Python 3.11 environment
   python3.11 -m venv venv-py311
   source venv-py311/bin/activate
   pip install pyannote.audio speechbrain
   ```

2. **Option 2**: Wait for dependency updates
   - The issue is with `sentencepiece` package not building correctly for Python 3.13
   - Monitor pyannote-audio and speechbrain projects for updates

3. **Option 3**: Use Docker
   - Create a container with Python 3.11 and all dependencies
   - Run the transcription service in the container

## Hugging Face Setup

To use real speaker diarization models:

1. Create account at https://huggingface.co
2. Accept model conditions at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Generate access token at https://huggingface.co/settings/tokens
4. Set token as environment variable: `export HF_TOKEN=your-token-here`

## Example Output

### Without Speaker Diarization
```json
{
  "segments": [
    {
      "start": "00:00:00.000",
      "end": "00:00:05.230",
      "text": "Welcome to our podcast!",
      "duration": 5.23
    }
  ]
}
```

### With Speaker Diarization
```json
{
  "segments": [
    {
      "start": "00:00:00.000",
      "end": "00:00:05.230",
      "text": "Welcome to our podcast!",
      "duration": 5.23,
      "speaker": "SPEAKER_0",
      "confidence": 0.95
    }
  ],
  "num_speakers": 2,
  "has_speaker_labels": true
}
```

## Future Improvements

1. **Speaker Recognition**: Identify recurring speakers across episodes
2. **Speaker Naming**: Allow manual labeling of speakers (Host, Guest, etc.)
3. **Overlap Handling**: Better handling of overlapping speech
4. **Real-time Processing**: Stream-based diarization for live podcasts
5. **Custom Models**: Fine-tune models for specific podcast domains

## Troubleshooting

### "pyannote-audio is not installed" Error
- This is expected with Python 3.13
- The system will automatically fall back to mock diarization
- To use real diarization, see installation options above

### "No HF_TOKEN found" Error
- Set your Hugging Face token: `export HF_TOKEN=your-token-here`
- Or pass it directly: `transcribe_file(..., hf_token="your-token")`

### Performance Issues
- Speaker diarization is computationally intensive
- Use GPU if available: pyannote will automatically detect CUDA
- Consider processing in batches for multiple files