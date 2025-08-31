# Demos Directory

This directory contains demonstration scripts showing how to use the beige-book features.

## Available Demos

### demo_diarization.py
Interactive demo showing speaker diarization capabilities:
- Mock diarization (no HF token required)
- Real diarization with pyannote (requires HF token)
- Multiple output format examples
- Speaker transition demonstrations

**Usage:**
```bash
# From project root
python demos/demo_diarization.py
```

## See Also

The `examples/` directory contains additional usage examples:
- `speaker_diarization_example.py` - Comprehensive diarization examples
- `library_usage.py` - Basic library usage
- `database_usage.py` - Database operations
- `protobuf_usage.py` - Protocol buffer format
- `api_to_database.py` - REST API with database