# Betterproto Model Serialization Summary

## Overview

The Beige Book project uses betterproto for protocol buffer definitions, which provides built-in serialization support for JSON and dictionary formats. This document summarizes the current state of serialization support for wire transmission.

## Current Status

### ✅ Supported Serialization Methods

1. **JSON Serialization**
   - `to_json()` - Converts protobuf message to JSON string
   - `from_json(json_str)` - Creates protobuf message from JSON string
   - Works on all betterproto message types

2. **Dictionary Serialization**
   - `to_dict()` - Converts protobuf message to Python dictionary
   - `from_dict(dict)` - Creates protobuf message from dictionary
   - Useful for custom processing and TOML conversion

### 🔍 Key Findings

1. **TranscriptionRequest and TranscriptionResponse have JSON/dict support**
   - Both models inherit from betterproto.Message
   - Automatic serialization/deserialization of nested messages
   - Enum values are serialized as strings (e.g., "INPUT_TYPE_FILE")

2. **REST API Integration**
   - API accepts JSON requests via Pydantic models
   - Converts to internal protobuf models using mapping functions
   - Returns JSON responses using to_dict() on protobuf messages
   - Handles large data appropriately for wire transmission

3. **TOML Support**
   - Not natively supported by betterproto
   - Can be achieved via dict conversion:
     ```python
     import toml
     request_dict = request.to_dict()
     toml_str = toml.dumps(request_dict)
     ```

### ⚠️ Known Issues

1. **Wrapper Class Recursion**
   - The models_betterproto.py has some recursive to_dict() calls that need fixing
   - Direct use of proto_models.py classes works correctly

2. **Datetime Handling**
   - ProcessingError uses google.protobuf.Timestamp
   - Requires special handling for datetime fields
   - The wrapper provides ISO format conversion

### 📊 Wire Format Comparison

| Format | Pros | Cons | Recommended Use |
|--------|------|------|-----------------|
| Protobuf Binary | Compact, fast | Size limits (~few MB) | Small messages |
| JSON | Human-readable, no size limits | Larger size | API communication, large results |
| TOML | Config-friendly | Requires conversion | Configuration files |

## Recommendations

1. **For API Communication**: Use JSON format
   - No size limitations for large transcription results
   - Human-readable for debugging
   - Wide client library support

2. **For Large Data (>few MB)**: Avoid protobuf binary format
   - Use JSON responses from the API
   - Stream large results if needed

3. **Client Integration**:
   ```python
   # Request
   request = create_file_request(...)
   json_request = request.to_json()
   
   # Response handling
   response = TranscriptionResponse().from_json(json_response)
   ```

## Example Usage

### Creating and Serializing a Request
```python
from beige_book.proto_models import TranscriptionRequest, InputConfig, ProcessingConfig, OutputConfig
from beige_book.proto_models import InputConfigInputType, ProcessingConfigModel, OutputConfigFormat

# Create request
request = TranscriptionRequest(
    input=InputConfig(
        type=InputConfigInputType.INPUT_TYPE_FILE,
        source="/path/to/audio.mp3"
    ),
    processing=ProcessingConfig(
        model=ProcessingConfigModel.MODEL_TINY,
        verbose=True
    ),
    output=OutputConfig(
        format=OutputConfigFormat.FORMAT_JSON,
        destination=""
    )
)

# Serialize to JSON
json_str = request.to_json()

# Serialize to dict
req_dict = request.to_dict()
```

### API Response Format
```json
{
  "success": true,
  "results": [{
    "filename": "audio.mp3",
    "file_hash": "abc123",
    "language": "en",
    "segments": [
      {"start_ms": 0, "end_ms": 5000, "text": "Hello world"}
    ],
    "full_text": "Hello world",
    "created_at": 1234567890
  }],
  "errors": [],
  "summary": {
    "total_items": 1,
    "processed": 1,
    "skipped": 0,
    "failed": 0,
    "elapsed_time": 2.5
  }
}
```

## Conclusion

The betterproto models fully support JSON and dictionary serialization for wire transmission. Clients can request JSON format responses to handle large transcription data without protobuf size limitations. The REST API properly handles the conversion between formats, making it suitable for production use.