"""
Comprehensive tests for the FastAPI REST API server.

Tests all API endpoints, request validation, response formats, and error handling.
"""

import json
import csv
import io
import toml
import tempfile
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock

from beige_book.api import app
from beige_book.models import TranscriptionResult, ProcessingError
from beige_book.proto_models import Segment


# Helper class for creating mock summaries with proper __dict__
class MockSummary:
    def __init__(self, total_items=0, processed=0, skipped=0, failed=0, elapsed_time=0.0):
        self.total_items = total_items
        self.processed = processed
        self.skipped = skipped
        self.failed = failed
        self.elapsed_time = elapsed_time
    
    def to_dict(self):
        return {
            "total_items": self.total_items,
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": self.failed,
            "elapsed_time": self.elapsed_time
        }


# Test audio file path - assuming the same harvard.wav file used in other tests
HARVARD_WAV = str(
    Path(__file__).parent.parent / ".." / ".." / "resources" / "audio" / "harvard.wav"
)


@pytest.fixture
def client():
    """Create a FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def mock_transcription_result():
    """Create a mock transcription result for testing"""
    segments = [
        Segment(
            start_ms=0,
            end_ms=2000,
            text="The stale smell of old beer lingers."
        ),
        Segment(
            start_ms=2000,
            end_ms=4000,
            text="It takes heat to bring out the odor."
        ),
        Segment(
            start_ms=4000,
            end_ms=6000,
            text="A cold dip restores health and zest."
        )
    ]
    
    return TranscriptionResult(
        filename="test_audio.mp3",
        file_hash="abc123def456",
        language="en",
        segments=segments,
        full_text="The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest.",
        created_at=1234567890  # Unix timestamp
    )


@pytest.fixture
def mock_service_response(mock_transcription_result):
    """Create a mock service response"""
    mock_response = Mock()
    mock_response.success = True
    mock_response.results = [mock_transcription_result]
    mock_response.errors = []
    
    # Create a proper mock for summary with the required attributes
    summary_dict = {
        "total_items": 1,
        "processed": 1,
        "skipped": 0,
        "failed": 0,
        "elapsed_time": 5.2
    }
    mock_response.summary = MockSummary(
        total_items=1,
        processed=1,
        skipped=0,
        failed=0,
        elapsed_time=5.2
    )
    
    return mock_response


class TestBasicEndpoints:
    """Test basic API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test the root endpoint returns API information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "Beige Book Transcription API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
        assert data["endpoints"]["transcribe"] == "/transcribe"
        assert data["endpoints"]["docs"] == "/docs"
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestTranscriptionEndpoint:
    """Test the main /transcribe endpoint"""
    
    @patch('beige_book.api.TranscriptionService')
    def test_single_file_json_output(self, mock_service_class, client, mock_service_response):
        """Test transcribing a single file with JSON output"""
        # Setup mock
        mock_service = Mock()
        mock_service.process.return_value = mock_service_response
        mock_service_class.return_value = mock_service
        
        request_data = {
            "input": {
                "type": "file",
                "source": "/path/to/audio.mp3"
            },
            "processing": {
                "model": "tiny",
                "verbose": False
            },
            "output": {
                "format": "json"
            }
        }
        
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 1
        assert data["results"][0]["filename"] == "test_audio.mp3"
        assert len(data["results"][0]["segments"]) == 3
        assert data["summary"]["processed"] == 1
    
    @patch('beige_book.api.TranscriptionService')
    def test_single_file_text_output(self, mock_service_class, client, mock_service_response):
        """Test transcribing a single file with text output"""
        # Setup mock
        mock_service = Mock()
        mock_service.process.return_value = mock_service_response
        mock_service_class.return_value = mock_service
        
        # Mock the OutputFormatter
        with patch('beige_book.api.OutputFormatter.format_results') as mock_formatter:
            mock_formatter.return_value = "The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest."
            
            request_data = {
                "input": {
                    "type": "file",
                    "source": "/path/to/audio.mp3"
                },
                "processing": {
                    "model": "tiny",
                    "verbose": False
                },
                "output": {
                    "format": "text"
                }
            }
            
            response = client.post("/transcribe", json=request_data)
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/plain; charset=utf-8"
            assert "The stale smell of old beer lingers" in response.text
    
    @patch('beige_book.api.TranscriptionService')
    def test_single_file_csv_output(self, mock_service_class, client, mock_service_response):
        """Test transcribing a single file with CSV output"""
        # Setup mock
        mock_service = Mock()
        mock_service.process.return_value = mock_service_response
        mock_service_class.return_value = mock_service
        
        # Mock the OutputFormatter
        with patch('beige_book.api.OutputFormatter.format_results') as mock_formatter:
            csv_content = """# File: test_audio.mp3
# SHA256: abc123def456
# Language: en
Start,End,Duration,Text
00:00:00.000,00:00:02.000,00:00:02.000,"The stale smell of old beer lingers."
00:00:02.000,00:00:04.000,00:00:02.000,"It takes heat to bring out the odor."
00:00:04.000,00:00:06.000,00:00:02.000,"A cold dip restores health and zest."
"""
            mock_formatter.return_value = csv_content
            
            request_data = {
                "input": {
                    "type": "file",
                    "source": "/path/to/audio.mp3"
                },
                "processing": {
                    "model": "base",
                    "verbose": True
                },
                "output": {
                    "format": "csv"
                }
            }
            
            response = client.post("/transcribe", json=request_data)
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/csv; charset=utf-8"
            
            # Parse CSV to validate
            lines = response.text.strip().split("\n")
            assert lines[0].startswith("# File:")
            assert "Start,End,Duration,Text" in response.text
    
    @patch('beige_book.api.TranscriptionService')
    def test_single_file_toml_output(self, mock_service_class, client, mock_service_response):
        """Test transcribing a single file with TOML output"""
        # Setup mock
        mock_service = Mock()
        mock_service.process.return_value = mock_service_response
        mock_service_class.return_value = mock_service
        
        # Mock the OutputFormatter
        with patch('beige_book.api.OutputFormatter.format_results') as mock_formatter:
            toml_content = """[transcription]
filename = "test_audio.mp3"
file_hash = "abc123def456"
language = "en"

[[segments]]
start = "00:00:00.000"
end = "00:00:02.000"
text = "The stale smell of old beer lingers."
"""
            mock_formatter.return_value = toml_content
            
            request_data = {
                "input": {
                    "type": "file",
                    "source": "/path/to/audio.mp3"
                },
                "processing": {
                    "model": "small",
                    "verbose": False
                },
                "output": {
                    "format": "toml"
                }
            }
            
            response = client.post("/transcribe", json=request_data)
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/toml"
            assert "[transcription]" in response.text
            assert "[[segments]]" in response.text
    
    @patch('beige_book.api.TranscriptionService')
    def test_sqlite_output(self, mock_service_class, client, mock_service_response):
        """Test saving transcription to SQLite database"""
        # Setup mock
        mock_service = Mock()
        mock_service.process.return_value = mock_service_response
        mock_service_class.return_value = mock_service
        
        request_data = {
            "input": {
                "type": "file",
                "source": "/path/to/audio.wav"
            },
            "processing": {
                "model": "medium",
                "verbose": True
            },
            "output": {
                "format": "sqlite",
                "database": {
                    "db_path": "/tmp/test.db",
                    "metadata_table": "transcriptions",
                    "segments_table": "segments"
                }
            }
        }
        
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "Saved to database" in data["message"]
        assert "/tmp/test.db" in data["message"]
        assert data["summary"]["processed"] == 1
    
    @patch('beige_book.api.TranscriptionService')
    def test_feed_processing(self, mock_service_class, client, mock_service_response):
        """Test processing RSS feeds"""
        # Setup mock with multiple results
        mock_response_multi = Mock()
        mock_response_multi.success = True
        mock_response_multi.results = [mock_service_response.results[0]] * 3
        mock_response_multi.errors = []
        mock_response_multi.summary = Mock(
            total_items=3,
            processed=3,
            skipped=0,
            failed=0,
            elapsed_time=15.6
        )
        mock_response_multi.summary.to_dict = lambda: {
            "total_items": 3,
            "processed": 3,
            "skipped": 0,
            "failed": 0,
            "elapsed_time": 15.6
        }
        
        mock_service = Mock()
        mock_service.process.return_value = mock_response_multi
        mock_service_class.return_value = mock_service
        
        request_data = {
            "input": {
                "type": "feed",
                "source": "/path/to/feeds.toml"
            },
            "processing": {
                "model": "tiny",
                "verbose": False,
                "feed_options": {
                    "limit": 10,
                    "order": "newest",
                    "max_retries": 3,
                    "initial_delay": 1.0
                }
            },
            "output": {
                "format": "json"
            }
        }
        
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["results"]) == 3
        assert data["summary"]["total_items"] == 3
        assert data["summary"]["processed"] == 3


class TestValidation:
    """Test request validation and error handling"""
    
    def test_missing_required_fields(self, client):
        """Test that missing required fields return 422"""
        # Missing input field
        request_data = {
            "processing": {"model": "tiny"},
            "output": {"format": "json"}
        }
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 422
    
    def test_invalid_input_type(self, client):
        """Test validation of input type"""
        request_data = {
            "input": {
                "type": "invalid",  # Invalid type
                "source": "/path/to/file"
            },
            "processing": {"model": "tiny"},
            "output": {"format": "json"}
        }
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 422
    
    def test_invalid_model_name(self, client):
        """Test validation of model name"""
        request_data = {
            "input": {"type": "file", "source": "/path/to/file"},
            "processing": {
                "model": "invalid_model"  # Invalid model
            },
            "output": {"format": "json"}
        }
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 422
    
    def test_invalid_output_format(self, client):
        """Test validation of output format"""
        request_data = {
            "input": {"type": "file", "source": "/path/to/file"},
            "processing": {"model": "tiny"},
            "output": {
                "format": "invalid_format"  # Invalid format
            }
        }
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 422
    
    def test_empty_source_validation(self, client):
        """Test that empty source is rejected"""
        request_data = {
            "input": {
                "type": "file",
                "source": ""  # Empty source
            },
            "processing": {"model": "tiny"},
            "output": {"format": "json"}
        }
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 422  # Pydantic validation error
    
    def test_sqlite_without_database_config(self, client):
        """Test that SQLite format requires database configuration"""
        request_data = {
            "input": {"type": "file", "source": "/path/to/file"},
            "processing": {"model": "tiny"},
            "output": {
                "format": "sqlite"
                # Missing database config
            }
        }
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 422
    
    def test_feed_options_with_file_input(self, client):
        """Test that feed options are rejected for file input"""
        request_data = {
            "input": {
                "type": "file",  # File input
                "source": "/path/to/file"
            },
            "processing": {
                "model": "tiny",
                "feed_options": {  # Feed options not allowed
                    "limit": 10,
                    "order": "newest"
                }
            },
            "output": {"format": "json"}
        }
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 422  # Pydantic validation error
        # The validation happens during Pydantic parsing
    
    def test_negative_feed_limit(self, client):
        """Test validation of feed limit"""
        request_data = {
            "input": {"type": "feed", "source": "/path/to/feeds.toml"},
            "processing": {
                "model": "tiny",
                "feed_options": {
                    "limit": -1  # Negative limit
                }
            },
            "output": {"format": "json"}
        }
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 422
    
    def test_negative_retry_delay(self, client):
        """Test validation of retry delay"""
        request_data = {
            "input": {"type": "feed", "source": "/path/to/feeds.toml"},
            "processing": {
                "model": "tiny",
                "feed_options": {
                    "initial_delay": -1.0  # Negative delay
                }
            },
            "output": {"format": "json"}
        }
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 422
    
    def test_empty_database_path(self, client):
        """Test validation of database path"""
        request_data = {
            "input": {"type": "file", "source": "/path/to/file"},
            "processing": {"model": "tiny"},
            "output": {
                "format": "sqlite",
                "database": {
                    "db_path": ""  # Empty path
                }
            }
        }
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 422


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @patch('beige_book.api.TranscriptionService')
    def test_service_validation_error(self, mock_service_class, client):
        """Test handling of service validation errors"""
        # Setup mock to raise ValueError
        mock_service = Mock()
        mock_service.process.side_effect = ValueError("Invalid audio file format")
        mock_service_class.return_value = mock_service
        
        request_data = {
            "input": {"type": "file", "source": "/path/to/file.txt"},
            "processing": {"model": "tiny"},
            "output": {"format": "json"}
        }
        
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 400
        assert "Invalid audio file format" in response.json()["detail"]
    
    @patch('beige_book.api.TranscriptionService')
    def test_service_processing_error(self, mock_service_class, client):
        """Test handling of service processing errors"""
        # Setup mock to return error response
        mock_response = Mock()
        mock_response.success = False
        mock_response.results = []
        mock_response.errors = [
            Mock(
                message="Failed to download audio file",
                source="http://example.com/audio.mp3",
                error_type="DOWNLOAD_ERROR"
            )
        ]
        mock_response.errors[0].to_dict = lambda: {
            "message": "Failed to download audio file",
            "source": "http://example.com/audio.mp3",
            "error_type": "DOWNLOAD_ERROR"
        }
        
        # Add a summary for failed requests too
        mock_response.summary = None  # Or provide a summary with failed count
        
        mock_service = Mock()
        mock_service.process.return_value = mock_response
        mock_service_class.return_value = mock_service
        
        request_data = {
            "input": {"type": "file", "source": "/path/to/file"},
            "processing": {"model": "tiny"},
            "output": {"format": "json"}
        }
        
        response = client.post("/transcribe", json=request_data)
        # When format is JSON and success is False, it still returns 200 with error details in the body
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert len(data["errors"]) == 1
        assert data["errors"][0]["message"] == "Failed to download audio file"
    
    @patch('beige_book.api.TranscriptionService')
    def test_unexpected_error(self, mock_service_class, client):
        """Test handling of unexpected errors"""
        # Setup mock to raise unexpected exception
        mock_service = Mock()
        mock_service.process.side_effect = Exception("Unexpected error occurred")
        mock_service_class.return_value = mock_service
        
        request_data = {
            "input": {"type": "file", "source": "/path/to/file"},
            "processing": {"model": "tiny"},
            "output": {"format": "json"}
        }
        
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]


class TestEdgeCases:
    """Test edge cases and special scenarios"""
    
    def test_feed_with_default_options(self, client):
        """Test that feed input gets default options when not provided"""
        with patch('beige_book.api.TranscriptionService') as mock_service_class:
            mock_service = Mock()
            mock_response = Mock()
            mock_response.success = True
            mock_response.results = []
            mock_response.errors = []
            mock_response.summary = MockSummary()
            mock_service.process.return_value = mock_response
            mock_service_class.return_value = mock_service
            
            request_data = {
                "input": {"type": "feed", "source": "/path/to/feeds.toml"},
                "processing": {
                    "model": "tiny"
                    # No feed_options provided
                },
                "output": {"format": "json"}
            }
            
            response = client.post("/transcribe", json=request_data)
            assert response.status_code == 200
            
            # Verify that the service was called
            mock_service.process.assert_called_once()
            call_args = mock_service.process.call_args[0][0]
            assert call_args.processing.feed_options is not None
    
    def test_large_model_selection(self, client):
        """Test using the large model"""
        with patch('beige_book.api.TranscriptionService') as mock_service_class:
            mock_service = Mock()
            mock_response = Mock()
            mock_response.success = True
            mock_response.results = []
            mock_response.errors = []
            mock_response.summary = MockSummary()
            mock_service.process.return_value = mock_response
            mock_service_class.return_value = mock_service
            
            request_data = {
                "input": {"type": "file", "source": "/path/to/file"},
                "processing": {
                    "model": "large",
                    "verbose": True
                },
                "output": {"format": "json"}
            }
            
            response = client.post("/transcribe", json=request_data)
            assert response.status_code == 200
    
    def test_custom_table_names(self, client):
        """Test using custom database table names"""
        with patch('beige_book.api.TranscriptionService') as mock_service_class:
            mock_service = Mock()
            mock_response = Mock()
            mock_response.success = True
            mock_response.results = []
            mock_response.errors = []
            mock_response.summary = MockSummary()
            mock_service.process.return_value = mock_response
            mock_service_class.return_value = mock_service
            
            request_data = {
                "input": {"type": "file", "source": "/path/to/file"},
                "processing": {"model": "tiny"},
                "output": {
                    "format": "sqlite",
                    "database": {
                        "db_path": "/tmp/custom.db",
                        "metadata_table": "custom_metadata",
                        "segments_table": "custom_segments"
                    }
                }
            }
            
            response = client.post("/transcribe", json=request_data)
            assert response.status_code == 200
            
            # Verify custom table names were passed
            call_args = mock_service.process.call_args[0][0]
            assert call_args.output.database.metadata_table == "custom_metadata"
            assert call_args.output.database.segments_table == "custom_segments"
    
    def test_output_destination_field(self, client):
        """Test that output destination field is handled correctly"""
        with patch('beige_book.api.TranscriptionService') as mock_service_class:
            mock_service = Mock()
            mock_response = Mock()
            mock_response.success = True
            mock_response.results = []
            mock_response.errors = []
            mock_response.summary = MockSummary()
            mock_service.process.return_value = mock_response
            mock_service_class.return_value = mock_service
            
            request_data = {
                "input": {"type": "file", "source": "/path/to/file"},
                "processing": {"model": "tiny"},
                "output": {
                    "format": "json",
                    "destination": "/tmp/output.json"
                }
            }
            
            response = client.post("/transcribe", json=request_data)
            assert response.status_code == 200
            
            # Verify destination was passed
            call_args = mock_service.process.call_args[0][0]
            assert call_args.output.destination == "/tmp/output.json"


class TestJSONEncoding:
    """Test that JSON responses are properly encoded (not double-encoded)"""
    
    @patch('beige_book.api.TranscriptionService')
    def test_json_response_encoding(self, mock_service_class, client, mock_transcription_result):
        """Test that JSON responses are not double-encoded"""
        # Create a result with special characters that would reveal double-encoding
        mock_result = Mock()
        mock_result.filename = "test.mp3"
        mock_result.file_hash = "abc123"
        mock_result.language = "en"
        mock_result.segments = []
        mock_result.full_text = 'Text with "quotes" and special chars: < > & \''
        mock_result.to_dict = lambda: {
            "filename": mock_result.filename,
            "file_hash": mock_result.file_hash,
            "language": mock_result.language,
            "segments": [],
            "full_text": mock_result.full_text
        }
        
        mock_response = Mock()
        mock_response.success = True
        mock_response.results = [mock_result]
        mock_response.errors = []
        mock_response.summary = None
        
        mock_service = Mock()
        mock_service.process.return_value = mock_response
        mock_service_class.return_value = mock_service
        
        request_data = {
            "input": {"type": "file", "source": "/path/to/file"},
            "processing": {"model": "tiny"},
            "output": {"format": "json"}
        }
        
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 200
        
        # Parse JSON to ensure it's valid
        data = response.json()
        
        # Check that special characters are properly handled
        assert data["results"][0]["full_text"] == 'Text with "quotes" and special chars: < > & \''
        
        # Ensure the raw response contains properly escaped quotes (single-encoding)
        raw_text = response.text
        assert '\\"quotes\\"' in raw_text  # Should be escaped once
        assert '\\\\"quotes\\\\"' not in raw_text  # Should not be double-escaped


class TestOpenAPISchema:
    """Test OpenAPI schema and documentation"""
    
    def test_openapi_schema_available(self, client):
        """Test that OpenAPI schema is available"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert schema["info"]["title"] == "Beige Book Transcription API"
        assert schema["info"]["version"] == "1.0.0"
        assert "/transcribe" in schema["paths"]
    
    def test_docs_available(self, client):
        """Test that docs are available"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger-ui" in response.text.lower()
    
    def test_redoc_available(self, client):
        """Test that ReDoc is available"""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "redoc" in response.text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])