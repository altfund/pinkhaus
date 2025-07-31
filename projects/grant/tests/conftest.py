"""
Pytest configuration and shared fixtures for grant tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
from pinkhaus_models import TranscriptionResult, Segment, TranscriptionDatabase


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_transcription_result():
    """Create a mock transcription result for testing."""
    segments = [
        Segment(
            start=0.0,
            end=5.5,
            text="Welcome to the Machine Learning Podcast."
        ),
        Segment(
            start=5.5,
            end=12.3,
            text="Today we're discussing neural networks and deep learning fundamentals."
        ),
        Segment(
            start=12.3,
            end=18.7,
            text="Our guest is Dr. Jane Smith, a leading researcher in the field."
        ),
        Segment(
            start=18.7,
            end=25.0,
            text="Let's dive into the basics of how neural networks learn."
        ),
    ]
    
    return TranscriptionResult(
        filename="test_podcast.mp3",
        file_hash="abc123def456",
        language="en",
        segments=segments,
        full_text=" ".join(seg.text for seg in segments)
    )


@pytest.fixture
def mock_db(temp_dir):
    """Create a mock database for testing."""
    db_path = temp_dir / "test.db"
    db = Mock(spec=TranscriptionDatabase)
    db.db_path = str(db_path)
    return db


@pytest.fixture
def mock_ollama_client():
    """Create a mock Ollama client."""
    client = Mock()
    
    # Mock embeddings response
    embedding_response = Mock()
    embedding_response.embedding = [0.1] * 384  # nomic-embed-text dimension
    client.embeddings.return_value = embedding_response
    
    # Mock generate response
    generate_response = Mock()
    generate_response.response = "Based on the context, neural networks are..."
    client.generate.return_value = generate_response
    
    return client


@pytest.fixture
def sample_segments():
    """Create sample segments for testing chunking."""
    return [
        Segment(start=0.0, end=10.0, text="First segment about introduction."),
        Segment(start=10.0, end=20.0, text="Second segment about main topic."),
        Segment(start=20.0, end=30.0, text="Third segment with more details."),
        Segment(start=30.0, end=40.0, text="Fourth segment concluding thoughts."),
        Segment(start=40.0, end=50.0, text="Fifth segment with final remarks."),
    ]


@pytest.fixture
def mock_chroma_client():
    """Create a mock ChromaDB client."""
    client = MagicMock()
    
    # Mock collection
    collection = MagicMock()
    collection.count.return_value = 0
    collection.add = MagicMock()
    collection.query = MagicMock(return_value={
        'ids': [['chunk_1', 'chunk_2']],
        'documents': [['Neural networks learn...', 'Deep learning is...']],
        'metadatas': [[
            {'transcription_id': '1', 'start_time': 0.0, 'end_time': 10.0},
            {'transcription_id': '1', 'start_time': 10.0, 'end_time': 20.0}
        ]],
        'distances': [[0.1, 0.2]]
    })
    
    client.get_or_create_collection.return_value = collection
    
    return client