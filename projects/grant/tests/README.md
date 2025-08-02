# Grant RAG Tests

This directory contains comprehensive tests for the Grant RAG (Retrieval-Augmented Generation) system.

## Test Coverage

### Core Components
- **Ollama Client** (`test_ollama_client.py`): Tests for the Ollama API client including generation, embeddings, and error handling
- **Chunking** (`test_chunking.py`): Tests for the podcast-aware text chunking system
- **Embeddings** (`test_embeddings.py`): Tests for the embedding service with caching capabilities
- **Vector Store** (`test_vector_store.py`): Tests for the ChromaDB vector storage integration
- **RAG Pipeline** (`test_rag.py`): Integration tests for the complete RAG pipeline

### Running Tests

```bash
# Run all tests
flox activate -- uv run pytest tests/

# Run with verbose output
flox activate -- uv run pytest tests/ -v

# Run a specific test file
flox activate -- uv run pytest tests/test_ollama_client.py

# Run with coverage
flox activate -- uv run pytest tests/ --cov=grant
```

## Test Fixtures

Common test fixtures are defined in `conftest.py`:
- `temp_dir`: Temporary directory for test files
- `mock_transcription_result`: Mock podcast transcription data
- `mock_db`: Mock database for testing
- `mock_ollama_client`: Mock Ollama client
- `sample_segments`: Sample podcast segments
- `mock_chroma_client`: Mock ChromaDB client

## Test Requirements

The tests use:
- `pytest` for the test framework
- `pytest-mock` for enhanced mocking capabilities
- Mock objects from `unittest.mock`

All dependencies are managed through the project's `pyproject.toml` file.