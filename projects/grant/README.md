# Grant - RAG Tool for Podcast Transcriptions

Grant is a Retrieval-Augmented Generation (RAG) tool that uses Ollama to answer queries with custom data, specifically designed for podcast transcriptions stored in the beige-book database format.

## Features

- 🎙️ **Podcast-Optimized**: Smart chunking that preserves segment boundaries and timestamps
- 🔍 **Semantic Search**: Find relevant content across all your podcast transcriptions
- 💬 **Natural Language Q&A**: Ask questions and get answers grounded in your podcast data
- 🏠 **Local & Private**: Everything runs locally using Ollama
- ⚡ **Efficient**: Embedding caching and batch processing for fast indexing
- 🎯 **Filtered Search**: Search by date range or specific episodes

## Installation

### Prerequisites

1. **Flox** (recommended): For environment management
   - Provides all dependencies including Ollama
   - See CLAUDE.md for project-specific conventions

2. **Alternative**: If not using Flox, you'll need:
   - **Ollama**: Install from [ollama.ai](https://ollama.ai)
   - **Python 3.13**: Required for the project

### Setup

#### Option 1: Using Flox (Recommended)

1. Activate the flox environment:
   ```bash
   cd grant
   flox activate
   ```
   
   This will automatically:
   - Install all system dependencies (sqlite, ollama, etc.)
   - Set up Python 3.13
   - Create and activate a virtual environment
   - Install all Python dependencies
   - Add pinkhaus-models to your Python path

2. Start the Ollama service:
   ```bash
   # Start just ollama
   flox services start ollama
   
   # Or start all services
   flox services start
   
   # Check service status
   flox services status
   ```

#### Option 2: Manual Setup

1. Clone and install dependencies:
   ```bash
   cd grant
   uv sync
   ```
   
   Note: pinkhaus-models is already included in the dependencies via pyproject.toml

2. Start Ollama:
   ```bash
   ollama serve
   ```

3. Pull required models:
   ```bash
   # Embedding model (required)
   uv run grant pull nomic-embed-text
   
   # LLM for answering questions (choose one)
   uv run grant pull llama3.2
   # or
   uv run grant pull mistral
   ```

## Quick Start

```bash
# 1. Activate environment and install
flox activate

# 2. Start Ollama service
flox services start ollama

# 3. Pull required models
uv run grant pull nomic-embed-text
uv run grant pull llama3.2

# 4. Index your podcast database
uv run grant index --db ../beige-book/protobuf_transcriptions.db

# 5. Ask questions!
uv run grant ask "What are the main topics discussed in these podcasts?"
```

## Usage

### Basic Commands

```bash
# List available models
uv run grant list

# Query a model directly (non-RAG)
uv run grant query llama3.2 "Hello, how are you?"

# Pull a model from Ollama
uv run grant pull llama3.2
```

> **Note**: The CLI commands assume the `grant` command is available. If you see "command not found", use `uv run python -m grant.cli` instead.

### RAG Commands

#### 1. Index Your Podcast Database

```bash
# Index all transcriptions
uv run grant index --db ../beige-book/protobuf_transcriptions.db

# Index a specific transcription
uv run grant index --transcription-id 123

# Custom chunking parameters
uv run grant index --chunk-size 768 --chunk-overlap 256
```

#### 2. Ask Questions

```bash
# Basic question
uv run grant ask "What was discussed about inflation?"

# Show source segments
uv run grant ask "What are the main economic concerns?" --show-sources

# Use a different model
uv run grant ask "Summarize the key points" --model mistral

# Increase context (retrieve more chunks)
uv run grant ask "Explain the Fed's position" --top-k 10
```

#### 3. Filtered Searches

```bash
# Search within date range
uv run grant ask "What happened in Q1?" --after "2024-01-01" --before "2024-03-31"

# Search specific transcription
uv run grant ask "Key takeaways?" --transcription-id 5

# Combine filters
uv run grant ask "Interest rate discussions" --after "2024-01-01" --show-sources
```

#### 4. View Statistics

```bash
uv run grant stats
```

## Architecture

### Components

1. **OllamaClient** (`ollama_client.py`)
   - Interfaces with Ollama API
   - Supports embeddings, generation, and chat

2. **PodcastChunker** (`chunking.py`)
   - Segment-aware text chunking
   - Preserves timestamps and metadata
   - Configurable chunk size and overlap

3. **EmbeddingService** (`embeddings.py`)
   - Generates embeddings using Ollama
   - Caches embeddings for efficiency
   - Batch processing support

4. **PodcastVectorStore** (`vector_store.py`)
   - ChromaDB integration
   - Metadata filtering
   - Similarity search

5. **RAGPipeline** (`rag.py`)
   - Orchestrates the full RAG workflow
   - Handles indexing and querying
   - Context assembly and answer generation

### Database Schema

Grant uses the shared `pinkhaus-models` package which defines:
- `TranscriptionMetadata`: Episode metadata and full text
- `TranscriptionSegment`: Time-stamped segments with text

## Configuration

### Default Settings

- **Embedding Model**: `nomic-embed-text`
- **LLM Model**: `llama3.2`
- **Chunk Size**: 512 tokens
- **Chunk Overlap**: 128 tokens
- **Top-K Retrieval**: 5 chunks
- **Temperature**: 0.7

### File Locations

- **Vector Store**: `./grant_chroma_db/`
  - Persists indexed embeddings between runs
  - Delete this directory to re-index from scratch
- **Embedding Cache**: `./.embedding_cache/`
  - Caches individual text embeddings
  - Safe to delete (will regenerate as needed)
- **Default DB Path**: `../beige-book/protobuf_transcriptions.db`

## Python API

```python
from grant import RAGPipeline, RAGConfig, OllamaClient

# Initialize
client = OllamaClient()
config = RAGConfig(
    embedding_model="nomic-embed-text",
    llm_model="llama3.2",
    chunk_size=512,
    top_k=5
)

# Create pipeline
rag = RAGPipeline(
    db_path="path/to/database.db",
    ollama_client=client,
    config=config
)

# Index
rag.index_all_transcriptions()

# Query
result = rag.query("What are the main topics discussed?")
print(result.answer)

# Access sources
for source in result.sources:
    print(f"From: {source['title']} at {source['timestamp']}")
```

## Advanced Usage

### Custom Embedding Models

```bash
# Use a different embedding model
uv run grant index --embedding-model mxbai-embed-large
uv run grant ask "question" --embedding-model mxbai-embed-large
```

### Async Operations

The library supports async operations for better performance:

```python
import asyncio
from grant import RAGPipeline, OllamaClient

async def main():
    client = OllamaClient()
    rag = RAGPipeline("database.db", client)
    
    result = await rag.query_async("Your question here")
    print(result.answer)

asyncio.run(main())
```

## Troubleshooting

### "Model not found" Error
```bash
# Check available models
uv run grant list

# Pull the required model
uv run grant pull <model-name>
```

### Slow Indexing
- Reduce batch size: `--batch-size 5`
- Check if embeddings are cached in `./.embedding_cache/`

### Out of Memory
- Reduce chunk size: `--chunk-size 256`
- Process fewer transcriptions at once

## Development

### Running Tests
```bash
# Run all tests
flox activate -- uv run pytest tests/

# Run with verbose output
flox activate -- uv run pytest tests/ -v

# Run specific test module
flox activate -- uv run pytest tests/test_rag.py
```

### Code Style
```bash
uv run ruff check .
uv run ruff format .
```

### Project Structure
```
grant/
├── grant/
│   ├── __init__.py
│   ├── cli.py              # CLI commands and interface
│   ├── ollama_client.py    # Ollama API client
│   ├── chunking.py         # Text chunking with segment awareness
│   ├── embeddings.py       # Embedding generation and caching
│   ├── vector_store.py     # ChromaDB vector storage
│   └── rag.py              # Main RAG pipeline
├── tests/                  # Comprehensive test suite
├── pyproject.toml         # Project dependencies
└── README.md              # This file

## License

[Add your license here]

## Contributing

[Add contribution guidelines]
