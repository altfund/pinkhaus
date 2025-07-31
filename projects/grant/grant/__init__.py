from .ollama_client import OllamaClient, OllamaMessage, OllamaResponse
from .rag import RAGPipeline, RAGConfig, RAGResult
from .chunking import PodcastChunker, TextChunk
from .embeddings import EmbeddingService
from .vector_store import PodcastVectorStore

__version__ = "0.1.0"
__all__ = [
    "OllamaClient",
    "OllamaMessage",
    "OllamaResponse",
    "RAGPipeline",
    "RAGConfig",
    "RAGResult",
    "PodcastChunker",
    "TextChunk",
    "EmbeddingService",
    "PodcastVectorStore",
]
