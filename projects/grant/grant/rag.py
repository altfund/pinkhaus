from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from pinkhaus_models import TranscriptionDatabase, TranscriptionMetadata

from .ollama_client import OllamaClient
from .chunking import PodcastChunker, TextChunk
from .embeddings import EmbeddingService
from .vector_store import PodcastVectorStore


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""

    embedding_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2"
    chunk_size: int = 512
    chunk_overlap: int = 128
    top_k: int = 5
    temperature: float = 0.7


@dataclass
class RAGResult:
    """Result from RAG query."""

    query: str
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class RAGPipeline:
    """Main RAG pipeline for podcast transcriptions."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        config: RAGConfig = RAGConfig(),
        vector_store_path: str = "./grant_chroma_db",
        db_path: Optional[str] = None,
    ):
        self.config = config
        self.db = TranscriptionDatabase(db_path) if db_path else None
        self.ollama = ollama_client

        # Initialize components
        self.chunker = PodcastChunker(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
        self.embeddings = EmbeddingService(
            ollama_client=ollama_client, model=config.embedding_model
        )
        self.vector_store = PodcastVectorStore(persist_directory=vector_store_path)

    def index_all_transcriptions(self, batch_size: int = 10):
        """Index all transcriptions in the database."""
        if not self.db:
            raise ValueError("Database path is required for indexing operations")

        transcriptions = self.db.get_all_transcriptions()
        total = len(transcriptions)

        print(f"Indexing {total} transcriptions...")

        for i, metadata in enumerate(transcriptions):
            print(
                f"\rProcessing {i + 1}/{total}: {metadata.feed_item_title or metadata.filename}",
                end="",
                flush=True,
            )
            self._index_transcription(metadata)

        print("\nIndexing complete!")

        stats = self.vector_store.get_collection_stats()
        print(f"Total chunks indexed: {stats['total_chunks']}")

    def index_transcription(self, transcription_id: int):
        """Index a specific transcription."""
        if not self.db:
            raise ValueError("Database path is required for indexing operations")

        metadata = self.db.get_transcription_metadata(transcription_id)
        if not metadata:
            raise ValueError(f"Transcription {transcription_id} not found")

        self._index_transcription(metadata)

    def _index_transcription(self, metadata: TranscriptionMetadata):
        """Internal method to index a transcription."""

        # Skip if already indexed
        if metadata.id and self._is_transcription_indexed(metadata.id):
            return

        if not self.db:
            raise ValueError("Database path is required for indexing operations")

        # Get segments
        segments = self.db.get_segments_for_transcription(metadata.id)

        # Create chunks
        chunks = self.chunker.chunk_transcription(metadata, segments)

        if not chunks:
            return

        # Generate embeddings
        embeddings = self.embeddings.embed_chunks(chunks)

        # Store in vector database
        self.vector_store.add_chunks(chunks, embeddings)

    def _is_transcription_indexed(self, transcription_id: int) -> bool:
        """Check if a transcription is already indexed."""

        # Direct metadata lookup - no vector search needed
        results = self.vector_store.get_by_metadata(
            where={"transcription_id": transcription_id}, limit=1
        )
        return len(results.get("ids", [])) > 0

    def query(
        self,
        query: str,
        n_results: Optional[int] = None,
        transcription_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> RAGResult:
        """Query the RAG system."""

        n_results = n_results or self.config.top_k

        # Generate query embedding
        query_embedding = self.embeddings.embed_text(query)

        # Search for relevant chunks
        if transcription_id or start_date or end_date:
            results = self.vector_store.search_by_metadata(
                query_embedding,
                n_results=n_results,
                transcription_id=transcription_id,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            results = self.vector_store.search(query_embedding, n_results)

        # Build context from results
        context = self._build_context(results)

        # Generate answer
        answer = self._generate_answer(query, context)

        # Prepare sources
        sources = self._prepare_sources(results)

        return RAGResult(
            query=query,
            answer=answer,
            sources=sources,
            metadata={
                "n_results": len(results),
                "model": self.config.llm_model,
                "embedding_model": self.config.embedding_model,
            },
        )

    async def query_async(
        self,
        query: str,
        n_results: Optional[int] = None,
        transcription_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> RAGResult:
        """Query the RAG system asynchronously."""

        n_results = n_results or self.config.top_k

        # Generate query embedding
        query_embedding = await self.embeddings.embed_text_async(query)

        # Search for relevant chunks (sync for now, ChromaDB doesn't have async)
        if transcription_id or start_date or end_date:
            results = self.vector_store.search_by_metadata(
                query_embedding,
                n_results=n_results,
                transcription_id=transcription_id,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            results = self.vector_store.search(query_embedding, n_results)

        # Build context from results
        context = self._build_context(results)

        # Generate answer
        answer = await self._generate_answer_async(query, context)

        # Prepare sources
        sources = self._prepare_sources(results)

        return RAGResult(
            query=query,
            answer=answer,
            sources=sources,
            metadata={
                "n_results": len(results),
                "model": self.config.llm_model,
                "embedding_model": self.config.embedding_model,
            },
        )

    def _build_context(self, results: List[Tuple[TextChunk, float]]) -> str:
        """Build context from search results."""

        if not results:
            return ""

        context_parts = []

        for chunk, score in results:
            # Add metadata context
            title = chunk.metadata.get("title", "Unknown")
            timestamp = ""

            if "start_time" in chunk.metadata and "end_time" in chunk.metadata:
                start = chunk.metadata["start_time"]
                end = chunk.metadata["end_time"]
                timestamp = f" [{self._format_time(start)} - {self._format_time(end)}]"

            context_parts.append(
                f"From: {title}{timestamp}\n"
                f"Relevance: {score:.2f}\n"
                f"Content: {chunk.text}\n"
            )

        return "\n---\n".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM."""

        if not context:
            return "I couldn't find any relevant information in the podcast transcriptions to answer your question."

        prompt = f"""You are a helpful assistant answering questions based on podcast transcriptions.

Context from podcast transcriptions:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, say so. Include relevant quotes when appropriate."""

        # Generate response
        response_text = ""
        for chunk in self.ollama.generate(
            self.config.llm_model,
            prompt,
            stream=True,
            options={"temperature": self.config.temperature},
        ):
            if "response" in chunk:
                response_text += chunk["response"]

        return response_text.strip()

    async def _generate_answer_async(self, query: str, context: str) -> str:
        """Generate answer using LLM asynchronously."""

        if not context:
            return "I couldn't find any relevant information in the podcast transcriptions to answer your question."

        prompt = f"""You are a helpful assistant answering questions based on podcast transcriptions.

Context from podcast transcriptions:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, say so. Include relevant quotes when appropriate."""

        # Generate response
        response_text = ""
        async for chunk in self.ollama.generate_async(
            self.config.llm_model,
            prompt,
            stream=True,
            options={"temperature": self.config.temperature},
        ):
            if "response" in chunk:
                response_text += chunk["response"]

        return response_text.strip()

    def _prepare_sources(
        self, results: List[Tuple[TextChunk, float]]
    ) -> List[Dict[str, Any]]:
        """Prepare source information from results."""

        sources = []

        for chunk, score in results:
            source = {
                "title": chunk.metadata.get("title", "Unknown"),
                "transcription_id": chunk.metadata.get("transcription_id"),
                "score": round(score, 3),
                "text_preview": chunk.text[:200] + "..."
                if len(chunk.text) > 200
                else chunk.text,
            }

            # Add timing info if available
            if "start_time" in chunk.metadata:
                source["start_time"] = chunk.metadata["start_time"]
                source["end_time"] = chunk.metadata.get("end_time")
                source["timestamp"] = self._format_time(chunk.metadata["start_time"])

            # Add publication date if available
            if "published" in chunk.metadata:
                source["published"] = chunk.metadata["published"]

            sources.append(source)

        return sources

    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS."""

        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""

        vector_stats = self.vector_store.get_collection_stats()

        stats = {
            "total_chunks": vector_stats["total_chunks"],
            "embedding_model": self.config.embedding_model,
            "llm_model": self.config.llm_model,
            "chunk_size": self.config.chunk_size,
        }

        # Count indexed transcriptions if database is available
        if self.db:
            transcriptions = self.db.get_all_transcriptions()
            indexed_count = sum(
                1 for t in transcriptions if t.id and self._is_transcription_indexed(t.id)
            )
            stats["total_transcriptions"] = len(transcriptions)
            stats["indexed_transcriptions"] = indexed_count
        else:
            stats["total_transcriptions"] = "N/A (no database)"
            stats["indexed_transcriptions"] = "N/A (no database)"

        return stats
