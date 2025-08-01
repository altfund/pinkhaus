import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple

from .chunking import TextChunk


class PodcastVectorStore:
    """ChromaDB vector store for podcast transcriptions."""

    def __init__(
        self,
        persist_directory: str = "./grant_chroma_db",
        collection_name: str = "podcast_segments",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Get or create collection
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Get or create the ChromaDB collection."""
        try:
            return self.client.get_collection(self.collection_name)
        except chromadb.errors.NotFoundError:
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Podcast transcription segments"},
            )

    def add_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]):
        """Add chunks with their embeddings to the vector store."""

        if not chunks:
            return

        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Add to collection
        self.collection.add(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
        )

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[TextChunk, float]]:
        """Search for similar chunks."""

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Convert results to chunks with scores
        chunks_with_scores = []

        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                chunk = TextChunk(
                    id=results["ids"][0][i],
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                )
                # Convert distance to similarity score (1 - normalized distance)
                distance = results["distances"][0][i]
                similarity = 1.0 - (distance / 2.0)  # Normalize for cosine distance

                chunks_with_scores.append((chunk, similarity))

        return chunks_with_scores

    def search_by_metadata(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        transcription_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Tuple[TextChunk, float]]:
        """Search with metadata filters."""

        where = {}

        if transcription_id is not None:
            where["transcription_id"] = transcription_id

        # ChromaDB doesn't support complex date queries easily,
        # so we'll filter dates post-retrieval if needed
        results = self.search(query_embedding, n_results * 2, where)

        # Post-filter by dates if needed
        if start_date or end_date:
            filtered_results = []
            for chunk, score in results:
                published = chunk.metadata.get("published")
                if published:
                    if start_date and published < start_date:
                        continue
                    if end_date and published > end_date:
                        continue
                filtered_results.append((chunk, score))
            results = filtered_results[:n_results]

        return results

    def get_chunk(self, chunk_id: str) -> Optional[TextChunk]:
        """Get a specific chunk by ID."""

        results = self.collection.get(
            ids=[chunk_id], include=["documents", "metadatas"]
        )

        if results["ids"]:
            return TextChunk(
                id=results["ids"][0],
                text=results["documents"][0],
                metadata=results["metadatas"][0],
            )

        return None

    def chunk_exists(self, chunk_id: str) -> bool:
        """Check if a chunk exists in the store."""

        results = self.collection.get(ids=[chunk_id])
        return len(results["ids"]) > 0

    def delete_chunks(self, chunk_ids: List[str]):
        """Delete chunks by IDs."""

        if chunk_ids:
            self.collection.delete(ids=chunk_ids)

    def delete_by_transcription(self, transcription_id: int):
        """Delete all chunks for a transcription."""

        self.collection.delete(where={"transcription_id": transcription_id})

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""

        count = self.collection.count()

        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
        }

    def reset(self):
        """Reset the entire collection."""

        self.client.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection()
