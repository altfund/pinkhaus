import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
import logging

from .chunking import TextChunk

logger = logging.getLogger(__name__)


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
            logger.debug(f"Results keys: {list(results.keys())}")
            logger.debug(f"Number of results returned: {len(results['ids'][0])}")
            if results["metadatas"] and results["metadatas"][0]:
                from datetime import datetime
                logger.debug("All results metadata:")
                for idx, meta in enumerate(results['metadatas'][0]):
                    logger.debug(f"\nResult {idx + 1}:")
                    logger.debug(f"  - Title: {meta.get('title', 'N/A')}")
                    if 'published_timestamp' in meta and meta['published_timestamp']:
                        unix_ts = meta['published_timestamp']
                        iso_date = datetime.fromtimestamp(unix_ts).isoformat()
                        logger.debug(f"  - Published timestamp: {unix_ts} (Unix) -> {iso_date} (ISO8601)")
                    if 'published' in meta:
                        logger.debug(f"  - Published (original) ISO string: {meta['published']}")
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

        where_conditions = []

        if transcription_id is not None:
            where_conditions.append({"transcription_id": transcription_id})
            logger.debug(f"Filtering by transcription_id: {transcription_id}")

        # Convert dates to Unix timestamps for ChromaDB filtering
        if start_date:
            from datetime import datetime
            start_timestamp = int(datetime.fromisoformat(
                start_date.replace('Z', '+00:00')
            ).timestamp())
            where_conditions.append({"published_timestamp": {"$gte": start_timestamp}})
            logger.debug(f"Filtering by start_date: {start_date} (timestamp: {start_timestamp})")

        if end_date:
            from datetime import datetime
            end_timestamp = int(datetime.fromisoformat(
                end_date.replace('Z', '+00:00')
            ).timestamp())
            where_conditions.append({"published_timestamp": {"$lte": end_timestamp}})
            logger.debug(f"Filtering by end_date: {end_date} (timestamp: {end_timestamp})")

        # Build the where clause
        if len(where_conditions) > 1:
            where = {"$and": where_conditions}
        elif len(where_conditions) == 1:
            where = where_conditions[0]
        else:
            where = None

        logger.debug(f"Final where clause for ChromaDB query: {where}")
        logger.debug(f"Requesting {n_results} results")

        # Now we can query directly with the correct n_results!
        results = self.search(query_embedding, n_results, where)
        logger.debug(f"Retrieved {len(results)} results from vector store")

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

    def get_by_metadata(self, where: Dict[str, Any], limit: int = 1) -> Dict[str, Any]:
        """Get chunks by metadata filter without vector search."""

        return self.collection.get(where=where, limit=limit)

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
