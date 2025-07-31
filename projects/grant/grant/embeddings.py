from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from .ollama_client import OllamaClient
from .chunking import TextChunk


class EmbeddingService:
    """Service for generating and caching embeddings using Ollama."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        model: str = "nomic-embed-text",
        cache_dir: Optional[str] = "./.embedding_cache",
    ):
        self.client = ollama_client
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)

    def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate embedding for a single text."""

        # Check cache first
        if use_cache and self.cache_dir:
            cached = self._get_cached_embedding(text)
            if cached is not None:
                return cached

        # Generate embedding
        response = self.client.embeddings(self.model, text)
        embedding = response.embedding

        # Cache the result
        if use_cache and self.cache_dir:
            self._cache_embedding(text, embedding)

        return embedding

    async def embed_text_async(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate embedding for a single text asynchronously."""

        # Check cache first
        if use_cache and self.cache_dir:
            cached = self._get_cached_embedding(text)
            if cached is not None:
                return cached

        # Generate embedding
        response = await self.client.embeddings_async(self.model, text)
        embedding = response.embedding

        # Cache the result
        if use_cache and self.cache_dir:
            self._cache_embedding(text, embedding)

        return embedding

    def embed_texts(
        self, texts: List[str], use_cache: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts."""

        return [self.embed_text(text, use_cache) for text in texts]

    def embed_texts_parallel(
        self, texts: List[str], max_workers: int = 4, use_cache: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in parallel."""

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.embed_text, text, use_cache) for text in texts
            ]
            return [future.result() for future in futures]

    def embed_chunks(
        self, chunks: List[TextChunk], batch_size: int = 10, use_cache: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for multiple chunks."""

        embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_embeddings = [
                self.embed_text(chunk.text, use_cache) for chunk in batch
            ]
            embeddings.extend(batch_embeddings)

        return embeddings

    async def embed_chunks_async(
        self, chunks: List[TextChunk], batch_size: int = 10, use_cache: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for multiple chunks asynchronously."""

        embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Process batch concurrently
            tasks = [self.embed_text_async(chunk.text, use_cache) for chunk in batch]
            batch_embeddings = await asyncio.gather(*tasks)
            embeddings.extend(batch_embeddings)

        return embeddings

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""

        content = f"{self.model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache if it exists."""

        if not self.cache_dir:
            return None

        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    return data["embedding"]
            except:
                # Invalid cache file, ignore
                pass

        return None

    def _cache_embedding(self, text: str, embedding: List[float]):
        """Cache an embedding."""

        if not self.cache_dir:
            return

        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.json"

        data = {"model": self.model, "text_preview": text[:100], "embedding": embedding}

        with open(cache_file, "w") as f:
            json.dump(data, f)

    def clear_cache(self):
        """Clear the embedding cache."""

        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()

    def ensure_model_available(self) -> bool:
        """Check if the embedding model is available."""

        try:
            models = self.client.list_models()
            model_names = [m["name"] for m in models.get("models", [])]

            # Check if our model is in the list (handle different naming)
            for name in model_names:
                if self.model in name or name in self.model:
                    return True

            return False
        except:
            return False

    def pull_model(self):
        """Pull the embedding model if not available."""

        print(f"Pulling embedding model: {self.model}")
        for chunk in self.client.pull_model(self.model):
            if "status" in chunk:
                print(f"\r{chunk['status']}", end="", flush=True)
        print("\nModel pulled successfully!")
