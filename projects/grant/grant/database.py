import sqlite3
import aiosqlite
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np


@dataclass
class PodcastTranscription:
    id: int
    filename: str
    file_hash: str
    language: str
    full_text: str
    model_name: Optional[str]
    feed_url: Optional[str]
    feed_item_id: Optional[str]
    feed_item_title: Optional[str]
    feed_item_published: Optional[datetime]
    created_at: datetime


@dataclass
class TranscriptionSegment:
    id: int
    transcription_id: int
    segment_index: int
    start_time: float
    end_time: float
    duration: float
    text: str


@dataclass
class TextChunk:
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class PodcastDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    async def get_async_connection(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        return conn
    
    def get_all_transcriptions(self) -> List[PodcastTranscription]:
        conn = self.get_connection()
        cursor = conn.execute("""
            SELECT * FROM transcription_metadata
            ORDER BY feed_item_published DESC
        """)
        
        transcriptions = []
        for row in cursor:
            transcriptions.append(self._row_to_transcription(row))
        
        conn.close()
        return transcriptions
    
    async def get_all_transcriptions_async(self) -> List[PodcastTranscription]:
        conn = await self.get_async_connection()
        cursor = await conn.execute("""
            SELECT * FROM transcription_metadata
            ORDER BY feed_item_published DESC
        """)
        
        transcriptions = []
        async for row in cursor:
            transcriptions.append(self._row_to_transcription(row))
        
        await conn.close()
        return transcriptions
    
    def get_segments_for_transcription(self, transcription_id: int) -> List[TranscriptionSegment]:
        conn = self.get_connection()
        cursor = conn.execute("""
            SELECT * FROM transcription_segments
            WHERE transcription_id = ?
            ORDER BY segment_index
        """, (transcription_id,))
        
        segments = []
        for row in cursor:
            segments.append(self._row_to_segment(row))
        
        conn.close()
        return segments
    
    def _row_to_transcription(self, row: sqlite3.Row) -> PodcastTranscription:
        return PodcastTranscription(
            id=row["id"],
            filename=row["filename"],
            file_hash=row["file_hash"],
            language=row["language"],
            full_text=row["full_text"],
            model_name=row["model_name"],
            feed_url=row["feed_url"],
            feed_item_id=row["feed_item_id"],
            feed_item_title=row["feed_item_title"],
            feed_item_published=datetime.fromisoformat(row["feed_item_published"]) if row["feed_item_published"] else None,
            created_at=datetime.fromisoformat(row["created_at"])
        )
    
    def _row_to_segment(self, row: sqlite3.Row) -> TranscriptionSegment:
        return TranscriptionSegment(
            id=row["id"],
            transcription_id=row["transcription_id"],
            segment_index=row["segment_index"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            duration=row["duration"],
            text=row["text"]
        )


class VectorStore:
    def __init__(self, db_path: str = "grant_vectors.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS text_chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_created 
            ON text_chunks(created_at)
        """)
        conn.commit()
        conn.close()
    
    def add_chunk(self, chunk: TextChunk):
        conn = sqlite3.connect(self.db_path)
        embedding_blob = chunk.embedding.tobytes() if chunk.embedding is not None else None
        
        conn.execute("""
            INSERT OR REPLACE INTO text_chunks (id, text, metadata, embedding)
            VALUES (?, ?, ?, ?)
        """, (chunk.id, chunk.text, json.dumps(chunk.metadata), embedding_blob))
        
        conn.commit()
        conn.close()
    
    def add_chunks(self, chunks: List[TextChunk]):
        conn = sqlite3.connect(self.db_path)
        
        data = [
            (
                chunk.id,
                chunk.text,
                json.dumps(chunk.metadata),
                chunk.embedding.tobytes() if chunk.embedding is not None else None
            )
            for chunk in chunks
        ]
        
        conn.executemany("""
            INSERT OR REPLACE INTO text_chunks (id, text, metadata, embedding)
            VALUES (?, ?, ?, ?)
        """, data)
        
        conn.commit()
        conn.close()
    
    def get_all_chunks(self, with_embeddings: bool = True) -> List[TextChunk]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT * FROM text_chunks")
        
        chunks = []
        for row in cursor:
            embedding = None
            if with_embeddings and row[3] is not None:
                embedding = np.frombuffer(row[3], dtype=np.float32)
            
            chunks.append(TextChunk(
                id=row[0],
                text=row[1],
                metadata=json.loads(row[2]),
                embedding=embedding
            ))
        
        conn.close()
        return chunks
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[TextChunk, float]]:
        chunks = self.get_all_chunks(with_embeddings=True)
        
        if not chunks or chunks[0].embedding is None:
            return []
        
        # Calculate cosine similarity
        similarities = []
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        for chunk in chunks:
            if chunk.embedding is not None:
                chunk_norm = chunk.embedding / np.linalg.norm(chunk.embedding)
                similarity = np.dot(query_norm, chunk_norm)
                similarities.append((chunk, float(similarity)))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def chunk_exists(self, chunk_id: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT 1 FROM text_chunks WHERE id = ?", (chunk_id,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists