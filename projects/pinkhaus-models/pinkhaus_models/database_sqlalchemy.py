"""SQLAlchemy-based database operations for transcriptions."""

from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager, asynccontextmanager
from pathlib import Path
import json
import numpy as np

from sqlalchemy import create_engine, select, and_, or_, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.engine import Engine

from .orm import Base, TranscriptionMetadataORM, TranscriptionSegmentORM, TextChunkORM
from .models import (
    TranscriptionMetadata,
    TranscriptionSegment,
    TranscriptionResult,
    Segment,
)


class DatabaseEngine:
    """Manages SQLAlchemy engine and session creation."""
    
    def __init__(self, database_url: str):
        """
        Initialize database engine.
        
        Args:
            database_url: SQLAlchemy database URL (e.g., 'sqlite:///path/to/db.db')
        """
        # Ensure parent directory exists for SQLite
        if database_url.startswith('sqlite:///'):
            db_path = database_url.replace('sqlite:///', '')
            parent = Path(db_path).parent
            if not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
        
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # For async operations
        if database_url.startswith('sqlite:'):
            async_url = database_url.replace('sqlite:', 'sqlite+aiosqlite:')
        else:
            async_url = database_url
        
        self.async_engine = create_async_engine(async_url)
        self.AsyncSessionLocal = async_sessionmaker(bind=self.async_engine, class_=AsyncSession)
    
    def create_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)
    
    @contextmanager
    def get_session(self) -> Session:
        """Get a database session."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncSession:
        """Get an async database session."""
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


class TranscriptionDatabase:
    """Handle database operations for transcriptions using SQLAlchemy."""
    
    def __init__(self, database_url: str = None, db_path: str = None):
        """
        Initialize database connection.
        
        Args:
            database_url: SQLAlchemy database URL
            db_path: Legacy parameter for SQLite path (deprecated, use database_url)
        """
        if database_url is None and db_path is not None:
            # Convert legacy db_path to SQLAlchemy URL
            database_url = f"sqlite:///{db_path}"
        elif database_url is None:
            raise ValueError("Either database_url or db_path must be provided")
        
        self.engine = DatabaseEngine(database_url)
        self.engine.create_tables()
    
    def create_tables(self, metadata_table: str = None, segments_table: str = None):
        """Create the database tables if they don't exist."""
        # Tables are created automatically in __init__
        pass
    
    def get_all_transcriptions(self, metadata_table: str = None) -> List[TranscriptionMetadata]:
        """Get all transcriptions from the database."""
        with self.engine.get_session() as session:
            stmt = select(TranscriptionMetadataORM).order_by(
                desc(TranscriptionMetadataORM.feed_item_published),
                desc(TranscriptionMetadataORM.created_at)
            )
            results = session.execute(stmt).scalars().all()
            
            return [self._orm_to_metadata(row) for row in results]
    
    async def get_all_transcriptions_async(self, metadata_table: str = None) -> List[TranscriptionMetadata]:
        """Get all transcriptions from the database asynchronously."""
        async with self.engine.get_async_session() as session:
            stmt = select(TranscriptionMetadataORM).order_by(
                desc(TranscriptionMetadataORM.feed_item_published),
                desc(TranscriptionMetadataORM.created_at)
            )
            results = await session.execute(stmt)
            rows = results.scalars().all()
            
            return [self._orm_to_metadata(row) for row in rows]
    
    def get_transcription_metadata(
        self, transcription_id: int, metadata_table: str = None
    ) -> Optional[TranscriptionMetadata]:
        """Get metadata for a specific transcription."""
        with self.engine.get_session() as session:
            result = session.get(TranscriptionMetadataORM, transcription_id)
            return self._orm_to_metadata(result) if result else None
    
    def get_segments_for_transcription(
        self, transcription_id: int, segments_table: str = None
    ) -> List[TranscriptionSegment]:
        """Get all segments for a transcription."""
        with self.engine.get_session() as session:
            stmt = select(TranscriptionSegmentORM).where(
                TranscriptionSegmentORM.transcription_id == transcription_id
            ).order_by(TranscriptionSegmentORM.segment_index)
            
            results = session.execute(stmt).scalars().all()
            return [self._orm_to_segment(row) for row in results]
    
    async def get_segments_for_transcription_async(
        self, transcription_id: int, segments_table: str = None
    ) -> List[TranscriptionSegment]:
        """Get all segments for a transcription asynchronously."""
        async with self.engine.get_async_session() as session:
            stmt = select(TranscriptionSegmentORM).where(
                TranscriptionSegmentORM.transcription_id == transcription_id
            ).order_by(TranscriptionSegmentORM.segment_index)
            
            results = await session.execute(stmt)
            rows = results.scalars().all()
            return [self._orm_to_segment(row) for row in rows]
    
    def get_transcription(
        self, transcription_id: int, metadata_table: str = None, segments_table: str = None
    ) -> Optional[Dict[str, Any]]:
        """Get complete transcription with metadata and segments."""
        with self.engine.get_session() as session:
            metadata = session.get(TranscriptionMetadataORM, transcription_id)
            if not metadata:
                return None
            
            segments = session.query(TranscriptionSegmentORM).filter_by(
                transcription_id=transcription_id
            ).order_by(TranscriptionSegmentORM.segment_index).all()
            
            return {
                "metadata": self._orm_to_dict(metadata),
                "segments": [self._orm_to_dict(seg) for seg in segments],
            }
    
    def save_transcription(
        self,
        result: TranscriptionResult,
        model_name: str = "unknown",
        metadata_table: str = None,
        segments_table: str = None,
        feed_url: Optional[str] = None,
        feed_item_id: Optional[str] = None,
        feed_item_title: Optional[str] = None,
        feed_item_published: Optional[str] = None,
    ) -> int:
        """Save a transcription result to the database."""
        with self.engine.get_session() as session:
            # Check if already exists
            existing = session.query(TranscriptionMetadataORM).filter_by(
                file_hash=result.file_hash,
                model_name=model_name
            ).first()
            
            if existing:
                return existing.id
            
            # Parse datetime string if provided
            feed_published_dt = None
            if feed_item_published:
                from datetime import datetime
                if isinstance(feed_item_published, str):
                    feed_published_dt = datetime.fromisoformat(feed_item_published)
                else:
                    feed_published_dt = feed_item_published
            
            # Create metadata
            metadata = TranscriptionMetadataORM(
                filename=result.filename,
                file_hash=result.file_hash,
                language=result.language,
                full_text=result.full_text,
                model_name=model_name,
                feed_url=feed_url,
                feed_item_id=feed_item_id,
                feed_item_title=feed_item_title,
                feed_item_published=feed_published_dt,
            )
            
            session.add(metadata)
            session.flush()  # Get the ID
            
            # Create segments
            for idx, segment in enumerate(result.segments):
                # Handle both regular segments and protobuf segments
                if hasattr(segment, "start_ms"):
                    # Protobuf segment (beige-book style)
                    start_time = segment.start_ms / 1000.0
                    end_time = segment.end_ms / 1000.0
                    text = segment.text
                else:
                    # Regular segment
                    start_time = segment.start
                    end_time = segment.end
                    text = segment.text
                
                duration = end_time - start_time
                
                seg_orm = TranscriptionSegmentORM(
                    transcription_id=metadata.id,
                    segment_index=idx,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    text=text.strip(),
                )
                session.add(seg_orm)
            
            session.commit()
            return metadata.id
    
    def search_transcriptions(
        self, query: str, metadata_table: str = None, limit: int = 10
    ) -> List[TranscriptionMetadata]:
        """Search transcriptions by text content."""
        with self.engine.get_session() as session:
            stmt = select(TranscriptionMetadataORM).where(
                TranscriptionMetadataORM.full_text.like(f"%{query}%")
            ).order_by(
                desc(TranscriptionMetadataORM.feed_item_published),
                desc(TranscriptionMetadataORM.created_at)
            ).limit(limit)
            
            results = session.execute(stmt).scalars().all()
            return [self._orm_to_metadata(row) for row in results]
    
    def find_by_hash(self, file_hash: str, metadata_table: str = None) -> List[Dict[str, Any]]:
        """Find all transcriptions for a given file hash."""
        with self.engine.get_session() as session:
            stmt = select(TranscriptionMetadataORM).where(
                TranscriptionMetadataORM.file_hash == file_hash
            ).order_by(desc(TranscriptionMetadataORM.created_at))
            
            results = session.execute(stmt).scalars().all()
            return [self._orm_to_dict(row) for row in results]
    
    def delete_transcription(
        self, transcription_id: int, metadata_table: str = None, segments_table: str = None
    ) -> bool:
        """Delete a transcription and its segments."""
        with self.engine.get_session() as session:
            metadata = session.get(TranscriptionMetadataORM, transcription_id)
            if metadata:
                session.delete(metadata)  # Cascade will delete segments
                session.commit()
                return True
            return False
    
    def export_to_dict(
        self, transcription_id: int, metadata_table: str = None, segments_table: str = None
    ) -> Optional[TranscriptionResult]:
        """Export a transcription from database back to TranscriptionResult object."""
        data = self.get_transcription(transcription_id)
        if not data:
            return None
        
        metadata = data["metadata"]
        segments = []
        
        for seg in data["segments"]:
            segments.append(
                Segment(
                    start=seg["start_time"],
                    end=seg["end_time"],
                    text=seg["text"],
                )
            )
        
        return TranscriptionResult(
            filename=metadata["filename"],
            file_hash=metadata["file_hash"],
            language=metadata["language"],
            segments=segments,
            full_text=metadata["full_text"],
        )
    
    def check_feed_item_exists(
        self, feed_url: str, feed_item_id: str, metadata_table: str = None
    ) -> bool:
        """Check if a feed item has already been processed."""
        with self.engine.get_session() as session:
            exists = session.query(TranscriptionMetadataORM).filter_by(
                feed_url=feed_url,
                feed_item_id=feed_item_id
            ).first() is not None
            
            return exists
    
    def get_transcriptions_by_date_range(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        metadata_table: str = None,
    ) -> List[TranscriptionMetadata]:
        """Get transcriptions within a date range."""
        with self.engine.get_session() as session:
            query = session.query(TranscriptionMetadataORM)
            
            if start_date:
                query = query.filter(TranscriptionMetadataORM.feed_item_published >= start_date)
            
            if end_date:
                query = query.filter(TranscriptionMetadataORM.feed_item_published <= end_date)
            
            query = query.order_by(
                desc(TranscriptionMetadataORM.feed_item_published),
                desc(TranscriptionMetadataORM.created_at)
            )
            
            results = query.all()
            return [self._orm_to_metadata(row) for row in results]
    
    # Helper methods for ORM conversion
    def _orm_to_metadata(self, orm_obj: TranscriptionMetadataORM) -> TranscriptionMetadata:
        """Convert ORM object to TranscriptionMetadata."""
        return TranscriptionMetadata(
            id=orm_obj.id,
            filename=orm_obj.filename,
            file_hash=orm_obj.file_hash,
            language=orm_obj.language,
            full_text=orm_obj.full_text,
            model_name=orm_obj.model_name,
            feed_url=orm_obj.feed_url,
            feed_item_id=orm_obj.feed_item_id,
            feed_item_title=orm_obj.feed_item_title,
            feed_item_published=orm_obj.feed_item_published,
            created_at=orm_obj.created_at,
        )
    
    def _orm_to_segment(self, orm_obj: TranscriptionSegmentORM) -> TranscriptionSegment:
        """Convert ORM object to TranscriptionSegment."""
        return TranscriptionSegment(
            id=orm_obj.id,
            transcription_id=orm_obj.transcription_id,
            segment_index=orm_obj.segment_index,
            start_time=orm_obj.start_time,
            end_time=orm_obj.end_time,
            duration=orm_obj.duration,
            text=orm_obj.text,
        )
    
    def _orm_to_dict(self, orm_obj) -> Dict[str, Any]:
        """Convert ORM object to dictionary."""
        if isinstance(orm_obj, TranscriptionMetadataORM):
            return {
                "id": orm_obj.id,
                "filename": orm_obj.filename,
                "file_hash": orm_obj.file_hash,
                "language": orm_obj.language,
                "full_text": orm_obj.full_text,
                "model_name": orm_obj.model_name,
                "feed_url": orm_obj.feed_url,
                "feed_item_id": orm_obj.feed_item_id,
                "feed_item_title": orm_obj.feed_item_title,
                "feed_item_published": orm_obj.feed_item_published.isoformat()
                if orm_obj.feed_item_published else None,
                "created_at": orm_obj.created_at.isoformat()
                if orm_obj.created_at else None,
            }
        elif isinstance(orm_obj, TranscriptionSegmentORM):
            return {
                "id": orm_obj.id,
                "transcription_id": orm_obj.transcription_id,
                "segment_index": orm_obj.segment_index,
                "start_time": orm_obj.start_time,
                "end_time": orm_obj.end_time,
                "duration": orm_obj.duration,
                "text": orm_obj.text,
            }
        else:
            raise ValueError(f"Unknown ORM type: {type(orm_obj)}")


class VectorStore:
    """Vector store for text chunks with embeddings using SQLAlchemy."""
    
    def __init__(self, database_url: str = None, db_path: str = None):
        """
        Initialize vector store.
        
        Args:
            database_url: SQLAlchemy database URL
            db_path: Legacy parameter for SQLite path (deprecated)
        """
        if database_url is None and db_path is not None:
            database_url = f"sqlite:///{db_path}"
        elif database_url is None:
            database_url = "sqlite:///grant_vectors.db"
        
        self.engine = DatabaseEngine(database_url)
        self.engine.create_tables()
    
    def add_chunk(self, chunk: Dict[str, Any]):
        """Add a text chunk to the store."""
        with self.engine.get_session() as session:
            embedding_blob = None
            if chunk.get("embedding") is not None:
                embedding_blob = chunk["embedding"].tobytes()
            
            chunk_orm = TextChunkORM(
                id=chunk["id"],
                text=chunk["text"],
                metadata_json=chunk["metadata"],
                embedding=embedding_blob,
            )
            
            session.merge(chunk_orm)  # Use merge to handle upsert
            session.commit()
    
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Add multiple text chunks to the store."""
        with self.engine.get_session() as session:
            for chunk in chunks:
                embedding_blob = None
                if chunk.get("embedding") is not None:
                    embedding_blob = chunk["embedding"].tobytes()
                
                chunk_orm = TextChunkORM(
                    id=chunk["id"],
                    text=chunk["text"],
                    metadata_json=chunk["metadata"],
                    embedding=embedding_blob,
                )
                
                session.merge(chunk_orm)
            
            session.commit()
    
    def get_all_chunks(self, with_embeddings: bool = True) -> List[Dict[str, Any]]:
        """Get all chunks from the store."""
        with self.engine.get_session() as session:
            chunks = session.query(TextChunkORM).all()
            
            result = []
            for chunk in chunks:
                chunk_dict = {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": chunk.metadata_json,
                }
                
                if with_embeddings and chunk.embedding is not None:
                    chunk_dict["embedding"] = np.frombuffer(chunk.embedding, dtype=np.float32)
                
                result.append(chunk_dict)
            
            return result
    
    def search_similar(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar chunks based on embedding similarity."""
        chunks = self.get_all_chunks(with_embeddings=True)
        
        if not chunks or chunks[0].get("embedding") is None:
            return []
        
        # Calculate cosine similarity
        similarities = []
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        for chunk in chunks:
            if chunk.get("embedding") is not None:
                chunk_norm = chunk["embedding"] / np.linalg.norm(chunk["embedding"])
                similarity = np.dot(query_norm, chunk_norm)
                similarities.append((chunk, float(similarity)))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def chunk_exists(self, chunk_id: str) -> bool:
        """Check if a chunk exists in the store."""
        with self.engine.get_session() as session:
            exists = session.get(TextChunkORM, chunk_id) is not None
            return exists