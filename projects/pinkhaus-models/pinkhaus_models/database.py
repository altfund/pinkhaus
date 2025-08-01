import sqlite3
import aiosqlite
from typing import Optional, List, Dict, Any
from contextlib import contextmanager, asynccontextmanager
from pathlib import Path

from .models import (
    TranscriptionMetadata,
    TranscriptionSegment,
    TranscriptionResult,
    Segment,
)


class TranscriptionDatabase:
    """Handle SQLite database operations for transcriptions."""

    def __init__(self, db_path: str):
        """Initialize database connection."""
        self.db_path = db_path
        self._ensure_parent_directory()

    def _ensure_parent_directory(self):
        """Ensure the parent directory for the database exists."""
        parent = Path(self.db_path).parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @asynccontextmanager
    async def _get_async_connection(self):
        """Async context manager for database connections."""
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        finally:
            await conn.close()

    def create_tables(
        self,
        metadata_table: str = "transcription_metadata",
        segments_table: str = "transcription_segments",
    ):
        """Create the database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create metadata table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {metadata_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    language TEXT NOT NULL,
                    full_text TEXT NOT NULL,
                    model_name TEXT,
                    feed_url TEXT,
                    feed_item_id TEXT,
                    feed_item_title TEXT,
                    feed_item_published TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(file_hash, model_name)
                )
            """)

            # Create segments table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {segments_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcription_id INTEGER NOT NULL,
                    segment_index INTEGER NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    duration REAL NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY (transcription_id) REFERENCES {metadata_table}(id),
                    UNIQUE(transcription_id, segment_index)
                )
            """)

            # Create indexes
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{segments_table}_transcription_id
                ON {segments_table}(transcription_id)
            """)

            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{metadata_table}_file_hash
                ON {metadata_table}(file_hash)
            """)

            cursor.execute(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_{metadata_table}_feed_item
                ON {metadata_table}(feed_url, feed_item_id)
                WHERE feed_url IS NOT NULL AND feed_item_id IS NOT NULL
            """)

    def get_all_transcriptions(
        self, metadata_table: str = "transcription_metadata"
    ) -> List[TranscriptionMetadata]:
        """Get all transcriptions from the database."""
        with self._get_connection() as conn:
            cursor = conn.execute(f"""
                SELECT * FROM {metadata_table}
                ORDER BY feed_item_published DESC, created_at DESC
            """)

            return [TranscriptionMetadata.from_row(dict(row)) for row in cursor]

    async def get_all_transcriptions_async(
        self, metadata_table: str = "transcription_metadata"
    ) -> List[TranscriptionMetadata]:
        """Get all transcriptions from the database asynchronously."""
        async with self._get_async_connection() as conn:
            cursor = await conn.execute(f"""
                SELECT * FROM {metadata_table}
                ORDER BY feed_item_published DESC, created_at DESC
            """)

            rows = await cursor.fetchall()
            return [TranscriptionMetadata.from_row(dict(row)) for row in rows]

    def get_transcription_metadata(
        self, transcription_id: int, metadata_table: str = "transcription_metadata"
    ) -> Optional[TranscriptionMetadata]:
        """Get metadata for a specific transcription."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT * FROM {metadata_table} WHERE id = ?
            """,
                (transcription_id,),
            )

            row = cursor.fetchone()
            return TranscriptionMetadata.from_row(dict(row)) if row else None

    def get_segments_for_transcription(
        self, transcription_id: int, segments_table: str = "transcription_segments"
    ) -> List[TranscriptionSegment]:
        """Get all segments for a transcription."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT * FROM {segments_table}
                WHERE transcription_id = ?
                ORDER BY segment_index
            """,
                (transcription_id,),
            )

            return [TranscriptionSegment.from_row(dict(row)) for row in cursor]

    async def get_segments_for_transcription_async(
        self, transcription_id: int, segments_table: str = "transcription_segments"
    ) -> List[TranscriptionSegment]:
        """Get all segments for a transcription asynchronously."""
        async with self._get_async_connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT * FROM {segments_table}
                WHERE transcription_id = ?
                ORDER BY segment_index
            """,
                (transcription_id,),
            )

            rows = await cursor.fetchall()
            return [TranscriptionSegment.from_row(dict(row)) for row in rows]

    def get_transcription(
        self,
        transcription_id: int,
        metadata_table: str = "transcription_metadata",
        segments_table: str = "transcription_segments",
    ) -> Optional[Dict[str, Any]]:
        """Get complete transcription with metadata and segments."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get metadata
            cursor.execute(
                f"""
                SELECT * FROM {metadata_table} WHERE id = ?
            """,
                (transcription_id,),
            )

            metadata = cursor.fetchone()
            if not metadata:
                return None

            # Get segments
            cursor.execute(
                f"""
                SELECT * FROM {segments_table}
                WHERE transcription_id = ?
                ORDER BY segment_index
            """,
                (transcription_id,),
            )

            segments = cursor.fetchall()

            return {
                "metadata": dict(metadata),
                "segments": [dict(seg) for seg in segments],
            }

    def get_full_transcription(
        self,
        transcription_id: int,
        metadata_table: str = "transcription_metadata",
        segments_table: str = "transcription_segments",
    ) -> Optional[Dict[str, Any]]:
        """Alias for get_transcription for backward compatibility."""
        return self.get_transcription(transcription_id, metadata_table, segments_table)

    def save_transcription(
        self,
        result: TranscriptionResult,
        model_name: str = "unknown",
        metadata_table: str = "transcription_metadata",
        segments_table: str = "transcription_segments",
        feed_url: Optional[str] = None,
        feed_item_id: Optional[str] = None,
        feed_item_title: Optional[str] = None,
        feed_item_published: Optional[str] = None,
    ) -> int:
        """Save a transcription result to the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if already exists
            cursor.execute(
                f"""
                SELECT id FROM {metadata_table}
                WHERE file_hash = ? AND model_name = ?
            """,
                (result.file_hash, model_name),
            )

            existing = cursor.fetchone()
            if existing:
                return existing["id"]

            # Insert metadata
            cursor.execute(
                f"""
                INSERT INTO {metadata_table}
                (filename, file_hash, language, full_text, model_name,
                 feed_url, feed_item_id, feed_item_title, feed_item_published)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result.filename,
                    result.file_hash,
                    result.language,
                    result.full_text,
                    model_name,
                    feed_url,
                    feed_item_id,
                    feed_item_title,
                    feed_item_published,
                ),
            )

            transcription_id = cursor.lastrowid

            # Insert segments
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

                cursor.execute(
                    f"""
                    INSERT INTO {segments_table}
                    (transcription_id, segment_index, start_time, end_time, duration, text)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        transcription_id,
                        idx,
                        start_time,
                        end_time,
                        duration,
                        text.strip(),
                    ),
                )

            return transcription_id

    def search_transcriptions(
        self,
        query: str,
        metadata_table: str = "transcription_metadata",
        limit: int = 10,
    ) -> List[TranscriptionMetadata]:
        """Search transcriptions by text content."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT * FROM {metadata_table}
                WHERE full_text LIKE ?
                ORDER BY feed_item_published DESC, created_at DESC
                LIMIT ?
            """,
                (f"%{query}%", limit),
            )

            return [TranscriptionMetadata.from_row(dict(row)) for row in cursor]

    def find_by_hash(
        self, file_hash: str, metadata_table: str = "transcription_metadata"
    ) -> List[Dict[str, Any]]:
        """Find all transcriptions for a given file hash."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT * FROM {metadata_table}
                WHERE file_hash = ?
                ORDER BY created_at DESC
            """,
                (file_hash,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def delete_transcription(
        self,
        transcription_id: int,
        metadata_table: str = "transcription_metadata",
        segments_table: str = "transcription_segments",
    ) -> bool:
        """Delete a transcription and its segments."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # First delete segments
            cursor.execute(
                f"""
                DELETE FROM {segments_table} WHERE transcription_id = ?
            """,
                (transcription_id,),
            )

            # Then delete metadata
            cursor.execute(
                f"""
                DELETE FROM {metadata_table} WHERE id = ?
            """,
                (transcription_id,),
            )

            return cursor.rowcount > 0

    def export_to_dict(
        self,
        transcription_id: int,
        metadata_table: str = "transcription_metadata",
        segments_table: str = "transcription_segments",
    ) -> Optional[TranscriptionResult]:
        """Export a transcription from database back to TranscriptionResult object."""
        data = self.get_transcription(transcription_id, metadata_table, segments_table)
        if not data:
            return None

        metadata = data["metadata"]
        segments = []

        for seg in data["segments"]:
            segments.append(
                Segment(
                    start_ms=int(seg["start_time"] * 1000),
                    end_ms=int(seg["end_time"] * 1000),
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

    def get_recent_transcriptions(
        self, limit: int = 10, metadata_table: str = "transcription_metadata"
    ) -> List[Dict[str, Any]]:
        """Get the most recent transcriptions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT * FROM {metadata_table}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            """,
                (limit,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def check_feed_item_exists(
        self,
        feed_url: str,
        feed_item_id: str,
        metadata_table: str = "transcription_metadata",
    ) -> bool:
        """Check if a feed item has already been processed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT COUNT(*) as count FROM {metadata_table}
                WHERE feed_url = ? AND feed_item_id = ?
            """,
                (feed_url, feed_item_id),
            )

            result = cursor.fetchone()
            return result["count"] > 0

    def get_transcriptions_by_date_range(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        metadata_table: str = "transcription_metadata",
    ) -> List[TranscriptionMetadata]:
        """Get transcriptions within a date range."""
        query = f"SELECT * FROM {metadata_table} WHERE 1=1"
        params = []

        if start_date:
            query += " AND feed_item_published >= ?"
            params.append(start_date)

        if end_date:
            query += " AND feed_item_published <= ?"
            params.append(end_date)

        query += " ORDER BY feed_item_published DESC, created_at DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [TranscriptionMetadata.from_row(dict(row)) for row in cursor]
