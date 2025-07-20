"""
Database operations for storing transcription data in SQLite.
"""

import sqlite3
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

from .transcriber import TranscriptionResult, Segment


class TranscriptionDatabase:
    """Handle SQLite database operations for transcriptions"""

    def __init__(self, db_path: str):
        """Initialize database connection"""
        self.db_path = db_path
        self._ensure_parent_directory()

    def _ensure_parent_directory(self):
        """Ensure the parent directory for the database exists"""
        parent = Path(self.db_path).parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_tables(self, metadata_table: str = "transcription_metadata",
                     segments_table: str = "transcription_segments"):
        """
        Create the database tables if they don't exist.

        Args:
            metadata_table: Name of the metadata table
            segments_table: Name of the segments table
        """
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

            # Create indexes for better query performance
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{segments_table}_transcription_id
                ON {segments_table}(transcription_id)
            """)

            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{metadata_table}_file_hash
                ON {metadata_table}(file_hash)
            """)

            # Create index for feed tracking
            cursor.execute(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_{metadata_table}_feed_item
                ON {metadata_table}(feed_url, feed_item_id)
                WHERE feed_url IS NOT NULL AND feed_item_id IS NOT NULL
            """)

    def save_transcription(self, result: TranscriptionResult,
                          model_name: str = "unknown",
                          metadata_table: str = "transcription_metadata",
                          segments_table: str = "transcription_segments",
                          feed_url: Optional[str] = None,
                          feed_item_id: Optional[str] = None,
                          feed_item_title: Optional[str] = None,
                          feed_item_published: Optional[str] = None) -> int:
        """
        Save a transcription result to the database.

        Args:
            result: TranscriptionResult object to save
            model_name: Name of the model used for transcription
            metadata_table: Name of the metadata table
            segments_table: Name of the segments table
            feed_url: URL of the RSS feed (optional)
            feed_item_id: Unique ID of the feed item (optional)
            feed_item_title: Title of the feed item (optional)
            feed_item_published: Publication date of the feed item (optional)

        Returns:
            The transcription_id of the saved record
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if this transcription already exists
            cursor.execute(f"""
                SELECT id FROM {metadata_table}
                WHERE file_hash = ? AND model_name = ?
            """, (result.file_hash, model_name))

            existing = cursor.fetchone()
            if existing:
                return existing['id']

            # Insert metadata
            cursor.execute(f"""
                INSERT INTO {metadata_table}
                (filename, file_hash, language, full_text, model_name,
                 feed_url, feed_item_id, feed_item_title, feed_item_published)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (result.filename, result.file_hash, result.language,
                  result.full_text, model_name, feed_url, feed_item_id,
                  feed_item_title, feed_item_published))

            transcription_id = cursor.lastrowid

            # Insert segments
            for idx, segment in enumerate(result.segments):
                duration = segment.end - segment.start
                cursor.execute(f"""
                    INSERT INTO {segments_table}
                    (transcription_id, segment_index, start_time, end_time, duration, text)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (transcription_id, idx, segment.start, segment.end,
                      duration, segment.text.strip()))

            return transcription_id

    def get_transcription(self, transcription_id: int,
                         metadata_table: str = "transcription_metadata",
                         segments_table: str = "transcription_segments") -> Optional[Dict[str, Any]]:
        """
        Retrieve a transcription by ID.

        Args:
            transcription_id: ID of the transcription to retrieve
            metadata_table: Name of the metadata table
            segments_table: Name of the segments table

        Returns:
            Dictionary with transcription data or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get metadata
            cursor.execute(f"""
                SELECT * FROM {metadata_table} WHERE id = ?
            """, (transcription_id,))

            metadata = cursor.fetchone()
            if not metadata:
                return None

            # Get segments
            cursor.execute(f"""
                SELECT * FROM {segments_table}
                WHERE transcription_id = ?
                ORDER BY segment_index
            """, (transcription_id,))

            segments = cursor.fetchall()

            return {
                'metadata': dict(metadata),
                'segments': [dict(seg) for seg in segments]
            }

    def check_feed_item_exists(self, feed_url: str, feed_item_id: str,
                               metadata_table: str = "transcription_metadata") -> bool:
        """
        Check if a feed item has already been processed.

        Args:
            feed_url: URL of the RSS feed
            feed_item_id: Unique ID of the feed item
            metadata_table: Name of the metadata table

        Returns:
            True if the feed item exists, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(f"""
                SELECT COUNT(*) as count FROM {metadata_table}
                WHERE feed_url = ? AND feed_item_id = ?
            """, (feed_url, feed_item_id))

            result = cursor.fetchone()
            return result['count'] > 0

    def find_by_hash(self, file_hash: str,
                    metadata_table: str = "transcription_metadata") -> List[Dict[str, Any]]:
        """
        Find all transcriptions for a given file hash.

        Args:
            file_hash: SHA256 hash of the file
            metadata_table: Name of the metadata table

        Returns:
            List of transcription metadata dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(f"""
                SELECT * FROM {metadata_table}
                WHERE file_hash = ?
                ORDER BY created_at DESC
            """, (file_hash,))

            return [dict(row) for row in cursor.fetchall()]

    def get_recent_transcriptions(self, limit: int = 10,
                                 metadata_table: str = "transcription_metadata") -> List[Dict[str, Any]]:
        """
        Get the most recent transcriptions.

        Args:
            limit: Maximum number of transcriptions to return
            metadata_table: Name of the metadata table

        Returns:
            List of transcription metadata dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(f"""
                SELECT * FROM {metadata_table}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def delete_transcription(self, transcription_id: int,
                           metadata_table: str = "transcription_metadata",
                           segments_table: str = "transcription_segments") -> bool:
        """
        Delete a transcription and its segments.

        Args:
            transcription_id: ID of the transcription to delete
            metadata_table: Name of the metadata table
            segments_table: Name of the segments table

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # First delete segments
            cursor.execute(f"""
                DELETE FROM {segments_table} WHERE transcription_id = ?
            """, (transcription_id,))

            # Then delete metadata
            cursor.execute(f"""
                DELETE FROM {metadata_table} WHERE id = ?
            """, (transcription_id,))

            return cursor.rowcount > 0

    def export_to_dict(self, transcription_id: int,
                      metadata_table: str = "transcription_metadata",
                      segments_table: str = "transcription_segments") -> Optional[TranscriptionResult]:
        """
        Export a transcription from database back to TranscriptionResult object.

        Args:
            transcription_id: ID of the transcription
            metadata_table: Name of the metadata table
            segments_table: Name of the segments table

        Returns:
            TranscriptionResult object or None if not found
        """
        data = self.get_transcription(transcription_id, metadata_table, segments_table)
        if not data:
            return None

        metadata = data['metadata']
        segments = []

        for seg in data['segments']:
            segments.append(Segment(
                start=seg['start_time'],
                end=seg['end_time'],
                text=seg['text']
            ))

        return TranscriptionResult(
            filename=metadata['filename'],
            file_hash=metadata['file_hash'],
            language=metadata['language'],
            segments=segments,
            full_text=metadata['full_text']
        )
