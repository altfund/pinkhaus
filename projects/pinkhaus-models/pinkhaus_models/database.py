"""
Database operations for storing transcription data in SQLite.
"""

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
        speakers_table: str = "speakers",
    ):
        """
        Create the database tables if they don't exist.

        Args:
            metadata_table: Name of the metadata table
            segments_table: Name of the segments table
            speakers_table: Name of the speakers table
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create metadata table with speaker diarization fields
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
                    num_speakers INTEGER,
                    has_speaker_labels BOOLEAN DEFAULT 0,
                    diarization_mode TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(file_hash, model_name)
                )
            """)

            # Create speakers table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {speakers_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcription_id INTEGER NOT NULL,
                    speaker_label TEXT NOT NULL,
                    total_segments INTEGER DEFAULT 0,
                    total_duration REAL DEFAULT 0.0,
                    first_appearance REAL,
                    last_appearance REAL,
                    FOREIGN KEY (transcription_id) REFERENCES {metadata_table}(id),
                    UNIQUE(transcription_id, speaker_label)
                )
            """)

            # Create segments table with speaker support
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {segments_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcription_id INTEGER NOT NULL,
                    segment_index INTEGER NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    duration REAL NOT NULL,
                    text TEXT NOT NULL,
                    speaker_id INTEGER,
                    speaker_confidence REAL,
                    FOREIGN KEY (transcription_id) REFERENCES {metadata_table}(id),
                    FOREIGN KEY (speaker_id) REFERENCES {speakers_table}(id),
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

            # Create indexes for speaker queries
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{speakers_table}_transcription_id
                ON {speakers_table}(transcription_id)
            """)

            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{segments_table}_speaker_id
                ON {segments_table}(speaker_id)
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

    def save_transcription(
        self,
        result: TranscriptionResult,
        model_name: str = "unknown",
        metadata_table: str = "transcription_metadata",
        segments_table: str = "transcription_segments",
        speakers_table: str = "speakers",
        feed_url: Optional[str] = None,
        feed_item_id: Optional[str] = None,
        feed_item_title: Optional[str] = None,
        feed_item_published: Optional[str] = None,
    ) -> int:
        """
        Save a transcription result to the database with speaker support.

        Args:
            result: TranscriptionResult object to save
            model_name: Name of the model used for transcription
            metadata_table: Name of the metadata table
            segments_table: Name of the segments table
            speakers_table: Name of the speakers table
            feed_url: URL of the RSS feed (optional)
            feed_item_id: Unique ID of the feed item (optional)
            feed_item_title: Title of the feed item (optional)
            feed_item_published: Publication date of the feed item (optional)

        Returns:
            The transcription_id of the saved record
        """
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

            # Get result as dict to access speaker metadata
            result_dict = result.to_dict()

            # Detect if we're using mock diarization
            diarization_mode = None
            if result_dict.get('has_speaker_labels'):
                # Check if any speaker has confidence exactly 1.0 and label format SPEAKER_N
                has_mock_pattern = any(
                    seg.get('speaker', '').startswith('SPEAKER_') and
                    seg.get('confidence', 0) == 1.0
                    for seg in result_dict['segments'] if seg.get('speaker')
                )
                diarization_mode = 'mock' if has_mock_pattern else 'real'

            # Insert metadata with speaker fields
            cursor.execute(
                f"""
                INSERT INTO {metadata_table}
                (filename, file_hash, language, full_text, model_name,
                 feed_url, feed_item_id, feed_item_title, feed_item_published,
                 num_speakers, has_speaker_labels, diarization_mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    result_dict.get('num_speakers'),
                    1 if result_dict.get('has_speaker_labels', False) else 0,
                    diarization_mode,
                ),
            )

            transcription_id = cursor.lastrowid

            # Create speaker entries if we have speaker labels
            speaker_ids = {}
            if result_dict.get('has_speaker_labels'):
                # Get unique speakers from segments
                unique_speakers = set()
                for seg in result.segments:
                    if hasattr(seg, 'speaker') and seg.speaker:
                        unique_speakers.add(seg.speaker)

                # Create speaker entries
                for speaker_label in unique_speakers:
                    cursor.execute(
                        f"""
                        INSERT INTO {speakers_table} (transcription_id, speaker_label)
                        VALUES (?, ?)
                    """,
                        (transcription_id, speaker_label),
                    )
                    speaker_ids[speaker_label] = cursor.lastrowid

            # Insert segments with speaker support
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

                # Get speaker info if available
                speaker_id = None
                speaker_confidence = None
                if hasattr(segment, 'speaker') and segment.speaker:
                    speaker_id = speaker_ids.get(segment.speaker)
                    if hasattr(segment, 'confidence'):
                        speaker_confidence = segment.confidence

                cursor.execute(
                    f"""
                    INSERT INTO {segments_table}
                    (transcription_id, segment_index, start_time, end_time, duration, text,
                     speaker_id, speaker_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        transcription_id,
                        idx,
                        start_time,
                        end_time,
                        duration,
                        text.strip(),
                        speaker_id,
                        speaker_confidence,
                    ),
                )

            # Update speaker statistics
            if speaker_ids:
                cursor.execute(
                    f"""
                    UPDATE {speakers_table}
                    SET total_segments = (
                        SELECT COUNT(*) FROM {segments_table} 
                        WHERE speaker_id = {speakers_table}.id
                    ),
                    total_duration = (
                        SELECT SUM(duration) FROM {segments_table} 
                        WHERE speaker_id = {speakers_table}.id
                    ),
                    first_appearance = (
                        SELECT MIN(start_time) FROM {segments_table} 
                        WHERE speaker_id = {speakers_table}.id
                    ),
                    last_appearance = (
                        SELECT MAX(end_time) FROM {segments_table} 
                        WHERE speaker_id = {speakers_table}.id
                    )
                    WHERE transcription_id = ?
                """,
                    (transcription_id,),
                )

            return transcription_id

    def get_transcription(
        self,
        transcription_id: int,
        metadata_table: str = "transcription_metadata",
        segments_table: str = "transcription_segments",
        speakers_table: str = "speakers",
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a transcription by ID with speaker information.

        Args:
            transcription_id: ID of the transcription to retrieve
            metadata_table: Name of the metadata table
            segments_table: Name of the segments table
            speakers_table: Name of the speakers table

        Returns:
            Dictionary with transcription data or None if not found
        """
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

            # Get segments with speaker information
            cursor.execute(
                f"""
                SELECT seg.*, spk.speaker_label 
                FROM {segments_table} seg
                LEFT JOIN {speakers_table} spk ON seg.speaker_id = spk.id
                WHERE seg.transcription_id = ?
                ORDER BY seg.segment_index
            """,
                (transcription_id,),
            )

            segments = cursor.fetchall()

            # Get speakers if available
            speakers = []
            if metadata['has_speaker_labels']:
                speakers = self.get_speakers(transcription_id, speakers_table)

            return {
                "metadata": dict(metadata),
                "segments": [dict(seg) for seg in segments],
                "speakers": speakers,
            }

    def check_feed_item_exists(
        self,
        feed_url: str,
        feed_item_id: str,
        metadata_table: str = "transcription_metadata",
    ) -> bool:
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

            cursor.execute(
                f"""
                SELECT COUNT(*) as count FROM {metadata_table}
                WHERE feed_url = ? AND feed_item_id = ?
            """,
                (feed_url, feed_item_id),
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

    def get_recent_transcriptions(
        self, limit: int = 10, metadata_table: str = "transcription_metadata"
    ) -> List[Dict[str, Any]]:
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

            cursor.execute(
                f"""
                SELECT * FROM {metadata_table}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            """,
                (limit,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_speakers(
        self,
        transcription_id: int,
        speakers_table: str = "speakers",
    ) -> List[Dict[str, Any]]:
        """
        Get all speakers for a transcription.

        Args:
            transcription_id: ID of the transcription
            speakers_table: Name of the speakers table

        Returns:
            List of speaker dictionaries with statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT * FROM {speakers_table}
                WHERE transcription_id = ?
                ORDER BY total_duration DESC
            """,
                (transcription_id,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_segments_by_speaker(
        self,
        transcription_id: int,
        speaker_label: str,
        segments_table: str = "transcription_segments",
        speakers_table: str = "speakers",
    ) -> List[Dict[str, Any]]:
        """
        Get all segments for a specific speaker.

        Args:
            transcription_id: ID of the transcription
            speaker_label: Label of the speaker (e.g., "SPEAKER_0")
            segments_table: Name of the segments table
            speakers_table: Name of the speakers table

        Returns:
            List of segment dictionaries for the speaker
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT seg.* FROM {segments_table} seg
                JOIN {speakers_table} spk ON seg.speaker_id = spk.id
                WHERE seg.transcription_id = ? AND spk.speaker_label = ?
                ORDER BY seg.segment_index
            """,
                (transcription_id, speaker_label),
            )

            return [dict(row) for row in cursor.fetchall()]

    def delete_transcription(
        self,
        transcription_id: int,
        metadata_table: str = "transcription_metadata",
        segments_table: str = "transcription_segments",
        speakers_table: str = "speakers",
    ) -> bool:
        """
        Delete a transcription and its segments and speakers.

        Args:
            transcription_id: ID of the transcription to delete
            metadata_table: Name of the metadata table
            segments_table: Name of the segments table
            speakers_table: Name of the speakers table

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # First delete segments
            cursor.execute(
                f"""
                DELETE FROM {segments_table} WHERE transcription_id = ?
            """,
                (transcription_id,),
            )

            # Delete speakers
            cursor.execute(
                f"""
                DELETE FROM {speakers_table} WHERE transcription_id = ?
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
        speakers_table: str = "speakers",
    ) -> Optional[TranscriptionResult]:
        """
        Export a transcription from database back to TranscriptionResult object with speaker support.

        Args:
            transcription_id: ID of the transcription
            metadata_table: Name of the metadata table
            segments_table: Name of the segments table
            speakers_table: Name of the speakers table

        Returns:
            TranscriptionResult object or None if not found
        """
        data = self.get_transcription(transcription_id, metadata_table, segments_table, speakers_table)
        if not data:
            return None

        metadata = data["metadata"]
        segments = []

        for seg in data["segments"]:
            # Create segment with basic info
            segment = Segment(start=seg["start_time"], end=seg["end_time"], text=seg["text"])

            # Add speaker info if available
            if seg.get("speaker_label"):
                segment.speaker = seg["speaker_label"]
            if seg.get("speaker_confidence") is not None:
                segment.confidence = seg["speaker_confidence"]

            segments.append(
                Segment(
                    start=seg["start_time"],
                    end=seg["end_time"],
                    text=seg["text"],
                )
            )

        # Create result
        result = TranscriptionResult(
            filename=metadata["filename"],
            file_hash=metadata["file_hash"],
            language=metadata["language"],
            segments=segments,
            full_text=metadata["full_text"],
        )

        # Add speaker metadata if available
        if metadata.get("num_speakers"):
            result._proto.num_speakers = metadata["num_speakers"]
        if metadata.get("has_speaker_labels"):
            result._proto.has_speaker_labels = metadata["has_speaker_labels"]

        return result

    def import_from_json(
        self,
        json_str: str,
        model_name: str = "unknown",
        **kwargs
    ) -> int:
        """
        Import transcription from JSON string and save to database.

        Args:
            json_str: JSON string containing transcription data
            model_name: Name of the model used for transcription
            **kwargs: Additional arguments passed to save_transcription

        Returns:
            The transcription_id of the saved record
        """
        result = TranscriptionResult.from_json(json_str)
        return self.save_transcription(result, model_name, **kwargs)

    def import_from_csv(
        self,
        csv_str: str,
        filename: str,
        file_hash: str,
        language: str = "en",
        model_name: str = "unknown",
        **kwargs
    ) -> int:
        """
        Import transcription from CSV string and save to database.
        Note: CSV format loses some metadata, so basic info must be provided.

        Args:
            csv_str: CSV string containing transcription data
            filename: Original filename
            file_hash: SHA256 hash of the original file
            language: Language code
            model_name: Name of the model used for transcription
            **kwargs: Additional arguments passed to save_transcription

        Returns:
            The transcription_id of the saved record
        """
        # Parse CSV to create TranscriptionResult
        import csv
        import io

        # Skip comment lines and find header
        lines = csv_str.strip().split('\n')
        data_lines = []
        for line in lines:
            if not line.startswith('#'):
                data_lines.append(line)

        reader = csv.DictReader(io.StringIO('\n'.join(data_lines)))
        segments = []
        full_text_parts = []

        for row in reader:
            # Parse time format HH:MM:SS.sss to seconds
            def parse_time(time_str):
                parts = time_str.split(':')
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds

            start = parse_time(row['Start'])
            end = parse_time(row['End'])
            text = row['Text']

            segment = Segment(start=start, end=end, text=text)

            # Add speaker info if present
            if 'Speaker' in row:
                segment.speaker = row['Speaker']
                # CSV doesn't include confidence, so we don't set it

            segments.append(segment)
            full_text_parts.append(text)

        # Create TranscriptionResult
        result = TranscriptionResult(
            filename=filename,
            file_hash=file_hash,
            language=language,
            segments=segments,
            full_text=' '.join(full_text_parts)
        )

        # Check if we have speaker info to set metadata
        if any(hasattr(seg, 'speaker') for seg in segments):
            result._proto.has_speaker_labels = True
            # Count unique speakers
            unique_speakers = set(seg.speaker for seg in segments if hasattr(seg, 'speaker') and seg.speaker)
            result._proto.num_speakers = len(unique_speakers)

        return self.save_transcription(result, model_name, **kwargs)

    def import_from_toml(
        self,
        toml_str: str,
        model_name: str = "unknown",
        **kwargs
    ) -> int:
        """
        Import transcription from TOML string and save to database.

        Args:
            toml_str: TOML string containing transcription data
            model_name: Name of the model used for transcription
            **kwargs: Additional arguments passed to save_transcription

        Returns:
            The transcription_id of the saved record
        """
        result = TranscriptionResult.from_toml(toml_str)
        return self.save_transcription(result, model_name, **kwargs)

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
