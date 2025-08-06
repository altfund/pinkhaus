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

    def create_speaker_identity_tables(
        self,
        profiles_table: str = "speaker_profiles",
        embeddings_table: str = "speaker_embeddings",
        occurrences_table: str = "speaker_occurrences",
        profile_metadata_table: str = "speaker_metadata",
        segments_table: str = "transcription_segments",
    ):
        """
        Create tables for persistent speaker identity tracking.

        Args:
            profiles_table: Name of the speaker profiles table
            embeddings_table: Name of the voice embeddings table
            occurrences_table: Name of the speaker occurrences table
            profile_metadata_table: Name of the profile metadata table
            segments_table: Name of the segments table to update
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create speaker profiles table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {profiles_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feed_url TEXT,
                    display_name TEXT NOT NULL,
                    canonical_label TEXT,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    total_appearances INTEGER DEFAULT 0,
                    total_duration_seconds REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(feed_url, display_name)
                )
            """)

            # Create voice embeddings table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {embeddings_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    source_transcription_id INTEGER,
                    source_segment_indices TEXT,
                    duration_seconds REAL,
                    quality_score REAL,
                    extraction_method TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (profile_id) REFERENCES {profiles_table}(id),
                    FOREIGN KEY (source_transcription_id) REFERENCES transcription_metadata(id)
                )
            """)

            # Create speaker occurrences table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {occurrences_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcription_id INTEGER NOT NULL,
                    temporary_label TEXT NOT NULL,
                    profile_id INTEGER,
                    confidence REAL,
                    is_verified BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (transcription_id) REFERENCES transcription_metadata(id),
                    FOREIGN KEY (profile_id) REFERENCES {profiles_table}(id),
                    UNIQUE(transcription_id, temporary_label)
                )
            """)

            # Create speaker metadata table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {profile_metadata_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (profile_id) REFERENCES {profiles_table}(id),
                    UNIQUE(profile_id, key)
                )
            """)

            # Add profile_id to segments table if it doesn't exist
            cursor.execute(f"""
                PRAGMA table_info({segments_table})
            """)
            columns = [col[1] for col in cursor.fetchall()]

            if 'profile_id' not in columns:
                cursor.execute(f"""
                    ALTER TABLE {segments_table} 
                    ADD COLUMN profile_id INTEGER 
                    REFERENCES {profiles_table}(id)
                """)

            # Create indexes for efficient queries
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{profiles_table}_feed_url
                ON {profiles_table}(feed_url)
            """)

            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{embeddings_table}_profile_id
                ON {embeddings_table}(profile_id)
            """)

            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{occurrences_table}_transcription_id
                ON {occurrences_table}(transcription_id)
            """)

            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{occurrences_table}_profile_id
                ON {occurrences_table}(profile_id)
            """)

            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{segments_table}_profile_id
                ON {segments_table}(profile_id)
            """)

            # Create failed items table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS failed_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feed_url TEXT NOT NULL,
                    feed_item_id TEXT NOT NULL,
                    feed_item_title TEXT,
                    audio_url TEXT,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    failure_count INTEGER DEFAULT 1,
                    first_failed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_failed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(feed_url, feed_item_id)
                )
            """)

            # Create processing state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feed_url TEXT NOT NULL,
                    feed_item_id TEXT NOT NULL,
                    feed_item_title TEXT,
                    audio_url TEXT,
                    state TEXT NOT NULL CHECK(state IN ('downloading', 'transcribing', 'indexing')),
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    pid INTEGER,
                    hostname TEXT,
                    UNIQUE(feed_url, feed_item_id)
                )
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

            # Handle speaker identification if embeddings are present
            if hasattr(result, '_speaker_embeddings') and result._speaker_embeddings:
                try:
                    # Import here to avoid circular dependency
                    from beige_book.speaker_matcher import SpeakerMatcher

                    # Create matcher with this database instance
                    matcher = SpeakerMatcher(self, embedding_method="mock")

                    # Get feed URL if provided
                    feed_url_for_matching = getattr(result, '_feed_url', None) or feed_url

                    # Identify speakers and link to profiles
                    speaker_mappings = matcher.identify_speakers_in_transcription(
                        transcription_id=transcription_id,
                        audio_path=result.filename,  # This might need to be full path
                        embeddings=result._speaker_embeddings,
                        feed_url=feed_url_for_matching
                    )

                    print(f"Identified {len(speaker_mappings)} speakers with persistent profiles")

                except Exception as e:
                    print(f"Warning: Speaker identification during save failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue without identification

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

    def create_speaker_profile(
        self,
        display_name: str,
        feed_url: Optional[str] = None,
        canonical_label: Optional[str] = None,
        profiles_table: str = "speaker_profiles",
    ) -> int:
        """
        Create a new persistent speaker profile.

        Args:
            display_name: Display name for the speaker
            feed_url: Optional feed URL to scope the profile to
            canonical_label: Optional canonical label (e.g., "HOST", "GUEST_1")
            profiles_table: Name of the profiles table

        Returns:
            Profile ID of the created profile
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if profile already exists
            cursor.execute(
                f"""
                SELECT id FROM {profiles_table}
                WHERE display_name = ? AND (feed_url = ? OR (feed_url IS NULL AND ? IS NULL))
                """,
                (display_name, feed_url, feed_url),
            )

            existing = cursor.fetchone()
            if existing:
                return existing["id"]

            # Create new profile
            cursor.execute(
                f"""
                INSERT INTO {profiles_table}
                (display_name, feed_url, canonical_label, first_seen, last_seen)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (display_name, feed_url, canonical_label),
            )

            return cursor.lastrowid

    def add_speaker_embedding(
        self,
        profile_id: int,
        embedding: bytes,  # Serialized numpy array
        embedding_dimension: int,
        source_transcription_id: Optional[int] = None,
        source_segment_indices: Optional[List[int]] = None,
        duration_seconds: Optional[float] = None,
        quality_score: Optional[float] = None,
        extraction_method: str = "speechbrain",
        embeddings_table: str = "speaker_embeddings",
    ) -> int:
        """
        Add a voice embedding to a speaker profile.

        Args:
            profile_id: ID of the speaker profile
            embedding: Serialized embedding vector (numpy array as bytes)
            embedding_dimension: Dimension of the embedding
            source_transcription_id: Optional source transcription
            source_segment_indices: Optional list of segment indices used
            duration_seconds: Total duration of audio used
            quality_score: Quality/confidence score
            extraction_method: Method used to extract embedding
            embeddings_table: Name of the embeddings table

        Returns:
            Embedding ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Convert segment indices to JSON string
            indices_json = None
            if source_segment_indices:
                import json
                indices_json = json.dumps(source_segment_indices)

            cursor.execute(
                f"""
                INSERT INTO {embeddings_table}
                (profile_id, embedding, embedding_dimension, source_transcription_id,
                 source_segment_indices, duration_seconds, quality_score, extraction_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_id,
                    embedding,
                    embedding_dimension,
                    source_transcription_id,
                    indices_json,
                    duration_seconds,
                    quality_score,
                    extraction_method,
                ),
            )

            return cursor.lastrowid

    def link_speaker_occurrence(
        self,
        transcription_id: int,
        temporary_label: str,
        profile_id: int,
        confidence: float,
        is_verified: bool = False,
        occurrences_table: str = "speaker_occurrences",
        profiles_table: str = "speaker_profiles",
    ) -> int:
        """
        Link a temporary speaker label to a permanent profile.

        Args:
            transcription_id: ID of the transcription
            temporary_label: Temporary label from diarization (e.g., "SPEAKER_0")
            profile_id: ID of the matched speaker profile
            confidence: Confidence score of the match
            is_verified: Whether this match has been human-verified
            occurrences_table: Name of the occurrences table
            profiles_table: Name of the profiles table

        Returns:
            Occurrence ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {occurrences_table}
                (transcription_id, temporary_label, profile_id, confidence, is_verified)
                VALUES (?, ?, ?, ?, ?)
                """,
                (transcription_id, temporary_label, profile_id, confidence, 1 if is_verified else 0),
            )

            occurrence_id = cursor.lastrowid

            # Update profile statistics
            cursor.execute(
                f"""
                UPDATE {profiles_table}
                SET total_appearances = total_appearances + 1,
                    last_seen = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (profile_id,),
            )

            return occurrence_id

    def get_speaker_embeddings(
        self,
        profile_id: int,
        embeddings_table: str = "speaker_embeddings",
    ) -> List[Dict[str, Any]]:
        """
        Get all embeddings for a speaker profile.

        Args:
            profile_id: ID of the speaker profile
            embeddings_table: Name of the embeddings table

        Returns:
            List of embedding records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT * FROM {embeddings_table}
                WHERE profile_id = ?
                ORDER BY created_at DESC
                """,
                (profile_id,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_speaker_profiles_for_feed(
        self,
        feed_url: str,
        profiles_table: str = "speaker_profiles",
    ) -> List[Dict[str, Any]]:
        """
        Get all speaker profiles for a specific feed.

        Args:
            feed_url: URL of the feed
            profiles_table: Name of the profiles table

        Returns:
            List of speaker profiles
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT * FROM {profiles_table}
                WHERE feed_url = ?
                ORDER BY total_appearances DESC
                """,
                (feed_url,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_speaker_history(
        self,
        profile_id: int,
        limit: int = 100,
        occurrences_table: str = "speaker_occurrences",
        metadata_table: str = "transcription_metadata",
    ) -> List[Dict[str, Any]]:
        """
        Get appearance history for a speaker.

        Args:
            profile_id: ID of the speaker profile
            limit: Maximum number of appearances to return
            occurrences_table: Name of the occurrences table
            metadata_table: Name of the metadata table

        Returns:
            List of appearances with transcription info
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT o.*, t.filename, t.created_at as transcription_date
                FROM {occurrences_table} o
                JOIN {metadata_table} t ON o.transcription_id = t.id
                WHERE o.profile_id = ?
                ORDER BY t.created_at DESC
                LIMIT ?
                """,
                (profile_id, limit),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_speaker_statements(
        self,
        profile_id: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        segments_table: str = "transcription_segments",
        metadata_table: str = "transcription_metadata",
    ) -> List[Dict[str, Any]]:
        """
        Get all statements made by a speaker.

        Args:
            profile_id: ID of the speaker profile
            start_date: Optional start date filter
            end_date: Optional end date filter
            segments_table: Name of the segments table
            metadata_table: Name of the metadata table

        Returns:
            List of segments with transcription context
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = f"""
                SELECT s.*, t.filename, t.created_at as transcription_date, t.feed_url
                FROM {segments_table} s
                JOIN {metadata_table} t ON s.transcription_id = t.id
                WHERE s.profile_id = ?
            """

            params = [profile_id]

            if start_date:
                query += " AND t.created_at >= ?"
                params.append(start_date)

            if end_date:
                query += " AND t.created_at <= ?"
                params.append(end_date)

            query += " ORDER BY t.created_at, s.start_time"

            cursor.execute(query, params)

            return [dict(row) for row in cursor.fetchall()]

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

    def record_failed_item(
        self,
        feed_url: str,
        feed_item_id: str,
        error_type: str,
        error_message: str,
        feed_item_title: Optional[str] = None,
        audio_url: Optional[str] = None,
    ) -> None:
        """Record a failed processing attempt for a feed item."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if already exists
            cursor.execute(
                """
                SELECT id, failure_count FROM failed_items
                WHERE feed_url = ? AND feed_item_id = ?
            """,
                (feed_url, feed_item_id),
            )

            existing = cursor.fetchone()
            if existing:
                # Update existing record
                cursor.execute(
                    """
                    UPDATE failed_items
                    SET failure_count = failure_count + 1,
                        last_failed_at = CURRENT_TIMESTAMP,
                        error_type = ?,
                        error_message = ?
                    WHERE id = ?
                """,
                    (error_type, error_message, existing["id"]),
                )
            else:
                # Insert new record
                cursor.execute(
                    """
                    INSERT INTO failed_items
                    (feed_url, feed_item_id, feed_item_title, audio_url, 
                     error_type, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        feed_url,
                        feed_item_id,
                        feed_item_title,
                        audio_url,
                        error_type,
                        error_message,
                    ),
                )

    def get_failed_item(
        self, feed_url: str, feed_item_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get failed item info if it exists."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM failed_items
                WHERE feed_url = ? AND feed_item_id = ?
            """,
                (feed_url, feed_item_id),
            )

            row = cursor.fetchone()
            return dict(row) if row else None

    def set_processing_state(
        self,
        feed_url: str,
        feed_item_id: str,
        state: str,
        feed_item_title: Optional[str] = None,
        audio_url: Optional[str] = None,
        pid: Optional[int] = None,
        hostname: Optional[str] = None,
    ) -> None:
        """Set the processing state for a feed item."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO processing_state
                (feed_url, feed_item_id, feed_item_title, audio_url, 
                 state, started_at, pid, hostname)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
            """,
                (
                    feed_url,
                    feed_item_id,
                    feed_item_title,
                    audio_url,
                    state,
                    pid,
                    hostname,
                ),
            )

    def clear_processing_state(self, feed_url: str, feed_item_id: str) -> None:
        """Clear the processing state for a feed item."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM processing_state
                WHERE feed_url = ? AND feed_item_id = ?
            """,
                (feed_url, feed_item_id),
            )

    def get_stale_processing_items(
        self, stale_minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """Get items that have been in processing state for too long."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM processing_state
                WHERE datetime(started_at) < datetime('now', '-' || ? || ' minutes')
                ORDER BY started_at
            """,
                (stale_minutes,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_failed_items_summary(self) -> List[Dict[str, Any]]:
        """Get summary of failed items grouped by feed."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    feed_url,
                    COUNT(*) as failed_count,
                    MAX(failure_count) as max_failures,
                    MIN(first_failed_at) as earliest_failure,
                    MAX(last_failed_at) as latest_failure
                FROM failed_items
                GROUP BY feed_url
                ORDER BY failed_count DESC
            """)

            return [dict(row) for row in cursor.fetchall()]

    def get_all_failed_items(self) -> List[Dict[str, Any]]:
        """Get all failed items with details."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM failed_items
                ORDER BY feed_url, last_failed_at DESC
            """)

            return [dict(row) for row in cursor.fetchall()]
