# This module provides backward compatibility.
# All functionality has been moved to database_sqlalchemy.py

from typing import Optional, List, Dict, Any

from .database_sqlalchemy import (
    TranscriptionDatabase as _TranscriptionDatabase,
    VectorStore as _VectorStore,
)


class TranscriptionDatabase(_TranscriptionDatabase):
    """Handle database operations for transcriptions.

    This class maintains backward compatibility while using SQLAlchemy internally.
    """

    def __init__(self, db_path: str = None, database_url: str = None):
        """Initialize database connection.

        Args:
            db_path: Path to the database file (legacy)
            database_url: SQLAlchemy database URL
        """
        if database_url:
            super().__init__(database_url=database_url)
        elif db_path:
            # Convert to SQLAlchemy URL
            database_url = f"sqlite:///{db_path}"
            super().__init__(database_url=database_url)
            self.db_path = db_path
        else:
            raise ValueError("Either db_path or database_url must be provided")

    # Legacy methods maintained for backward compatibility
    def get_recent_transcriptions(
        self, limit: int = 10, metadata_table: str = "transcription_metadata"
    ) -> List[Dict[str, Any]]:
        """Get the most recent transcriptions."""
        transcriptions = self.get_all_transcriptions(metadata_table)
        return [t.to_dict() for t in transcriptions[:limit]]

    def get_full_transcription(
        self,
        transcription_id: int,
        metadata_table: str = "transcription_metadata",
        segments_table: str = "transcription_segments",
    ) -> Optional[Dict[str, Any]]:
        """Alias for get_transcription for backward compatibility."""
        return self.get_transcription(transcription_id, metadata_table, segments_table)


class VectorStore(_VectorStore):
    """Vector store for text chunks with embeddings.

    This class maintains backward compatibility while using SQLAlchemy internally.
    """

    def __init__(self, db_path: str = "grant_vectors.db"):
        """Initialize vector store.

        Args:
            db_path: Path to the database file
        """
        # Convert to SQLAlchemy URL
        database_url = f"sqlite:///{db_path}"
        super().__init__(database_url=database_url)
        self.db_path = db_path
