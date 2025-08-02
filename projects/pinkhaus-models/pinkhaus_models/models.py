from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Segment:
    """A transcription segment with timing information."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    text: str

    @property
    def start_ms(self) -> int:
        """Start time in milliseconds."""
        return int(self.start * 1000)

    @property
    def end_ms(self) -> int:
        """End time in milliseconds."""
        return int(self.end * 1000)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text.strip(),
            "duration": self.duration,
        }


@dataclass
class TranscriptionResult:
    """Complete transcription result."""

    filename: str
    file_hash: str
    language: str
    segments: List[Segment]
    full_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "filename": self.filename,
            "file_hash": self.file_hash,
            "language": self.language,
            "segments": [seg.to_dict() for seg in self.segments],
            "full_text": self.full_text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionResult":
        """Create from dictionary."""
        segments = [
            Segment(start=seg["start"], end=seg["end"], text=seg["text"])
            for seg in data["segments"]
        ]

        return cls(
            filename=data["filename"],
            file_hash=data["file_hash"],
            language=data["language"],
            segments=segments,
            full_text=data["full_text"],
        )


@dataclass
class TranscriptionMetadata:
    """Database model for transcription metadata."""

    id: Optional[int] = None
    filename: str = ""
    file_hash: str = ""
    language: str = ""
    full_text: str = ""
    model_name: Optional[str] = None
    feed_url: Optional[str] = None
    feed_item_id: Optional[str] = None
    feed_item_title: Optional[str] = None
    feed_item_published: Optional[datetime] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "filename": self.filename,
            "file_hash": self.file_hash,
            "language": self.language,
            "full_text": self.full_text,
            "model_name": self.model_name,
            "feed_url": self.feed_url,
            "feed_item_id": self.feed_item_id,
            "feed_item_title": self.feed_item_title,
            "feed_item_published": self.feed_item_published.isoformat()
            if self.feed_item_published
            else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "TranscriptionMetadata":
        """Create from database row."""
        return cls(
            id=row.get("id"),
            filename=row.get("filename", ""),
            file_hash=row.get("file_hash", ""),
            language=row.get("language", ""),
            full_text=row.get("full_text", ""),
            model_name=row.get("model_name"),
            feed_url=row.get("feed_url"),
            feed_item_id=row.get("feed_item_id"),
            feed_item_title=row.get("feed_item_title"),
            feed_item_published=datetime.fromisoformat(row["feed_item_published"])
            if row.get("feed_item_published")
            else None,
            created_at=datetime.fromisoformat(row["created_at"])
            if row.get("created_at")
            else None,
        )


@dataclass
class TranscriptionSegment:
    """Database model for transcription segments."""

    id: Optional[int] = None
    transcription_id: int = 0
    segment_index: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "transcription_id": self.transcription_id,
            "segment_index": self.segment_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "text": self.text,
        }

    def to_segment(self) -> Segment:
        """Convert to Segment object."""
        return Segment(start=self.start_time, end=self.end_time, text=self.text)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "TranscriptionSegment":
        """Create from database row."""
        return cls(
            id=row.get("id"),
            transcription_id=row.get("transcription_id", 0),
            segment_index=row.get("segment_index", 0),
            start_time=row.get("start_time", 0.0),
            end_time=row.get("end_time", 0.0),
            duration=row.get("duration", 0.0),
            text=row.get("text", ""),
        )
