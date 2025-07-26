"""
Transcription library using Protocol Buffers as the primary data structure.
"""

# Extended result support
from .transcription_pb2 import ExtendedTranscriptionResult
import os
import json
import hashlib
import io
import base64
import time
from contextlib import redirect_stdout
from typing import List, Dict, Any, Optional
from datetime import timedelta
from tabulate import tabulate
import whisper
import toml

from .transcription_pb2 import (
    TranscriptionResult as TranscriptionResultProto,
    Segment as SegmentProto,
)


# Backward compatibility - provide a Segment class
class Segment:
    """Segment class for backward compatibility."""

    def __init__(self, start: float, end: float, text: str):
        # Store times in seconds for backward compatibility
        self.start = start
        self.end = end
        self.text = text
        # Also store as milliseconds for consistency
        self.start_ms = int(start * 1000)
        self.end_ms = int(end * 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with formatted times."""
        return {
            "start": TranscriptionResult.format_time(self.start),
            "end": TranscriptionResult.format_time(self.end),
            "text": self.text.strip(),
            "duration": round(self.end - self.start, 2),
        }

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds to HH:MM:SS.mmm"""
        return TranscriptionResult.format_time(seconds)


class TranscriptionResult:
    """
    Transcription result that uses protobuf internally.

    This provides the same API as the original TranscriptionResult but
    stores data in a protobuf object for efficient serialization.
    """

    def __init__(self, proto: Optional[TranscriptionResultProto] = None, **kwargs):
        """Initialize with an existing protobuf or create a new one.

        For backward compatibility, also accepts keyword arguments.
        """
        self._proto = proto if proto is not None else TranscriptionResultProto()

        # Set creation timestamp if creating new
        if proto is None and "created_at" not in kwargs:
            self._proto.created_at = int(time.time())

        # Handle backward compatibility with keyword arguments
        if kwargs:
            if "filename" in kwargs:
                self.filename = kwargs["filename"]
            if "file_hash" in kwargs:
                self.file_hash = kwargs["file_hash"]
            if "language" in kwargs:
                self.language = kwargs["language"]
            if "full_text" in kwargs:
                self.full_text = kwargs["full_text"]
            if "segments" in kwargs:
                # Handle list of Segment objects
                for seg in kwargs["segments"]:
                    if hasattr(seg, "start"):
                        self.add_segment(seg.start, seg.end, seg.text)
                    else:
                        # Handle dict-like segments
                        self.add_segment(seg["start"], seg["end"], seg["text"])

    # Properties that map to protobuf fields
    @property
    def filename(self) -> str:
        return self._proto.filename

    @filename.setter
    def filename(self, value: str):
        self._proto.filename = value

    @property
    def file_hash(self) -> str:
        return self._proto.file_hash

    @file_hash.setter
    def file_hash(self, value: str):
        self._proto.file_hash = value

    @property
    def language(self) -> str:
        return self._proto.language

    @language.setter
    def language(self, value: str):
        self._proto.language = value

    @property
    def full_text(self) -> str:
        return self._proto.full_text

    @full_text.setter
    def full_text(self, value: str):
        self._proto.full_text = value

    @property
    def segments(self) -> List[SegmentProto]:
        """Return the protobuf segments directly."""
        return self._proto.segments

    @property
    def created_at(self) -> int:
        """Get creation timestamp."""
        return self._proto.created_at

    @created_at.setter
    def created_at(self, value: int):
        """Set creation timestamp."""
        self._proto.created_at = value

    def add_segment(self, start: float, end: float, text: str) -> SegmentProto:
        """Add a segment to the transcription.

        Args:
            start: Start time in seconds (will be converted to milliseconds)
            end: End time in seconds (will be converted to milliseconds)
            text: Segment text
        """
        segment = self._proto.segments.add()
        segment.start_ms = int(start * 1000)  # Convert seconds to milliseconds
        segment.end_ms = int(end * 1000)  # Convert seconds to milliseconds
        segment.text = text.strip()
        return segment

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds to HH:MM:SS.mmm"""
        td = timedelta(seconds=seconds)
        total_seconds = td.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    # Protobuf serialization methods
    def to_protobuf(self) -> TranscriptionResultProto:
        """Get the underlying protobuf object."""
        return self._proto

    def to_protobuf_bytes(self) -> bytes:
        """Serialize to protobuf bytes."""
        return self._proto.SerializeToString()

    @classmethod
    def from_protobuf_bytes(cls, data: bytes) -> "TranscriptionResult":
        """Deserialize from protobuf bytes."""
        proto = TranscriptionResultProto()
        proto.ParseFromString(data)
        return cls(proto)

    def to_protobuf_base64(self) -> str:
        """Serialize to base64-encoded protobuf."""
        return base64.b64encode(self.to_protobuf_bytes()).decode("utf-8")

    @classmethod
    def from_protobuf_base64(cls, data: str) -> "TranscriptionResult":
        """Deserialize from base64-encoded protobuf."""
        return cls.from_protobuf_bytes(base64.b64decode(data))

    # Format conversion methods
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "filename": self.filename,
            "file_hash": self.file_hash,
            "language": self.language,
            "segments": [
                {
                    "start": self.format_time(seg.start_ms / 1000.0),
                    "end": self.format_time(seg.end_ms / 1000.0),
                    "text": seg.text,
                    "duration": round((seg.end_ms - seg.start_ms) / 1000.0, 2),
                }
                for seg in self.segments
            ],
            "full_text": self.full_text,
            "created_at": self.created_at,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_csv(self) -> str:
        """Convert to CSV format."""
        output = io.StringIO()
        # Add metadata comments
        output.write(f"# File: {self.filename}\n")
        output.write(f"# SHA256: {self.file_hash}\n")
        output.write(f"# Language: {self.language}\n")
        output.write("Start,End,Duration,Text\n")
        for seg in self.segments:
            start_fmt = self.format_time(seg.start_ms / 1000.0)
            end_fmt = self.format_time(seg.end_ms / 1000.0)
            duration = round((seg.end_ms - seg.start_ms) / 1000.0, 3)
            # Escape quotes in text
            text = seg.text.replace('"', '""').strip()
            output.write(f'{start_fmt},{end_fmt},{duration},"{text}"\n')
        return output.getvalue()

    def to_table(self) -> str:
        """Convert to formatted table."""
        headers = ["Start", "End", "Duration", "Text"]
        rows = [
            (
                self.format_time(seg.start_ms / 1000.0),
                self.format_time(seg.end_ms / 1000.0),
                round((seg.end_ms - seg.start_ms) / 1000.0, 3),
                seg.text[:50] + "..." if len(seg.text) > 50 else seg.text,
            )
            for seg in self.segments
        ]
        table = tabulate(rows, headers=headers, tablefmt="grid")

        # Add file info header
        header = f"File: {self.filename}\nSHA256: {self.file_hash}\nLanguage: {self.language}\n\n"
        return header + table

    def to_toml(self) -> str:
        """Convert to TOML string."""
        # Build TOML manually to control order
        output = "[transcription]\n"
        output += f'filename = "{self.filename}"\n'
        output += f'file_hash = "{self.file_hash}"\n'
        output += f'language = "{self.language}"\n'
        output += f'full_text = """{self.full_text}"""\n'
        output += "\n"

        # Add segments as array of tables
        for i, seg in enumerate(self.segments):
            output += "[[segments]]\n"
            output += f"index = {i}\n"
            output += f'start = "{self.format_time(seg.start_ms / 1000.0)}"\n'
            output += f'end = "{self.format_time(seg.end_ms / 1000.0)}"\n'
            output += f"duration = {round((seg.end_ms - seg.start_ms) / 1000.0, 2)}\n"
            output += f'text = "{seg.text}"\n'
            output += "\n"

        return output.rstrip()

    def format(self, format_type: str) -> str:
        """Format output based on requested format."""
        if format_type == "text":
            return self.full_text
        elif format_type == "json":
            return self.to_json()
        elif format_type == "table":
            return self.to_table()
        elif format_type == "csv":
            return self.to_csv()
        elif format_type == "toml":
            return self.to_toml()
        elif format_type == "protobuf":
            return self.to_protobuf_base64()
        else:
            raise ValueError(f"Unknown format: {format_type}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionResult":
        """Create TranscriptionResult from dictionary."""
        result = cls()
        result.filename = data["filename"]
        result.file_hash = data["file_hash"]
        result.language = data["language"]
        result.full_text = data["full_text"]

        # Convert segments
        for seg_data in data.get("segments", []):
            # Parse time string back to seconds if needed
            if isinstance(seg_data.get("start"), str):
                # Parse HH:MM:SS.mmm format
                time_parts = seg_data["start"].split(":")
                start = (
                    float(time_parts[0]) * 3600
                    + float(time_parts[1]) * 60
                    + float(time_parts[2])
                )
                time_parts = seg_data["end"].split(":")
                end = (
                    float(time_parts[0]) * 3600
                    + float(time_parts[1]) * 60
                    + float(time_parts[2])
                )
            else:
                start = seg_data["start"]
                end = seg_data["end"]

            result.add_segment(start, end, seg_data["text"])

        return result

    @classmethod
    def from_json(cls, json_str: str) -> "TranscriptionResult":
        """Create TranscriptionResult from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_toml(cls, toml_str: str) -> "TranscriptionResult":
        """Create TranscriptionResult from TOML string."""
        data = toml.loads(toml_str)

        # TOML structure is slightly different
        transcription = data.get("transcription", {})
        segments_data = data.get("segments", [])

        result = cls()
        result.filename = transcription["filename"]
        result.file_hash = transcription["file_hash"]
        result.language = transcription["language"]
        result.full_text = transcription["full_text"]

        # Convert TOML segments format
        for seg in segments_data:
            # Parse time string to seconds
            time_parts = seg["start"].split(":")
            start = (
                float(time_parts[0]) * 3600
                + float(time_parts[1]) * 60
                + float(time_parts[2])
            )
            time_parts = seg["end"].split(":")
            end = (
                float(time_parts[0]) * 3600
                + float(time_parts[1]) * 60
                + float(time_parts[2])
            )

            result.add_segment(start, end, seg["text"])

        return result

    @classmethod
    def from_protobuf(cls, proto: TranscriptionResultProto) -> "TranscriptionResult":
        """Create from an existing protobuf object."""
        return cls(proto)

    # Backward compatibility - create a Segment-like interface
    def get_segments_list(self) -> List[Dict[str, Any]]:
        """Get segments as a list of dictionaries for backward compatibility."""
        return [
            {
                "start": seg.start_ms / 1000.0,
                "end": seg.end_ms / 1000.0,
                "text": seg.text,
                "to_dict": lambda s=seg: {
                    "start": self.format_time(s.start_ms / 1000.0),
                    "end": self.format_time(s.end_ms / 1000.0),
                    "text": s.text,
                },
            }
            for seg in self.segments
        ]


class AudioTranscriber:
    """Main transcription class for audio files using protobuf."""

    def __init__(self, model_name: str = "tiny"):
        """Initialize transcriber with specified model."""
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            # Capture model loading output
            with io.StringIO() as buf, redirect_stdout(buf):
                self._model = whisper.load_model(self.model_name)
        return self._model

    @staticmethod
    def calculate_file_hash(filepath: str) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def transcribe_file(
        self, filepath: str, verbose: bool = False
    ) -> TranscriptionResult:
        """
        Transcribe an audio file and return structured result.

        Args:
            filepath: Path to the audio file
            verbose: Unused, kept for backward compatibility

        Returns:
            TranscriptionResult with transcription data
        """
        # Verify file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Calculate file hash
        file_hash = self.calculate_file_hash(filepath)

        # Transcribe with Whisper
        result = self.model.transcribe(filepath)

        # Create protobuf-based result
        transcription = TranscriptionResult()
        transcription.filename = os.path.basename(filepath)
        transcription.file_hash = file_hash
        transcription.language = result.get("language", "unknown")
        transcription.full_text = result["text"].strip()
        # created_at is automatically set in __init__

        # Add segments
        for segment in result.get("segments", []):
            transcription.add_segment(
                start=segment["start"], end=segment["end"], text=segment["text"]
            )

        return transcription


def create_extended_result(
    result: TranscriptionResult,
    feed_url: Optional[str] = None,
    item_id: Optional[str] = None,
    title: Optional[str] = None,
    audio_url: Optional[str] = None,
    published: Optional[str] = None,
) -> ExtendedTranscriptionResult:
    """Create an ExtendedTranscriptionResult with optional feed metadata."""
    extended = ExtendedTranscriptionResult()
    extended.transcription.CopyFrom(result.to_protobuf())

    if any([feed_url, item_id, title, audio_url, published]):
        if feed_url:
            extended.feed_metadata.feed_url = feed_url
        if item_id:
            extended.feed_metadata.item_id = item_id
        if title:
            extended.feed_metadata.title = title
        if audio_url:
            extended.feed_metadata.audio_url = audio_url
        if published:
            extended.feed_metadata.published = published

    return extended
