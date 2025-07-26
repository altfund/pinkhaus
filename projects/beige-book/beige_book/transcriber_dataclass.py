"""
Transcription library for audio files using OpenAI's Whisper model.
"""

import os
import json
import hashlib
import io
from contextlib import redirect_stdout
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import timedelta
from tabulate import tabulate
import whisper
import toml


@dataclass
class Segment:
    """Represents a transcription segment with timing information"""

    start: float
    end: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with formatted times"""
        return {
            "start": self.format_time(self.start),
            "end": self.format_time(self.end),
            "text": self.text.strip(),
            "duration": round(self.end - self.start, 2),
        }

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds to HH:MM:SS.mmm"""
        td = timedelta(seconds=seconds)
        total_seconds = td.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


@dataclass
class TranscriptionResult:
    """Complete transcription result with metadata"""

    filename: str
    file_hash: str
    language: str
    segments: List[Segment]
    full_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "filename": self.filename,
            "file_hash": self.file_hash,
            "language": self.language,
            "segments": [seg.to_dict() for seg in self.segments],
            "full_text": self.full_text,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_csv_rows(self) -> List[List[str]]:
        """Convert to CSV-ready rows"""
        rows = []
        for seg in self.segments:
            rows.append(
                [
                    seg.format_time(seg.start),
                    seg.format_time(seg.end),
                    f"{seg.end - seg.start:.2f}",
                    seg.text.strip(),
                ]
            )
        return rows

    def to_csv(self) -> str:
        """Convert to CSV format with metadata comments"""
        output = f"# File: {self.filename}\n"
        output += f"# SHA256: {self.file_hash}\n"
        output += f"# Language: {self.language}\n"
        output += "Start,End,Duration,Text\n"
        for row in self.to_csv_rows():
            # Properly escape CSV fields
            escaped_text = row[3].replace('"', '""')
            output += f'{row[0]},{row[1]},{row[2]},"{escaped_text}"\n'
        return output

    def to_table(self) -> str:
        """Convert to formatted table"""
        headers = ["Start", "End", "Duration", "Text"]
        rows = self.to_csv_rows()
        table = tabulate(rows, headers=headers, tablefmt="grid")
        # Add file info header
        header = f"File: {self.filename}\nSHA256: {self.file_hash}\nLanguage: {self.language}\n\n"
        return header + table

    def to_toml(self) -> str:
        """Convert to TOML string"""
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
            output += f'start = "{seg.format_time(seg.start)}"\n'
            output += f'end = "{seg.format_time(seg.end)}"\n'
            output += f"duration = {round(seg.end - seg.start, 2)}\n"
            output += f'text = "{seg.text.strip()}"\n'
            output += "\n"

        return output.rstrip()

    def format(self, format_type: str) -> str:
        """Format output based on requested format"""
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
            # Return base64 encoded protobuf for text representation
            if hasattr(self, "to_protobuf_base64"):
                return self.to_protobuf_base64()
            else:
                raise ValueError("Protobuf support not available")
        else:
            raise ValueError(f"Unknown format: {format_type}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionResult":
        """Create TranscriptionResult from dictionary"""
        # Convert segment dictionaries to Segment objects
        segments = []
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

            segments.append(Segment(start=start, end=end, text=seg_data["text"]))

        return cls(
            filename=data["filename"],
            file_hash=data["file_hash"],
            language=data["language"],
            segments=segments,
            full_text=data["full_text"],
        )

    @classmethod
    def from_json(cls, json_str: str) -> "TranscriptionResult":
        """Create TranscriptionResult from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_toml(cls, toml_str: str) -> "TranscriptionResult":
        """Create TranscriptionResult from TOML string"""
        data = toml.loads(toml_str)

        # TOML structure is slightly different
        transcription = data.get("transcription", {})
        segments_data = data.get("segments", [])

        # Convert TOML segments format
        segments = []
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

            segments.append(Segment(start=start, end=end, text=seg["text"]))

        return cls(
            filename=transcription["filename"],
            file_hash=transcription["file_hash"],
            language=transcription["language"],
            segments=segments,
            full_text=transcription["full_text"],
        )


class AudioTranscriber:
    """Main transcription class for audio files"""

    def __init__(self, model_name: str = "tiny"):
        """Initialize transcriber with specified model"""
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy load the model"""
        if self._model is None:
            self._model = whisper.load_model(self.model_name)
        return self._model

    @staticmethod
    def calculate_file_hash(filepath: str) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def transcribe_file(
        self, filename: str, verbose: bool = False
    ) -> TranscriptionResult:
        """
        Transcribe an audio file using Whisper.

        Args:
            filename: Path to the audio file
            verbose: Whether to show progress output

        Returns:
            TranscriptionResult with segments and metadata

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' not found")

        # Calculate file hash before transcription
        file_hash = self.calculate_file_hash(filename)

        # Get full result with segments
        if verbose:
            result = self.model.transcribe(audio=filename, verbose=True)
        else:
            # Suppress output for non-verbose mode
            with redirect_stdout(io.StringIO()):
                result = self.model.transcribe(audio=filename, verbose=False)

        # Create structured result
        segments = []
        for seg in result["segments"]:
            segments.append(
                Segment(start=seg["start"], end=seg["end"], text=seg["text"])
            )

        transcription = TranscriptionResult(
            filename=os.path.basename(filename),
            file_hash=file_hash,
            language=result.get("language", "unknown"),
            segments=segments,
            full_text=result["text"].strip(),
        )

        return transcription
