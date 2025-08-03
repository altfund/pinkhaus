"""
Transcription library using betterproto-generated Protocol Buffers.
"""
import logging

# Extended result support
from .proto_models import ExtendedTranscriptionResult, FeedMetadata
import os
import json
import hashlib
import io
import base64
import time
from contextlib import redirect_stdout
from typing import Dict, Any, Optional
from datetime import timedelta
from tabulate import tabulate
import whisper

from .proto_models import TranscriptionResult, Segment

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Main transcription class for audio files using betterproto."""

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
        self, filepath: str,
            verbose: bool = False,
            enable_diarization: bool = False,
            hf_token: str = None,
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

        # Create betterproto-based result
        transcription = TranscriptionResult(
            filename=os.path.basename(filepath),
            file_hash=file_hash,
            language=result.get("language", "unknown"),
            full_text=result["text"].strip(),
            created_at=int(time.time()),
        )

        # Add segments
        for segment in result.get("segments", []):
            transcription.segments.append(
                Segment(
                    start_ms=int(segment["start"] * 1000),
                    end_ms=int(segment["end"] * 1000),
                    text=segment["text"].strip(),
                )
            )

        # Optionally add speaker diarization
        if enable_diarization:
            try:
                from .speaker_diarizer import SpeakerDiarizer

                diarizer = SpeakerDiarizer(auth_token=hf_token)
                # Use mock mode if pyannote not available
                try:
                    diarization = diarizer.diarize_file(filepath, use_mock=False)
                except ImportError:
                    logger.warning("pyannote-audio not available, using mock diarization");
                    diarization = diarizer.diarize_file(filepath, use_mock=True)

                # Align speakers with segments
                segments_list = transcription.get_segments_list()
                enhanced_segments = diarizer.align_with_transcription(
                    diarization, segments_list
                )

                # Update protobuf segments with speaker info
                for i, (seg, enhanced) in enumerate(zip(transcription.segments, enhanced_segments)):
                    if 'speaker' in enhanced:
                        seg.speaker = enhanced['speaker']
                    if 'confidence' in enhanced:
                        seg.confidence = enhanced.get('confidence', 1.0)

                # Update metadata
                transcription.num_speakers = diarization.num_speakers
                transcription.has_speaker_labels = True

            except Exception as e:
                print(f"Warning: Speaker diarization failed: {e}")
                # Continue without diarization


        return transcription


# Add extension methods to TranscriptionResult for backward compatibility
def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS.mmm"""
    td = timedelta(seconds=seconds)
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary format."""
    return {
        "filename": self.filename,
        "file_hash": self.file_hash,
        "language": self.language,
        "segments": [
            {
                "start": format_time(seg.start_ms / 1000.0),
                "end": format_time(seg.end_ms / 1000.0),
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
        start_fmt = format_time(seg.start_ms / 1000.0)
        end_fmt = format_time(seg.end_ms / 1000.0)
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
            format_time(seg.start_ms / 1000.0),
            format_time(seg.end_ms / 1000.0),
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
        output += f'start = "{format_time(seg.start_ms / 1000.0)}"\n'
        output += f'end = "{format_time(seg.end_ms / 1000.0)}"\n'
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
        return base64.b64encode(bytes(self)).decode("utf-8")
    else:
        raise ValueError(f"Unknown format: {format_type}")


# Add methods to TranscriptionResult class
TranscriptionResult.to_dict = to_dict
TranscriptionResult.to_json = to_json
TranscriptionResult.to_csv = to_csv
TranscriptionResult.to_table = to_table
TranscriptionResult.to_toml = to_toml
TranscriptionResult.format = format


def create_extended_result(
    result: TranscriptionResult,
    feed_url: Optional[str] = None,
    item_id: Optional[str] = None,
    title: Optional[str] = None,
    audio_url: Optional[str] = None,
    published: Optional[str] = None,
) -> ExtendedTranscriptionResult:
    """Create an ExtendedTranscriptionResult with optional feed metadata."""
    extended = ExtendedTranscriptionResult(transcription=result)

    if any([feed_url, item_id, title, audio_url, published]):
        if not extended.feed_metadata:
            extended.feed_metadata = FeedMetadata()
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


# Backward compatibility - provide access to Segment at module level
__all__ = [
    "AudioTranscriber",
    "TranscriptionResult",
    "Segment",
    "create_extended_result",
]
