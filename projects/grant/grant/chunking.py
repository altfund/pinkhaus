import hashlib
from typing import List, Dict, Any
import tiktoken
from dataclasses import dataclass

from pinkhaus_models import TranscriptionMetadata, TranscriptionSegment


@dataclass
class TextChunk:
    """A chunk of text with metadata for vector storage."""

    id: str
    text: str
    metadata: Dict[str, Any]

    def __hash__(self):
        return hash(self.id)


class PodcastChunker:
    """Chunk podcast transcriptions using segment boundaries."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        encoding_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk_transcription(
        self, metadata: TranscriptionMetadata, segments: List[TranscriptionSegment]
    ) -> List[TextChunk]:
        """Chunk a transcription using segment boundaries."""

        if not segments:
            # Fallback to simple text chunking if no segments
            return self._chunk_text(metadata.full_text, metadata)

        # Use segment-aware chunking
        return self._chunk_with_segments(metadata, segments)

    def _chunk_with_segments(
        self, metadata: TranscriptionMetadata, segments: List[TranscriptionSegment]
    ) -> List[TextChunk]:
        """Create chunks that respect segment boundaries."""

        chunks = []
        current_segments = []
        current_tokens = 0
        chunk_index = 0

        for segment in segments:
            segment_tokens = len(self.encoding.encode(segment.text))

            # If adding this segment would exceed chunk size, create a chunk
            if current_tokens + segment_tokens > self.chunk_size and current_segments:
                chunk = self._create_chunk_from_segments(
                    metadata, current_segments, chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1

                # Keep some overlap
                overlap_segments = self._get_overlap_segments(current_segments)
                current_segments = overlap_segments
                current_tokens = sum(
                    len(self.encoding.encode(s.text)) for s in overlap_segments
                )

            current_segments.append(segment)
            current_tokens += segment_tokens

        # Create final chunk
        if current_segments:
            chunk = self._create_chunk_from_segments(
                metadata, current_segments, chunk_index
            )
            chunks.append(chunk)

        return chunks

    def _chunk_text(
        self, text: str, metadata: TranscriptionMetadata
    ) -> List[TextChunk]:
        """Simple text chunking when segments aren't available."""

        tokens = self.encoding.encode(text)
        chunks = []

        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunk_metadata = {
                "transcription_id": metadata.id,
                "chunk_index": chunk_index,
                "token_count": len(chunk_tokens),
                "title": metadata.feed_item_title,
                "published": metadata.feed_item_published.isoformat()
                if metadata.feed_item_published
                else None,
                "published_timestamp": int(metadata.feed_item_published.timestamp())
                if metadata.feed_item_published
                else None,
                "feed_url": metadata.feed_url,
                "char_start": len(self.encoding.decode(tokens[:start])),
                "char_end": len(self.encoding.decode(tokens[:end])),
            }

            chunk_id = self._generate_chunk_id(
                metadata.id or 0, chunk_index, chunk_text
            )

            chunks.append(
                TextChunk(id=chunk_id, text=chunk_text, metadata=chunk_metadata)
            )

            # Move with overlap
            start = end - self.chunk_overlap if end < len(tokens) else end
            chunk_index += 1

        return chunks

    def _create_chunk_from_segments(
        self,
        metadata: TranscriptionMetadata,
        segments: List[TranscriptionSegment],
        chunk_index: int,
    ) -> TextChunk:
        """Create a chunk from a list of segments."""

        text = " ".join(seg.text.strip() for seg in segments)

        chunk_metadata = {
            "transcription_id": metadata.id,
            "chunk_index": chunk_index,
            "segment_count": len(segments),
            "first_segment_index": segments[0].segment_index,
            "last_segment_index": segments[-1].segment_index,
            "start_time": segments[0].start_time,
            "end_time": segments[-1].end_time,
            "duration": segments[-1].end_time - segments[0].start_time,
            "title": metadata.feed_item_title,
            "published": metadata.feed_item_published.isoformat()
            if metadata.feed_item_published
            else None,
            "published_timestamp": int(metadata.feed_item_published.timestamp())
            if metadata.feed_item_published
            else None,
            "feed_url": metadata.feed_url,
            "token_count": len(self.encoding.encode(text)),
        }

        chunk_id = self._generate_chunk_id(metadata.id or 0, chunk_index, text)

        return TextChunk(id=chunk_id, text=text, metadata=chunk_metadata)

    def _get_overlap_segments(
        self, segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """Get segments for overlap based on token count."""

        overlap_segments = []
        overlap_tokens = 0

        # Work backwards from the end
        for segment in reversed(segments):
            segment_tokens = len(self.encoding.encode(segment.text))
            if overlap_tokens + segment_tokens <= self.chunk_overlap:
                overlap_segments.insert(0, segment)
                overlap_tokens += segment_tokens
            else:
                break

        return overlap_segments

    def _generate_chunk_id(
        self, transcription_id: int, chunk_index: int, text: str
    ) -> str:
        """Generate a unique ID for a chunk."""

        content = f"{transcription_id}:{chunk_index}:{text[:100]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
