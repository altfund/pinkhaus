"""
Blog content processor for converting blog posts to transcription-like format.
"""

import re
import hashlib
from typing import List, Tuple
import trafilatura

from pinkhaus_models import TranscriptionResult, Segment


class BlogProcessor:
    """Process blog content into transcription-like segments."""

    def __init__(self, words_per_segment: int = 50, words_per_minute: int = 120):
        """
        Initialize blog processor.

        Args:
            words_per_segment: Number of words per segment (default 50)
            words_per_minute: Reading speed in words per minute (default 120, which gives 10 words = 5 seconds)
        """
        self.words_per_segment = words_per_segment
        self.words_per_minute = words_per_minute
        self.seconds_per_word = 60.0 / words_per_minute

    def extract_text_from_html(self, html_content: str) -> str:
        """
        Extract plain text from HTML content using trafilatura.

        Args:
            html_content: HTML content string

        Returns:
            Plain text extracted from HTML
        """
        # Use trafilatura to extract text
        # It automatically handles HTML entities, scripts, styles, etc.
        text = trafilatura.extract(html_content, 
                                  include_comments=False,
                                  include_tables=True,
                                  no_fallback=False)
        
        if text is None:
            # Fallback to basic extraction if trafilatura fails
            text = trafilatura.extract(html_content, 
                                     favor_precision=False,
                                     favor_recall=True)
            if text is None:
                # Last resort - strip all HTML tags
                text = re.sub(r'<[^>]+>', ' ', html_content)
                text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def segment_text(self, text: str) -> List[str]:
        """
        Segment text into chunks of approximately words_per_segment words.

        Args:
            text: Text to segment

        Returns:
            List of text segments
        """
        # Split into sentences first to avoid breaking mid-sentence
        sentences = re.split(r"(?<=[.!?])\s+", text)

        segments = []
        current_segment = []
        current_word_count = 0

        for sentence in sentences:
            words = sentence.split()
            sentence_word_count = len(words)

            # If adding this sentence would exceed our limit, start a new segment
            if (
                current_word_count + sentence_word_count > self.words_per_segment
                and current_segment
            ):
                segments.append(" ".join(current_segment))
                current_segment = [sentence]
                current_word_count = sentence_word_count
            else:
                current_segment.append(sentence)
                current_word_count += sentence_word_count

        # Add the last segment
        if current_segment:
            segments.append(" ".join(current_segment))

        return segments

    def calculate_timing(self, segments: List[str]) -> List[Tuple[float, float]]:
        """
        Calculate start and end times for each segment based on reading speed.

        Args:
            segments: List of text segments

        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        timings = []
        current_time = 0.0

        for segment in segments:
            word_count = len(segment.split())
            duration = word_count * self.seconds_per_word
            end_time = current_time + duration

            timings.append((current_time, end_time))
            current_time = end_time

        return timings

    def process_blog_content(
        self, content: str, filename: str, title: str = ""
    ) -> TranscriptionResult:
        """
        Process blog content into a TranscriptionResult.

        Args:
            content: Blog content (HTML or plain text)
            filename: Filename or URL for the blog post
            title: Title of the blog post

        Returns:
            TranscriptionResult object
        """
        # Extract plain text if content appears to be HTML
        if "<" in content and ">" in content:
            text = self.extract_text_from_html(content)
        else:
            text = content

        # Add title to the beginning if provided
        if title:
            text = f"{title}\n\n{text}"

        # Generate file hash
        file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Segment the text
        text_segments = self.segment_text(text)

        # Calculate timings
        timings = self.calculate_timing(text_segments)

        # Create Segment objects
        segments = []
        for i, (segment_text, (start_time, end_time)) in enumerate(
            zip(text_segments, timings)
        ):
            segment = Segment(start=start_time, end=end_time, text=segment_text.strip())
            segments.append(segment)

        # Create TranscriptionResult
        result = TranscriptionResult(
            filename=filename,
            file_hash=file_hash,
            language="en",  # Assume English for now
            segments=segments,
            full_text=text,
        )

        return result
