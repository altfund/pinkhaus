"""
Blog content processor for converting blog posts to transcription-like format.
"""

import re
import hashlib
from typing import List, Tuple
from html import unescape
from html.parser import HTMLParser

from pinkhaus_models import TranscriptionResult, Segment


class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML content."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.in_script = False
        self.in_style = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str]]):
        if tag == "script":
            self.in_script = True
        elif tag == "style":
            self.in_style = True
        elif tag in ["p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li"]:
            # Add spacing for block elements
            if self.text_parts and self.text_parts[-1] != "\n\n":
                self.text_parts.append("\n\n")

    def handle_endtag(self, tag: str):
        if tag == "script":
            self.in_script = False
        elif tag == "style":
            self.in_style = False

    def handle_data(self, data: str):
        if not self.in_script and not self.in_style:
            text = data.strip()
            if text:
                self.text_parts.append(text)

    def get_text(self) -> str:
        return " ".join(self.text_parts)


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
        Extract plain text from HTML content.

        Args:
            html_content: HTML content string

        Returns:
            Plain text extracted from HTML
        """
        # First unescape HTML entities
        html_content = unescape(html_content)

        # Use our HTML parser to extract text
        parser = HTMLTextExtractor()
        parser.feed(html_content)
        text = parser.get_text()

        # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text.strip()

    def segment_text(self, text: str) -> List[str]:
        """
        Segment text into chunks of approximately words_per_segment words.

        Args:
            text: Text to segment

        Returns:
            List of text segments
        """
        # Split into sentences first to avoid breaking mid-sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)

        segments = []
        current_segment = []
        current_word_count = 0

        for sentence in sentences:
            words = sentence.split()
            sentence_word_count = len(words)

            # If adding this sentence would exceed our limit, start a new segment
            if current_word_count + sentence_word_count > self.words_per_segment and current_segment:
                segments.append(' '.join(current_segment))
                current_segment = [sentence]
                current_word_count = sentence_word_count
            else:
                current_segment.append(sentence)
                current_word_count += sentence_word_count

        # Add the last segment
        if current_segment:
            segments.append(' '.join(current_segment))

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

    def process_blog_content(self, content: str, filename: str, title: str = "") -> TranscriptionResult:
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
        if '<' in content and '>' in content:
            text = self.extract_text_from_html(content)
        else:
            text = content

        # Add title to the beginning if provided
        if title:
            text = f"{title}\n\n{text}"

        # Generate file hash
        file_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

        # Segment the text
        text_segments = self.segment_text(text)

        # Calculate timings
        timings = self.calculate_timing(text_segments)

        # Create Segment objects
        segments = []
        for i, (segment_text, (start_time, end_time)) in enumerate(zip(text_segments, timings)):
            segment = Segment(
                start=start_time,
                end=end_time,
                text=segment_text.strip()
            )
            segments.append(segment)

        # Create TranscriptionResult
        result = TranscriptionResult(
            filename=filename,
            file_hash=file_hash,
            language="en",  # Assume English for now
            segments=segments,
            full_text=text
        )

        return result