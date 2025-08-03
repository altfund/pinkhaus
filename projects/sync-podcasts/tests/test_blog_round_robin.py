"""Test round-robin functionality with mixed podcast and blog feeds."""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time

from sync_podcasts.sync import PodcastSyncer, SyncConfig
from beige_book import TranscriptionService
from pinkhaus_models import TranscriptionDatabase


class TestBlogRoundRobin:
    """Test round-robin processing with blog feeds."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_server(self):
        """Start a local HTTP server to serve test feeds."""
        # Change to fixtures directory
        fixtures_dir = Path(__file__).parent / "fixtures"
        os.chdir(fixtures_dir)

        # Create server
        server = HTTPServer(("localhost", 8000), SimpleHTTPRequestHandler)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        # Give server time to start
        time.sleep(0.5)

        yield server

        server.shutdown()

    def test_blog_feed_parsing(self):
        """Test that blog feed can be parsed correctly."""
        from beige_book.feed_parser import FeedParser

        parser = FeedParser()

        # Parse the test blog feed directly
        fixtures_dir = Path(__file__).parent / "fixtures"
        feed_path = fixtures_dir / "feed4.xml"

        # Read and parse as if from URL
        import feedparser
        feed = feedparser.parse(str(feed_path))

        assert len(feed.entries) == 3

        # Check first entry
        entry = feed.entries[0]
        assert "Future of AI" in entry.title
        assert hasattr(entry, "content") or "content" in entry

        # Parse using our FeedParser
        items = parser.parse_feed(f"file://{feed_path}")
        blog_items = [item for item in items if item.feed_type == "blog"]

        assert len(blog_items) == 3
        assert all(item.content for item in blog_items)
        assert "Artificial intelligence" in blog_items[0].content

    def test_mixed_feed_round_robin(self, temp_dir, test_server):
        """Test round-robin processing with mixed podcast and blog feeds."""
        # Create test database
        db_path = os.path.join(temp_dir, "test.db")

        # Create syncer with test configuration
        config = SyncConfig(
            feeds_path=str(Path(__file__).parent / "fixtures" / "test_feeds_with_blog.toml"),
            db_path=db_path,
            vector_store_path=os.path.join(temp_dir, "vector_store"),
            round_robin=True,
            days_back=365,  # Get all test items
            verbose=True
        )

        syncer = PodcastSyncer(config)

        # Process items
        processed_count = 0
        max_iterations = 20  # Prevent infinite loop

        for i in range(max_iterations):
            if syncer.process_one_podcast():
                processed_count += 1
            else:
                break

        # Verify items were processed
        assert processed_count > 0

        # Check database
        db = TranscriptionDatabase(db_path)
        transcriptions = db.get_all_transcriptions()

        # Should have both podcast and blog items
        feed_urls = {t.feed_url for t in transcriptions}
        assert len(feed_urls) >= 2  # At least 2 different feeds

        # Check for blog content
        blog_transcriptions = [
            t for t in transcriptions
            if "blog" in t.feed_url.lower() or "feed4" in t.feed_url
        ]

        if blog_transcriptions:
            # Verify blog content was segmented properly
            blog_id = blog_transcriptions[0].id
            segments = db.get_segments_for_transcription(blog_id)
            assert len(segments) > 0

            # Check segment has proper timing
            first_segment = segments[0]
            assert first_segment.start_time == 0.0
            assert first_segment.end_time > 0.0
            assert first_segment.duration > 0.0
            assert len(first_segment.text) > 0

    def test_blog_content_processing(self, temp_dir):
        """Test that blog content is processed correctly."""
        from beige_book.blog_processor import BlogProcessor

        processor = BlogProcessor(words_per_segment=50, words_per_minute=120)

        # Test HTML content
        html_content = """
        <h1>Test Blog Post</h1>
        <p>This is a test blog post with some content. It has multiple paragraphs
        and should be segmented properly.</p>
        <p>The blog processor should extract text from HTML, segment it into
        chunks of approximately 50 words, and calculate appropriate timing
        information for each segment.</p>
        """

        result = processor.process_blog_content(
            content=html_content,
            filename="test_blog.html",
            title="Test Blog Post"
        )

        assert len(result.segments) >= 1
        assert result.filename == "test_blog.html"
        assert result.language == "en"

        # Check timing calculation (10 words = 5 seconds)
        total_words = len(result.full_text.split())
        expected_duration = (total_words / 120) * 60  # seconds
        actual_duration = result.segments[-1].end

        # Should be close to expected
        assert abs(actual_duration - expected_duration) < 5.0