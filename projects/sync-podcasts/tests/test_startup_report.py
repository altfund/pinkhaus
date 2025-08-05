"""Test startup reporting functionality."""

import os
import tempfile
import logging
from unittest.mock import Mock, patch, MagicMock
import pytest

from sync_podcasts.sync import PodcastSyncer, SyncConfig
from pinkhaus_models import TranscriptionDatabase
from beige_book.feed_parser import FeedItem
from datetime import datetime


class TestStartupReport:
    """Test the startup report functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def temp_feeds_file(self):
        """Create a temporary feeds.toml file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
[feeds]
rss = [
    "https://example.com/feed1.rss",
    "https://example.com/feed2.rss"
]
""")
            feeds_path = f.name
        yield feeds_path
        # Cleanup
        if os.path.exists(feeds_path):
            os.unlink(feeds_path)
    
    def test_startup_report_with_unprocessed_items(self, temp_db, temp_feeds_file, caplog):
        """Test startup report shows unprocessed items."""
        # Create some mock feed items
        mock_items_feed1 = [
            FeedItem(
                feed_url="https://example.com/feed1.rss",
                item_id="ep1",
                title="Episode 1",
                audio_url="https://example.com/ep1.mp3",
                published=datetime.now()
            ),
            FeedItem(
                feed_url="https://example.com/feed1.rss",
                item_id="ep2",
                title="Episode 2",
                audio_url="https://example.com/ep2.mp3",
                published=datetime.now()
            ),
        ]
        
        mock_items_feed2 = [
            FeedItem(
                feed_url="https://example.com/feed2.rss",
                item_id="news1",
                title="News 1",
                audio_url="https://example.com/news1.mp3",
                published=datetime.now()
            ),
        ]
        
        config = SyncConfig(
            feeds_path=temp_feeds_file,
            db_path=temp_db,
            skip_validation=True
        )
        
        # Mock the feed parser
        def mock_parse_feed(feed_url):
            if feed_url == "https://example.com/feed1.rss":
                return mock_items_feed1
            elif feed_url == "https://example.com/feed2.rss":
                return mock_items_feed2
            return []
        
        with patch('sync_podcasts.sync.TranscriptionService'):
            with patch('sync_podcasts.sync.OllamaClient'):
                with patch('beige_book.feed_parser.FeedParser.parse_feed', side_effect=mock_parse_feed):
                    with caplog.at_level(logging.INFO):
                        syncer = PodcastSyncer(config)
        
        # Check the log output
        assert "STARTUP REPORT" in caplog.text
        assert "No failed items found" in caplog.text
        assert "Unprocessed items found" in caplog.text
        assert "https://example.com/feed1.rss: 2/2 items to process" in caplog.text
        assert "https://example.com/feed2.rss: 1/1 items to process" in caplog.text
        assert "Total unprocessed items: 3" in caplog.text
    
    def test_startup_report_with_failed_items(self, temp_db, temp_feeds_file, caplog):
        """Test startup report shows failed items."""
        # Setup database with failed items
        db = TranscriptionDatabase(temp_db)
        db.create_tables()
        
        # Add some failed items
        db.record_failed_item(
            feed_url="https://example.com/feed1.rss",
            feed_item_id="ep1",
            error_type="DownloadError",
            error_message="Connection timeout",
            feed_item_title="Episode 1"
        )
        db.record_failed_item(
            feed_url="https://example.com/feed1.rss",
            feed_item_id="ep2",
            error_type="TranscriptionError",
            error_message="Model failed",
            feed_item_title="Episode 2"
        )
        
        config = SyncConfig(
            feeds_path=temp_feeds_file,
            db_path=temp_db,
            skip_validation=True
        )
        
        with patch('sync_podcasts.sync.TranscriptionService'):
            with patch('sync_podcasts.sync.OllamaClient'):
                with patch('beige_book.feed_parser.FeedParser.parse_feed', return_value=[]):
                    with caplog.at_level(logging.INFO):
                        syncer = PodcastSyncer(config)
        
        # Check the log output
        assert "Failed items found in 'failed_items' table" in caplog.text
        assert "https://example.com/feed1.rss: 2 failed items" in caplog.text
    
    def test_startup_report_all_up_to_date(self, temp_db, temp_feeds_file, caplog):
        """Test startup report when all items are processed."""
        # Setup database with processed items
        db = TranscriptionDatabase(temp_db)
        db.create_tables()
        
        # Mock that all items are already processed
        config = SyncConfig(
            feeds_path=temp_feeds_file,
            db_path=temp_db,
            skip_validation=True
        )
        
        # Create empty feed items (simulating all processed)
        with patch('sync_podcasts.sync.TranscriptionService'):
            with patch('sync_podcasts.sync.OllamaClient'):
                with patch('beige_book.feed_parser.FeedParser.parse_feed', return_value=[]):
                    with caplog.at_level(logging.INFO):
                        syncer = PodcastSyncer(config)
        
        # Check the log output
        assert "All items up to date!" in caplog.text
        assert "No failed items found" in caplog.text
    
    def test_startup_report_handles_feed_errors(self, temp_db, temp_feeds_file, caplog):
        """Test startup report handles feed parsing errors gracefully."""
        config = SyncConfig(
            feeds_path=temp_feeds_file,
            db_path=temp_db,
            skip_validation=True
        )
        
        # Mock feed parser to raise an error
        def mock_parse_feed(feed_url):
            raise Exception("Feed unavailable")
        
        with patch('sync_podcasts.sync.TranscriptionService'):
            with patch('sync_podcasts.sync.OllamaClient'):
                with patch('beige_book.feed_parser.FeedParser.parse_feed', side_effect=mock_parse_feed):
                    with caplog.at_level(logging.INFO):
                        syncer = PodcastSyncer(config)
        
        # Check that errors are handled gracefully
        assert "Error checking feed" in caplog.text
        assert "Feed unavailable" in caplog.text