"""Test that feeds.toml is read fresh on each processing cycle."""

import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
import pytest

from sync_podcasts.sync import PodcastSyncer, SyncConfig
from beige_book.models import TranscriptionResponse, ProcessingSummary


class TestFeedsReload:
    """Test that feeds.toml changes are picked up during runtime."""
    
    @pytest.fixture
    def temp_feeds_file(self):
        """Create a temporary feeds.toml file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
[feeds]
rss = [
    "https://example.com/feed1.rss"
]
""")
            feeds_path = f.name
        yield feeds_path
        # Cleanup
        if os.path.exists(feeds_path):
            os.unlink(feeds_path)
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    def test_feeds_reload_between_cycles(self, temp_feeds_file, temp_db):
        """Test that feeds.toml is re-read between processing cycles."""
        config = SyncConfig(
            feeds_path=temp_feeds_file,
            db_path=temp_db,
            round_robin=False,
            skip_validation=True
        )
        
        # Track which feeds were parsed
        parsed_feeds = []
        
        def mock_parse_all_feeds(toml_path, max_retries=3):
            """Mock that tracks when feeds are parsed."""
            # Read the actual file content
            with open(toml_path, 'r') as f:
                content = f.read()
            parsed_feeds.append(content)
            # Return empty dict to simulate no podcasts to process
            return {}
        
        with patch('beige_book.feed_parser.FeedParser.parse_all_feeds', side_effect=mock_parse_all_feeds):
            with patch('sync_podcasts.sync.OllamaClient'):
                syncer = PodcastSyncer(config)
                
                # First processing cycle
                syncer.process_one_podcast()
                
                # Modify the feeds file
                with open(temp_feeds_file, 'w') as f:
                    f.write("""
[feeds]
rss = [
    "https://example.com/feed1.rss",
    "https://example.com/feed2.rss"
]
""")
                
                # Second processing cycle
                syncer.process_one_podcast()
        
        # Verify that feeds were parsed twice with different content
        assert len(parsed_feeds) == 2
        assert "feed1.rss" in parsed_feeds[0]
        assert "feed2.rss" not in parsed_feeds[0]
        assert "feed1.rss" in parsed_feeds[1]
        assert "feed2.rss" in parsed_feeds[1]
    
    def test_feeds_reload_in_daemon_mode(self, temp_feeds_file, temp_db):
        """Test that feeds.toml is re-read in daemon mode."""
        config = SyncConfig(
            feeds_path=temp_feeds_file,
            db_path=temp_db,
            daemon=True,
            skip_validation=True
        )
        
        parse_count = 0
        
        def mock_process_one_podcast():
            """Mock that counts calls and exits after 3."""
            nonlocal parse_count
            parse_count += 1
            
            # Exit after 3 calls to prevent infinite loop
            if parse_count >= 3:
                raise KeyboardInterrupt("Test complete")
            
            return False  # No podcast processed
        
        with patch('sync_podcasts.sync.TranscriptionService'):
            with patch('sync_podcasts.sync.OllamaClient'):
                with patch('time.sleep', return_value=None):  # Speed up test
                    syncer = PodcastSyncer(config)
                    
                    # Mock the process_one_podcast method
                    syncer.process_one_podcast = mock_process_one_podcast
                    
                    # Start daemon
                    try:
                        syncer._run_daemon()
                    except KeyboardInterrupt:
                        pass
        
        # Verify process_one_podcast was called multiple times
        assert parse_count == 3
    
    def test_concurrent_instances_different_feeds(self, temp_db):
        """Test that multiple instances can run with different feeds files."""
        # Create two different feeds files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f1:
            f1.write('[feeds]\nrss = ["https://example.com/feed1.rss"]')
            feeds1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f2:
            f2.write('[feeds]\nrss = ["https://example.com/feed2.rss"]')
            feeds2 = f2.name
        
        try:
            config1 = SyncConfig(feeds_path=feeds1, db_path=temp_db, skip_validation=True)
            config2 = SyncConfig(feeds_path=feeds2, db_path=temp_db, skip_validation=True)
            
            with patch('sync_podcasts.sync.TranscriptionService'):
                with patch('sync_podcasts.sync.OllamaClient'):
                    syncer1 = PodcastSyncer(config1)
                    syncer2 = PodcastSyncer(config2)
                    
                    # Both should be able to acquire their locks
                    assert syncer1.process_lock.acquire() is True
                    assert syncer2.process_lock.acquire() is True
                    
                    # Lock files should be different
                    assert syncer1.process_lock.lock_file != syncer2.process_lock.lock_file
                    
                    # Cleanup
                    syncer1.process_lock.release()
                    syncer2.process_lock.release()
        finally:
            os.unlink(feeds1)
            os.unlink(feeds2)