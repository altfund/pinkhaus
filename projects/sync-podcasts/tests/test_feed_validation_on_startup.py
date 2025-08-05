"""Test feed validation on sync-podcasts startup."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from sync_podcasts.sync import PodcastSyncer, SyncConfig
from sync_podcasts.validate_feed import FeedValidationResult


class TestFeedValidationOnStartup:
    """Test that sync-podcasts validates feeds on startup."""
    
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
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    def test_startup_with_valid_feeds(self, temp_feeds_file, temp_db):
        """Test that sync-podcasts starts normally with valid feeds."""
        config = SyncConfig(
            feeds_path=temp_feeds_file,
            db_path=temp_db
        )
        
        # Mock the validation to return all valid
        with patch('sync_podcasts.sync.validate_feeds_toml') as mock_validate:
            mock_validate.return_value = {
                "https://example.com/feed1.rss": FeedValidationResult(
                    is_valid=True,
                    feed_title="Feed 1",
                    feed_type="RSS",
                    audio_entries=10
                ),
                "https://example.com/feed2.rss": FeedValidationResult(
                    is_valid=True,
                    feed_title="Feed 2",
                    feed_type="RSS",
                    audio_entries=5
                )
            }
            
            # Mock other dependencies
            with patch('sync_podcasts.sync.TranscriptionService'):
                with patch('sync_podcasts.sync.OllamaClient'):
                    with patch.object(PodcastSyncer, '_show_startup_report'):
                        # Should initialize without raising
                        syncer = PodcastSyncer(config)
                        
                        # Verify validation was called
                        mock_validate.assert_called_once_with(temp_feeds_file, move_invalid=False)
    
    def test_startup_with_invalid_feeds(self, temp_feeds_file, temp_db):
        """Test that sync-podcasts exits with error when invalid feeds are found."""
        config = SyncConfig(
            feeds_path=temp_feeds_file,
            db_path=temp_db
        )
        
        # Mock the validation to return one invalid feed
        with patch('sync_podcasts.sync.validate_feeds_toml') as mock_validate:
            mock_validate.return_value = {
                "https://example.com/feed1.rss": FeedValidationResult(
                    is_valid=True,
                    feed_title="Feed 1",
                    feed_type="RSS",
                    audio_entries=10
                ),
                "https://example.com/feed2.rss": FeedValidationResult(
                    is_valid=False,
                    error_message="No audio content found",
                    feed_title="Blog Feed"
                )
            }
            
            # Mock other dependencies
            with patch('sync_podcasts.sync.TranscriptionService'):
                with patch('sync_podcasts.sync.OllamaClient'):
                    # Should raise SystemExit
                    with pytest.raises(SystemExit) as exc_info:
                        syncer = PodcastSyncer(config)
                    
                    assert exc_info.value.code == 1
    
    def test_startup_with_toml_parsing_error(self, temp_feeds_file, temp_db):
        """Test that sync-podcasts exits when feeds file cannot be parsed."""
        config = SyncConfig(
            feeds_path=temp_feeds_file,
            db_path=temp_db
        )
        
        # Mock the validation to return TOML error
        with patch('sync_podcasts.sync.validate_feeds_toml') as mock_validate:
            mock_validate.return_value = {
                "_error": FeedValidationResult(
                    is_valid=False,
                    error_message="Invalid TOML syntax"
                )
            }
            
            # Mock other dependencies
            with patch('sync_podcasts.sync.TranscriptionService'):
                with patch('sync_podcasts.sync.OllamaClient'):
                    # Should raise SystemExit
                    with pytest.raises(SystemExit) as exc_info:
                        syncer = PodcastSyncer(config)
                    
                    assert exc_info.value.code == 1
    
    def test_error_message_includes_validate_feed_suggestion(self, temp_feeds_file, temp_db, caplog):
        """Test that error message suggests using validate-feed tool."""
        config = SyncConfig(
            feeds_path=temp_feeds_file,
            db_path=temp_db
        )
        
        # Mock the validation to return invalid feeds
        with patch('sync_podcasts.sync.validate_feeds_toml') as mock_validate:
            mock_validate.return_value = {
                "https://example.com/bad.rss": FeedValidationResult(
                    is_valid=False,
                    error_message="Network error"
                )
            }
            
            # Mock other dependencies
            with patch('sync_podcasts.sync.TranscriptionService'):
                with patch('sync_podcasts.sync.OllamaClient'):
                    # Should raise SystemExit
                    with pytest.raises(SystemExit):
                        syncer = PodcastSyncer(config)
                    
                    # Check log messages
                    assert "INVALID FEEDS DETECTED" in caplog.text
                    assert "validate-feed" in caplog.text
                    assert "--move-invalid" in caplog.text
                    assert temp_feeds_file in caplog.text
    
    def test_multiple_invalid_feeds_all_shown(self, temp_feeds_file, temp_db, caplog):
        """Test that all invalid feeds are shown in error message."""
        config = SyncConfig(
            feeds_path=temp_feeds_file,
            db_path=temp_db
        )
        
        # Mock the validation to return multiple invalid feeds
        with patch('sync_podcasts.sync.validate_feeds_toml') as mock_validate:
            mock_validate.return_value = {
                "https://example.com/feed1.rss": FeedValidationResult(
                    is_valid=False,
                    error_message="No audio content",
                    feed_title="Blog 1"
                ),
                "https://example.com/feed2.rss": FeedValidationResult(
                    is_valid=False,
                    error_message="Connection refused"
                ),
                "https://example.com/feed3.rss": FeedValidationResult(
                    is_valid=True,
                    feed_title="Valid Feed"
                )
            }
            
            # Mock other dependencies
            with patch('sync_podcasts.sync.TranscriptionService'):
                with patch('sync_podcasts.sync.OllamaClient'):
                    # Should raise SystemExit
                    with pytest.raises(SystemExit):
                        syncer = PodcastSyncer(config)
                    
                    # Check that both invalid feeds are mentioned
                    assert "https://example.com/feed1.rss" in caplog.text
                    assert "https://example.com/feed2.rss" in caplog.text
                    assert "No audio content" in caplog.text
                    assert "Connection refused" in caplog.text
                    assert "Blog 1" in caplog.text
                    assert "Found 2 invalid feed(s)" in caplog.text