"""Test utilities for sync-podcasts tests."""

from unittest.mock import patch
from sync_podcasts.validate_feed import FeedValidationResult


def mock_valid_feed_validation():
    """
    Create a mock for validate_feeds_toml that always returns valid feeds.
    
    This is a decorator that can be used to mock feed validation in tests
    that don't care about validation logic.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            def mock_validate(toml_path, move_invalid=False):
                # Return a simple valid result
                # We don't know what feeds are in the file, so just return empty
                return {}
            
            with patch('sync_podcasts.sync.validate_feeds_toml', side_effect=mock_validate):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def create_mock_validation_results(feed_urls, all_valid=True):
    """
    Create mock validation results for given feed URLs.
    
    Args:
        feed_urls: List of feed URLs
        all_valid: If True, all feeds are valid. If False, all are invalid.
        
    Returns:
        Dict mapping feed URLs to FeedValidationResult objects
    """
    results = {}
    for i, url in enumerate(feed_urls):
        if all_valid:
            results[url] = FeedValidationResult(
                is_valid=True,
                feed_title=f"Feed {i+1}",
                feed_type="RSS",
                audio_entries=10
            )
        else:
            results[url] = FeedValidationResult(
                is_valid=False,
                error_message="Test error"
            )
    return results