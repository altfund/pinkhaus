"""Feed validation utilities for sync-podcasts."""

from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import feedparser
import socket
import logging
from pathlib import Path
import toml
from datetime import datetime

from beige_book.feed_parser import FeedParser

logger = logging.getLogger(__name__)


@dataclass
class FeedValidationResult:
    """Result of feed validation."""

    is_valid: bool
    feed_title: Optional[str] = None
    feed_description: Optional[str] = None
    total_entries: int = 0
    audio_entries: int = 0
    error_message: Optional[str] = None
    feed_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "feed_title": self.feed_title,
            "feed_description": self.feed_description,
            "total_entries": self.total_entries,
            "audio_entries": self.audio_entries,
            "error_message": self.error_message,
            "feed_type": self.feed_type,
        }


class FeedValidator:
    """Validates RSS feeds for podcast compatibility."""

    def __init__(self):
        """Initialize the feed validator."""
        self.feed_parser = FeedParser()

    def validate_feed(self, feed_url: str) -> FeedValidationResult:
        """
        Validate if a URL is a valid RSS feed with audio content.

        Args:
            feed_url: URL to validate

        Returns:
            FeedValidationResult with validation details
        """
        try:
            # Set timeout for network operations
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(30)

            try:
                # Parse the feed
                feed = feedparser.parse(feed_url)
            finally:
                socket.setdefaulttimeout(old_timeout)

            # Check for network/parsing errors
            if feed.bozo:
                if hasattr(feed, "bozo_exception"):
                    exc = feed.bozo_exception
                    error_msg = str(exc)

                    # Check for common network errors
                    if any(
                        err in error_msg.lower()
                        for err in [
                            "urlopen error",
                            "connection",
                            "timeout",
                            "no route to host",
                            "name or service not known",
                        ]
                    ):
                        return FeedValidationResult(
                            is_valid=False, error_message=f"Network error: {error_msg}"
                        )

                    # Check if it's not an RSS/XML feed at all
                    if (
                        "not well-formed" in error_msg.lower()
                        or "syntax error" in error_msg.lower()
                    ):
                        return FeedValidationResult(
                            is_valid=False,
                            error_message="URL does not appear to be a valid RSS/XML feed",
                        )

                    # Other parsing errors - might still be usable
                    logger.warning(f"Feed parsing warning: {error_msg}")

            # Check if we got any data
            if not hasattr(feed, "feed") and not hasattr(feed, "entries"):
                return FeedValidationResult(
                    is_valid=False, error_message="No feed data found at URL"
                )

            # Extract feed metadata
            feed_info = feed.get("feed", {})
            feed_title = feed_info.get("title", "Untitled Feed")
            feed_description = feed_info.get("description", "")

            # Detect feed type
            feed_type = "RSS"
            if feed.version:
                if "atom" in feed.version.lower():
                    feed_type = "Atom"
                elif "rss" in feed.version.lower():
                    feed_type = "RSS"

            # Count entries
            total_entries = len(feed.entries) if hasattr(feed, "entries") else 0

            if total_entries == 0:
                return FeedValidationResult(
                    is_valid=False,
                    feed_title=feed_title,
                    feed_description=feed_description,
                    feed_type=feed_type,
                    error_message="Feed contains no entries",
                )

            # Try to parse with our FeedParser to check for audio content
            try:
                items = self.feed_parser.parse_feed(feed_url)
                audio_entries = len(items)

                if audio_entries == 0:
                    return FeedValidationResult(
                        is_valid=False,
                        feed_title=feed_title,
                        feed_description=feed_description,
                        total_entries=total_entries,
                        audio_entries=0,
                        feed_type=feed_type,
                        error_message=f"Feed has {total_entries} entries but no audio content found",
                    )

                # Success!
                return FeedValidationResult(
                    is_valid=True,
                    feed_title=feed_title,
                    feed_description=feed_description,
                    total_entries=total_entries,
                    audio_entries=audio_entries,
                    feed_type=feed_type,
                )

            except Exception as e:
                return FeedValidationResult(
                    is_valid=False,
                    feed_title=feed_title,
                    feed_description=feed_description,
                    total_entries=total_entries,
                    feed_type=feed_type,
                    error_message=f"Error checking for audio content: {str(e)}",
                )

        except Exception as e:
            return FeedValidationResult(
                is_valid=False, error_message=f"Unexpected error: {str(e)}"
            )


def validate_feed_url(url: str) -> Tuple[bool, str]:
    """
    Simple validation function for use by other modules.

    Args:
        url: URL to validate

    Returns:
        Tuple of (is_valid, message)
    """
    validator = FeedValidator()
    result = validator.validate_feed(url)

    if result.is_valid:
        message = f"Valid {result.feed_type} feed: {result.feed_title} ({result.audio_entries} audio episodes)"
    else:
        message = result.error_message or "Invalid feed"

    return result.is_valid, message


def validate_feeds_toml(
    toml_path: str, move_invalid: bool = False
) -> Dict[str, FeedValidationResult]:
    """
    Validate all feeds in a TOML file.

    Args:
        toml_path: Path to TOML file containing RSS feed URLs
        move_invalid: If True, move invalid feeds to invalid-feeds.toml

    Returns:
        Dictionary mapping feed URLs to their validation results
    """
    validator = FeedValidator()
    feed_parser = FeedParser()

    try:
        feed_urls = feed_parser.parse_toml_feeds(toml_path)
    except Exception as e:
        # Return error for TOML parsing
        return {
            "_error": FeedValidationResult(
                is_valid=False, error_message=f"Error parsing TOML file: {str(e)}"
            )
        }

    results = {}
    for feed_url in feed_urls:
        logger.info(f"Validating feed: {feed_url}")
        results[feed_url] = validator.validate_feed(feed_url)

    # If move_invalid is True, update the TOML files
    if move_invalid:
        valid_feeds = [url for url, result in results.items() if result.is_valid]
        invalid_feeds = [url for url, result in results.items() if not result.is_valid]

        if invalid_feeds:
            # Update the original TOML file with only valid feeds
            update_toml_feeds(toml_path, valid_feeds)

            # Create invalid-feeds.toml with invalid feeds
            invalid_toml_path = Path(toml_path).parent / "invalid-feeds.toml"
            create_invalid_feeds_toml(Path(invalid_toml_path), invalid_feeds, results)

    return results


def update_toml_feeds(toml_path: str, valid_feeds: List[str]) -> None:
    """
    Update TOML file to contain only valid feeds.

    Args:
        toml_path: Path to TOML file
        valid_feeds: List of valid feed URLs
    """
    with open(toml_path, "r") as f:
        data = toml.load(f)

    # Update the feeds list
    if "feeds" in data and "rss" in data["feeds"]:
        data["feeds"]["rss"] = valid_feeds
    elif "rss" in data:
        data["rss"] = valid_feeds

    # Write back to file
    with open(toml_path, "w") as f:
        toml.dump(data, f)


def create_invalid_feeds_toml(
    invalid_toml_path: Path,
    invalid_feeds: List[str],
    results: Dict[str, FeedValidationResult],
) -> None:
    """
    Create or update invalid-feeds.toml with invalid feeds and their errors.

    Args:
        invalid_toml_path: Path to invalid feeds TOML file
        invalid_feeds: List of invalid feed URLs
        results: Validation results for context
    """
    # Check if invalid-feeds.toml already exists and load existing invalid feeds
    existing_invalid_feeds = {}
    if invalid_toml_path.exists():
        try:
            with open(invalid_toml_path, "r") as f:
                content = f.read()
                # Parse the content manually to extract feed URLs and their metadata
                lines = content.split("\n")
                current_url = None
                current_error = None
                current_title = None

                for line in lines:
                    line = line.strip()
                    if line.startswith("# Error:"):
                        current_error = line.replace("# Error:", "").strip()
                    elif line.startswith("# Title:"):
                        current_title = line.replace("# Title:", "").strip()
                    elif line.startswith('"') and line.endswith('",'):
                        # Extract URL from quoted string (remove leading " and trailing ",)
                        current_url = line[1:-2]
                        if current_url:
                            existing_invalid_feeds[current_url] = {
                                "url": current_url,
                                "error": current_error or "Previously invalid",
                                "title": current_title,
                            }
                        # Reset for next entry
                        current_error = None
                        current_title = None
        except Exception:
            # If we can't read it, start fresh
            pass

    # Add/update invalid feeds with their error messages
    for feed_url in invalid_feeds:
        result = results.get(feed_url)
        error_msg = result.error_message if result else "Unknown error"
        # Update or add the feed info
        existing_invalid_feeds[feed_url] = {
            "url": feed_url,
            "error": error_msg,
            "title": result.feed_title if result and result.feed_title else None,
        }

    # Convert back to list for writing
    invalid_feeds_list = list(existing_invalid_feeds.values())

    # Write to file with comments
    with open(invalid_toml_path, "w") as f:
        f.write("# Invalid RSS feeds moved from feeds.toml\n")
        f.write("# These feeds failed validation and need to be fixed or removed\n")
        f.write(f"# Last updated: {datetime.now().isoformat()}\n\n")
        f.write("[feeds]\n")
        f.write("invalid_rss = [\n")
        for item in invalid_feeds_list:
            f.write(f"    # Error: {item['error']}\n")
            if item.get("title"):
                f.write(f"    # Title: {item['title']}\n")
            f.write(f'    "{item["url"]}",\n')
        f.write("]\n")
