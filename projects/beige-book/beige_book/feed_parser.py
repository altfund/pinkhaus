"""
RSS feed parser for processing podcast and audio feeds.
"""

import toml
import feedparser
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class FeedItem:
    """Represents a single item from an RSS feed"""
    feed_url: str
    item_id: str
    title: str
    audio_url: str
    published: Optional[datetime] = None
    description: Optional[str] = None
    duration: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'feed_url': self.feed_url,
            'item_id': self.item_id,
            'title': self.title,
            'audio_url': self.audio_url,
            'published': self.published.isoformat() if self.published else None,
            'description': self.description,
            'duration': self.duration
        }


class FeedParser:
    """Parse RSS feeds and extract audio items"""
    
    def __init__(self):
        """Initialize the feed parser"""
        self.supported_audio_types = {
            'audio/mpeg', 'audio/mp3', 'audio/mp4', 'audio/m4a',
            'audio/ogg', 'audio/wav', 'audio/x-m4a', 'audio/x-wav'
        }
    
    def parse_toml_feeds(self, toml_path: str) -> List[str]:
        """
        Parse a TOML file containing RSS feed URLs.
        
        Args:
            toml_path: Path to the TOML file
            
        Returns:
            List of RSS feed URLs
        """
        with open(toml_path, 'r') as f:
            data = toml.load(f)
        
        # Support both feeds.rss and rss directly
        if 'feeds' in data and 'rss' in data['feeds']:
            return data['feeds']['rss']
        elif 'rss' in data:
            return data['rss']
        else:
            raise ValueError("No RSS feeds found in TOML file. Expected 'feeds.rss' or 'rss' key.")
    
    def parse_feed_with_retry(self, feed_url: str, max_retries: int = 3, initial_delay: float = 1.0) -> List[FeedItem]:
        """
        Parse a single RSS feed with retry logic and exponential backoff.
        
        Args:
            feed_url: URL of the RSS feed
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            
        Returns:
            List of FeedItem objects
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return self.parse_feed(feed_url)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Calculate exponential backoff with jitter
                    delay = initial_delay * (2 ** attempt)  # 1s, 2s, 4s
                    # Add jitter (±20%) to prevent thundering herd
                    import random
                    jitter = delay * 0.2 * (2 * random.random() - 1)
                    actual_delay = delay + jitter

                    logger.warning(f"Feed parsing attempt {attempt + 1} failed for {feed_url}, retrying in {actual_delay:.1f}s: {e}")
                    time.sleep(actual_delay)
                else:
                    logger.error(f"All feed parsing attempts failed for {feed_url}")

        raise last_error
    
    def parse_feed(self, feed_url: str) -> List[FeedItem]:
        """
        Parse a single RSS feed and extract audio items.

        Args:
            feed_url: URL of the RSS feed

        Returns:
            List of FeedItem objects
        """
        logger.info(f"Parsing feed: {feed_url}")

        try:
            # Parse the feed with timeout
            # feedparser uses urllib which respects socket timeout
            import socket
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(30)  # 30 second timeout

            try:
                feed = feedparser.parse(feed_url)
            finally:
                socket.setdefaulttimeout(old_timeout)

            if feed.bozo:
                # Check if it's a serious error (network/connection issues)
                if hasattr(feed, 'bozo_exception'):
                    exc = feed.bozo_exception
                    # Raise on network errors to trigger retry
                    if any(err in str(exc).lower() for err in ['urlopen error', 'connection', 'timeout', 'no route to host']):
                        raise Exception(f"Network error parsing feed: {exc}")
                    else:
                        logger.warning(f"Feed parsing had issues: {exc}")

            items = []

            for entry in feed.entries:
                # Extract audio URL from enclosures
                audio_url = self._extract_audio_url(entry)
                if not audio_url:
                    logger.debug(f"No audio found for entry: {entry.get('title', 'Unknown')}")
                    continue

                # Extract item ID (prefer guid, fallback to link)
                item_id = entry.get('id', entry.get('link', audio_url))

                # Parse publication date
                published = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6])

                # Extract duration if available
                duration = None
                if hasattr(entry, 'itunes_duration'):
                    duration = entry.itunes_duration

                item = FeedItem(
                    feed_url=feed_url,
                    item_id=item_id,
                    title=entry.get('title', 'Untitled'),
                    audio_url=audio_url,
                    published=published,
                    description=entry.get('description', ''),
                    duration=duration
                )

                items.append(item)

            logger.info(f"Found {len(items)} audio items in feed")
            return items
            
        except Exception as e:
            logger.error(f"Error parsing feed {feed_url}: {e}")
            raise
    
    def _extract_audio_url(self, entry: Any) -> Optional[str]:
        """
        Extract audio URL from feed entry.
        
        Args:
            entry: Feed entry object
            
        Returns:
            Audio URL if found, None otherwise
        """
        # Check enclosures
        if hasattr(entry, 'enclosures'):
            for enclosure in entry.enclosures:
                if enclosure.get('type', '').lower() in self.supported_audio_types:
                    return enclosure.get('href', enclosure.get('url'))
        
        # Check links
        if hasattr(entry, 'links'):
            for link in entry.links:
                if link.get('type', '').lower() in self.supported_audio_types:
                    return link.get('href')
                # Some feeds use rel="enclosure"
                if link.get('rel') == 'enclosure' and link.get('href'):
                    return link.get('href')
        
        return None
    
    def parse_all_feeds(self, toml_path: str, max_retries: int = 3) -> Dict[str, List[FeedItem]]:
        """
        Parse all feeds from a TOML file with retry logic.
        
        Args:
            toml_path: Path to the TOML file
            max_retries: Maximum number of retry attempts per feed
            
        Returns:
            Dictionary mapping feed URLs to their items
        """
        feed_urls = self.parse_toml_feeds(toml_path)
        results = {}
        
        for feed_url in feed_urls:
            try:
                items = self.parse_feed_with_retry(feed_url, max_retries)
                results[feed_url] = items
            except Exception as e:
                logger.error(f"Failed to parse feed {feed_url} after {max_retries} attempts: {e}")
                results[feed_url] = []
        
        return results