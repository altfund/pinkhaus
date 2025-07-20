"""
RSS feed parser for processing podcast and audio feeds.
"""

import toml
import feedparser
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

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
            # Parse the feed
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:
                logger.warning(f"Feed parsing had issues: {feed.bozo_exception}")
            
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
    
    def parse_all_feeds(self, toml_path: str) -> Dict[str, List[FeedItem]]:
        """
        Parse all feeds from a TOML file.
        
        Args:
            toml_path: Path to the TOML file
            
        Returns:
            Dictionary mapping feed URLs to their items
        """
        feed_urls = self.parse_toml_feeds(toml_path)
        results = {}
        
        for feed_url in feed_urls:
            try:
                items = self.parse_feed(feed_url)
                results[feed_url] = items
            except Exception as e:
                logger.error(f"Failed to parse feed {feed_url}: {e}")
                results[feed_url] = []
        
        return results