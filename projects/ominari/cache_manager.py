"""
Simple in-memory cache manager for Ominari DApp
"""
import time
from typing import Any, Optional, Dict
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """Simple TTL-based cache manager"""
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            entry = self.cache[key]
            if entry['expires'] > time.time():
                self.stats['hits'] += 1
                logger.debug(f"Cache hit: {key}")
                return entry['value']
            else:
                # Expired
                del self.cache[key]
                self.stats['evictions'] += 1
        
        self.stats['misses'] += 1
        logger.debug(f"Cache miss: {key}")
        return None
    
    def set(self, key: str, value: Any, ttl: int = 60):
        """Set value in cache with TTL in seconds"""
        self.cache[key] = {
            'value': value,
            'expires': time.time() + ttl
        }
        logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
    
    def delete(self, key: str):
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def cleanup_expired(self):
        """Remove expired entries"""
        now = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry['expires'] <= now
        ]
        for key in expired_keys:
            del self.cache[key]
            self.stats['evictions'] += 1
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total * 100) if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'hit_rate': f"{hit_rate:.1f}%"
        }


# Global cache instance
cache = CacheManager()


def cached(ttl: int = 60):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


# Specialized caches for different data types
class MarketDataCache:
    """Specialized cache for market data"""
    
    def __init__(self, ttl: int = 30):
        self.ttl = ttl
        self.cache = CacheManager()
    
    @cached(ttl=30)
    def get_markets_with_odds(self, limit: int = 150):
        """Cache market queries"""
        from database_v2 import db_manager
        from models import Market, Odd
        from sqlalchemy import and_, not_
        
        with db_manager.get_db_session() as db:
            results = db.query(Market, Odd).join(
                Odd, Market.source_id == Odd.source_id
            ).filter(
                not_(and_(Odd.decimal_odds.in_([2.5, 2.8, 3.0])))
            ).order_by(Market.maturity_date.desc()).limit(limit).all()
            
            markets = []
            for market, odd in results:
                markets.append({
                    'market': market,
                    'odd': odd
                })
            
            return markets
    
    def invalidate(self):
        """Invalidate market cache"""
        self.cache.clear()


# Create specialized cache instances
market_cache = MarketDataCache()


# Background cleanup task
import threading

def cache_cleanup_worker():
    """Background worker to cleanup expired cache entries"""
    while True:
        time.sleep(60)  # Run every minute
        cache.cleanup_expired()
        logger.debug(f"Cache stats: {cache.get_stats()}")

# Start cleanup thread
cleanup_thread = threading.Thread(target=cache_cleanup_worker, daemon=True)
cleanup_thread.start()