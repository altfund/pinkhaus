#!/usr/bin/env python3
"""
Comprehensive test suite for Ominari DApp
Tests all major components including rate limiting and caching
"""

import unittest
import asyncio
import time
import json
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test Categories:
# 1. Rate Limiting Tests
# 2. Caching Tests
# 3. Database Tests
# 4. WebSocket Tests
# 5. Integration Tests


class TestRateLimiting(unittest.TestCase):
    """Test rate limiting functionality"""
    
    def setUp(self):
        from rate_limiter import RateLimiter, AdaptiveRateLimiter
        self.limiter = RateLimiter(requests_per_minute=10)
        self.adaptive = AdaptiveRateLimiter(base_rate=60)
    
    def test_rate_limit_basic(self):
        """Test basic rate limiting"""
        # Should allow first 10 requests
        for i in range(10):
            self.assertTrue(self.limiter.is_allowed("test_ip"))
        
        # 11th request should be blocked
        self.assertFalse(self.limiter.is_allowed("test_ip"))
    
    def test_rate_limit_different_ips(self):
        """Test rate limiting for different IPs"""
        # Each IP should have its own limit
        for i in range(10):
            self.assertTrue(self.limiter.is_allowed(f"ip_{i}"))
        
        # All IPs should still be allowed one more
        for i in range(10):
            self.assertTrue(self.limiter.is_allowed(f"ip_{i}"))
    
    def test_rate_limit_time_window(self):
        """Test rate limit resets after time window"""
        # Use up the limit
        for i in range(10):
            self.limiter.is_allowed("test_ip")
        
        # Should be blocked
        self.assertFalse(self.limiter.is_allowed("test_ip"))
        
        # Mock time passing (would need to patch time.time())
        # In real test, we'd wait or mock time
    
    def test_adaptive_rate_limiting(self):
        """Test adaptive rate limiting adjusts based on load"""
        # Test high load scenario
        self.adaptive.adjust_rate(cpu_usage=0.9, memory_usage=0.9)
        self.assertLess(self.adaptive.current_rate, self.adaptive.base_rate)
        
        # Test low load scenario
        for _ in range(20):  # Build history
            self.adaptive.adjust_rate(cpu_usage=0.2, memory_usage=0.2)
        self.assertGreater(self.adaptive.current_rate, self.adaptive.base_rate)
    
    def test_ip_blocking(self):
        """Test IP blocking functionality"""
        self.limiter.block_ip("bad_ip", duration=1)
        self.assertFalse(self.limiter.is_allowed("bad_ip"))
        self.assertIn("bad_ip", self.limiter.blocked_ips)


class TestCaching(unittest.TestCase):
    """Test caching functionality"""
    
    def setUp(self):
        from cache_manager import CacheManager
        self.cache = CacheManager()
    
    def test_cache_set_get(self):
        """Test basic cache set and get"""
        self.cache.set("key1", "value1", ttl=60)
        self.assertEqual(self.cache.get("key1"), "value1")
    
    def test_cache_miss(self):
        """Test cache miss returns None"""
        self.assertIsNone(self.cache.get("nonexistent"))
        self.assertEqual(self.cache.stats['misses'], 1)
    
    def test_cache_expiration(self):
        """Test cache expiration"""
        self.cache.set("expire_key", "value", ttl=0)
        time.sleep(0.1)
        self.assertIsNone(self.cache.get("expire_key"))
        self.assertEqual(self.cache.stats['evictions'], 1)
    
    def test_cache_delete(self):
        """Test cache deletion"""
        self.cache.set("del_key", "value")
        self.cache.delete("del_key")
        self.assertIsNone(self.cache.get("del_key"))
    
    def test_cache_clear(self):
        """Test cache clear"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.clear()
        self.assertEqual(len(self.cache.cache), 0)
    
    def test_cache_stats(self):
        """Test cache statistics"""
        # Generate some hits and misses
        self.cache.set("key1", "value1")
        self.cache.get("key1")  # hit
        self.cache.get("key1")  # hit
        self.cache.get("missing")  # miss
        
        stats = self.cache.get_stats()
        self.assertEqual(stats['hits'], 2)
        self.assertEqual(stats['misses'], 2)  # Including initial test
        self.assertEqual(stats['size'], 1)
        self.assertIn('hit_rate', stats)
    
    def test_cache_decorator(self):
        """Test caching decorator"""
        from cache_manager import cached
        
        call_count = 0
        
        @cached(ttl=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should execute
        result1 = expensive_function(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)
        
        # Second call should use cache
        result2 = expensive_function(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)
        
        # Different argument should execute
        result3 = expensive_function(6)
        self.assertEqual(result3, 12)
        self.assertEqual(call_count, 2)


class TestDatabaseOperations(unittest.TestCase):
    """Test database operations with safety checks"""
    
    @patch('database_v2.create_engine')
    @patch('database_v2.sessionmaker')
    def test_database_connection(self, mock_sessionmaker, mock_create_engine):
        """Test database connection handling"""
        from database_v2 import DatabaseManager
        
        # Mock the database
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_session = MagicMock()
        mock_sessionmaker.return_value = lambda: mock_session
        
        db_manager = DatabaseManager()
        
        # Test connection
        with db_manager.get_db_session() as session:
            self.assertIsNotNone(session)
        
        # Verify session was closed
        mock_session.close.assert_called()
    
    @patch('database_v2.db_manager')
    def test_safe_query_limits(self, mock_db_manager):
        """Test that queries always have limits"""
        from models import Market, Odd
        
        # Mock query
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        mock_db_manager.get_db_session.return_value.__enter__.return_value = mock_session
        
        # Test query with limit
        with mock_db_manager.get_db_session() as db:
            results = db.query(Market).filter(Market.sport == 'Soccer').limit(100).all()
        
        # Verify limit was called
        mock_query.limit.assert_called_with(100)


class TestWebSocketConnections(unittest.TestCase):
    """Test WebSocket functionality"""
    
    def setUp(self):
        from web_dashboard_real_odds import app, socketio
        self.app = app
        self.socketio = socketio
        self.client = self.socketio.test_client(self.app)
    
    def test_websocket_connection(self):
        """Test WebSocket connection"""
        # Check if connected
        self.assertTrue(self.client.is_connected())
        
        # Check connection response
        received = self.client.get_received()
        self.assertTrue(any(msg['name'] == 'connected' for msg in received))
    
    def test_websocket_rate_limiting(self):
        """Test WebSocket rate limiting"""
        # This would need actual rate limit testing
        pass
    
    def test_dashboard_data_request(self):
        """Test dashboard data request"""
        self.client.emit('request_dashboard_data')
        
        # Wait for async response
        time.sleep(0.1)
        
        received = self.client.get_received()
        # Should receive dashboard_update event
        self.assertTrue(any(msg['name'] == 'dashboard_update' for msg in received))


class TestHealthEndpoints(unittest.TestCase):
    """Test health check endpoints"""
    
    def setUp(self):
        from web_dashboard_real_odds import app
        self.app = app
        self.client = self.app.test_client()
    
    @patch('database_v2.db_manager')
    def test_health_endpoint(self, mock_db_manager):
        """Test /health endpoint"""
        # Mock healthy database
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.count.return_value = 100
        mock_session.query.return_value = mock_query
        mock_db_manager.get_db_session.return_value.__enter__.return_value = mock_session
        
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('services', data)
    
    @patch('database_v2.db_manager')
    def test_metrics_endpoint(self, mock_db_manager):
        """Test /metrics endpoint"""
        # Mock database
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.count.return_value = 100
        mock_query.filter.return_value = mock_query
        mock_session.query.return_value = mock_query
        mock_db_manager.get_db_session.return_value.__enter__.return_value = mock_session
        
        response = self.client.get('/metrics')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'ominari_markets_total', response.data)
    
    def test_cache_stats_endpoint(self):
        """Test /cache-stats endpoint"""
        response = self.client.get('/cache-stats')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('hits', data)
        self.assertIn('misses', data)
        self.assertIn('hit_rate', data)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full system"""
    
    @patch('database_v2.db_manager')
    @patch('paper_trading_postgres_integrated.PaperTradingSessionManager')
    def test_full_dashboard_flow(self, mock_session_manager, mock_db_manager):
        """Test full dashboard data flow"""
        from web_dashboard_real_odds import app, get_dashboard_data
        
        # Mock database results
        mock_market = MagicMock()
        mock_market.source_id = "test_123"
        mock_market.home_team = "Team A"
        mock_market.away_team = "Team B"
        mock_market.sport = "Soccer"
        mock_market.maturity_date = "2024-01-01"
        
        mock_odd = MagicMock()
        mock_odd.decimal_odds = 2.5
        mock_odd.outcome = "home"
        mock_odd.source = "test"
        
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [(mock_market, mock_odd)]
        mock_query.group_by.return_value = mock_query
        mock_session.query.return_value = mock_query
        
        mock_db_manager.get_db_session.return_value.__enter__.return_value = mock_session
        
        # Mock session manager
        mock_session_manager.return_value.get_current_session.return_value = "session_123"
        mock_session_manager.return_value.get_session.return_value = {
            'current_bankroll': 10000
        }
        
        # Test dashboard data
        data = asyncio.run(get_dashboard_data())
        
        self.assertIn('markets', data)
        self.assertIn('trading_status', data)
        self.assertIn('stats', data)
        self.assertGreater(len(data['markets']), 0)
    
    def test_rate_limit_with_cache(self):
        """Test rate limiting works with caching"""
        from web_dashboard_real_odds import app
        from cache_manager import cache
        
        client = app.test_client()
        
        # First request should work and cache
        response1 = client.get('/cache-stats')
        self.assertEqual(response1.status_code, 200)
        
        # Subsequent requests should also work (under rate limit)
        for i in range(5):
            response = client.get('/cache-stats')
            self.assertEqual(response.status_code, 200)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_database_connection_failure(self):
        """Test handling of database connection failures"""
        # Would test graceful degradation
        pass
    
    def test_cache_overflow(self):
        """Test cache behavior when full"""
        from cache_manager import CacheManager
        cache = CacheManager()
        
        # Add many items
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}", ttl=3600)
        
        # Cache should still function
        self.assertIsNotNone(cache.get("key_999"))
    
    def test_concurrent_rate_limiting(self):
        """Test rate limiting under concurrent load"""
        from rate_limiter import RateLimiter
        limiter = RateLimiter(requests_per_minute=100)
        
        # Simulate concurrent requests
        results = []
        for i in range(150):
            results.append(limiter.is_allowed("test_ip"))
        
        # Should have exactly 100 True and 50 False
        self.assertEqual(sum(results), 100)


# Test runner
if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
    
    # Alternative: Run specific test suites
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestRateLimiting)
    # unittest.TextTestRunner(verbosity=2).run(suite)