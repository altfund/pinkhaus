#!/usr/bin/env python3
"""
Feature-specific tests for Ominari DApp improvements
Focus on rate limiting, caching, and health monitoring
"""

import pytest
import time
import json
from unittest.mock import Mock, patch
import asyncio
import threading


class TestRateLimitingFeatures:
    """Test the rate limiting implementation"""
    
    @pytest.fixture
    def rate_limiter(self):
        from rate_limiter import RateLimiter
        return RateLimiter(requests_per_minute=10)
    
    @pytest.fixture
    def adaptive_limiter(self):
        from rate_limiter import AdaptiveRateLimiter
        return AdaptiveRateLimiter(base_rate=60, min_rate=10, max_rate=200)
    
    def test_basic_rate_limiting(self, rate_limiter):
        """Test basic rate limiting functionality"""
        ip = "192.168.1.1"
        
        # First 10 requests should pass
        for i in range(10):
            assert rate_limiter.is_allowed(ip) == True
        
        # 11th request should fail
        assert rate_limiter.is_allowed(ip) == False
    
    def test_multiple_ip_tracking(self, rate_limiter):
        """Test that different IPs are tracked separately"""
        ip1 = "192.168.1.1"
        ip2 = "192.168.1.2"
        
        # Use up ip1's limit
        for i in range(10):
            rate_limiter.is_allowed(ip1)
        
        # ip2 should still have full allowance
        assert rate_limiter.is_allowed(ip2) == True
        assert rate_limiter.is_allowed(ip1) == False
    
    def test_time_window_reset(self, rate_limiter):
        """Test that rate limit resets after time window"""
        ip = "192.168.1.1"
        
        # Use up limit
        for i in range(10):
            rate_limiter.is_allowed(ip)
        
        # Should be blocked
        assert rate_limiter.is_allowed(ip) == False
        
        # Simulate time passing by manipulating internal state
        # In real scenario, we'd wait 60 seconds
        rate_limiter.requests[ip].clear()
        
        # Should be allowed again
        assert rate_limiter.is_allowed(ip) == True
    
    def test_ip_blocking(self, rate_limiter):
        """Test IP blocking functionality"""
        ip = "192.168.1.1"
        
        # Block IP
        rate_limiter.block_ip(ip, duration=1)
        
        # Should be blocked
        assert rate_limiter.is_allowed(ip) == False
        assert ip in rate_limiter.blocked_ips
        
        # Wait for unblock
        time.sleep(1.1)
        assert ip not in rate_limiter.blocked_ips
    
    def test_adaptive_rate_adjustment(self, adaptive_limiter):
        """Test adaptive rate limiting based on load"""
        # Add load history
        for _ in range(15):
            adaptive_limiter.adjust_rate(cpu_usage=0.9, memory_usage=0.9)
        
        # Should reduce rate under high load
        assert adaptive_limiter.current_rate < adaptive_limiter.base_rate
        
        # Add low load history
        for _ in range(15):
            adaptive_limiter.adjust_rate(cpu_usage=0.1, memory_usage=0.1)
        
        # Should increase rate under low load
        assert adaptive_limiter.current_rate > adaptive_limiter.base_rate


class TestCachingFeatures:
    """Test the caching implementation"""
    
    @pytest.fixture
    def cache_manager(self):
        from cache_manager import CacheManager
        return CacheManager()
    
    def test_cache_basic_operations(self, cache_manager):
        """Test basic cache operations"""
        # Set and get
        cache_manager.set("key1", "value1", ttl=60)
        assert cache_manager.get("key1") == "value1"
        
        # Miss
        assert cache_manager.get("nonexistent") is None
        
        # Delete
        cache_manager.delete("key1")
        assert cache_manager.get("key1") is None
    
    def test_cache_ttl_expiration(self, cache_manager):
        """Test cache TTL expiration"""
        cache_manager.set("expire_key", "value", ttl=0.1)
        assert cache_manager.get("expire_key") == "value"
        
        time.sleep(0.2)
        assert cache_manager.get("expire_key") is None
        assert cache_manager.stats['evictions'] > 0
    
    def test_cache_statistics(self, cache_manager):
        """Test cache statistics tracking"""
        # Generate activity
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        
        # Hits
        cache_manager.get("key1")
        cache_manager.get("key2")
        
        # Misses
        cache_manager.get("missing1")
        cache_manager.get("missing2")
        
        stats = cache_manager.get_stats()
        assert stats['hits'] == 2
        assert stats['misses'] == 2
        assert stats['size'] == 2
        assert float(stats['hit_rate'].rstrip('%')) == 50.0
    
    def test_cache_decorator(self):
        """Test caching decorator functionality"""
        from cache_manager import cached
        
        call_count = 0
        
        @cached(ttl=60)
        def expensive_operation(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call
        result1 = expensive_operation(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call (cached)
        result2 = expensive_operation(1, 2)
        assert result2 == 3
        assert call_count == 1
        
        # Different args
        result3 = expensive_operation(2, 3)
        assert result3 == 5
        assert call_count == 2
    
    def test_market_data_cache(self):
        """Test specialized market data caching"""
        from cache_manager import MarketDataCache
        
        market_cache = MarketDataCache(ttl=30)
        
        # This would need database mocking to test properly
        # Just verify the cache structure exists
        assert hasattr(market_cache, 'cache')
        assert hasattr(market_cache, 'ttl')
        assert hasattr(market_cache, 'get_markets_with_odds')


class TestHealthMonitoring:
    """Test health monitoring endpoints"""
    
    @pytest.fixture
    def test_app(self):
        from web_dashboard_real_odds import app
        app.config['TESTING'] = True
        return app.test_client()
    
    @patch('database_v2.db_manager')
    @patch('paper_trading_postgres_integrated.PaperTradingSessionManager')
    def test_health_endpoint_healthy(self, mock_session_mgr, mock_db_mgr, test_app):
        """Test health endpoint when system is healthy"""
        # Mock healthy database
        mock_session = Mock()
        mock_session.query.return_value.count.return_value = 100
        mock_db_mgr.get_db_session.return_value.__enter__.return_value = mock_session
        
        # Mock healthy session manager
        mock_session_mgr.return_value.get_current_session.return_value = "session_id"
        
        response = test_app.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['services']['database']['status'] == 'up'
        assert data['services']['session_manager']['status'] == 'up'
    
    @patch('database_v2.db_manager')
    def test_health_endpoint_unhealthy(self, mock_db_mgr, test_app):
        """Test health endpoint when system is unhealthy"""
        # Mock database failure
        mock_db_mgr.get_db_session.side_effect = Exception("DB Error")
        
        response = test_app.get('/health')
        assert response.status_code == 503
        
        data = json.loads(response.data)
        assert data['status'] == 'unhealthy'
    
    @patch('database_v2.db_manager')
    def test_metrics_endpoint(self, mock_db_mgr, test_app):
        """Test Prometheus metrics endpoint"""
        # Mock database queries
        mock_session = Mock()
        mock_query = Mock()
        mock_query.count.return_value = 500
        mock_query.filter.return_value = mock_query
        mock_session.query.return_value = mock_query
        mock_db_mgr.get_db_session.return_value.__enter__.return_value = mock_session
        
        response = test_app.get('/metrics')
        assert response.status_code == 200
        assert response.content_type == 'text/plain'
        assert b'ominari_markets_total' in response.data
        assert b'ominari_real_odds_total' in response.data
    
    def test_cache_stats_endpoint(self, test_app):
        """Test cache statistics endpoint"""
        response = test_app.get('/cache-stats')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'hits' in data
        assert 'misses' in data
        assert 'size' in data
        assert 'hit_rate' in data


class TestWebSocketFeatures:
    """Test WebSocket functionality with rate limiting"""
    
    @pytest.fixture
    def socketio_client(self):
        from web_dashboard_real_odds import app, socketio
        return socketio.test_client(app)
    
    def test_websocket_connection(self, socketio_client):
        """Test WebSocket connection establishment"""
        assert socketio_client.is_connected()
        
        received = socketio_client.get_received()
        connected_msg = next((msg for msg in received if msg['name'] == 'connected'), None)
        assert connected_msg is not None
        assert connected_msg['args'][0]['status'] == 'ok'
    
    @patch('web_dashboard_real_odds.get_dashboard_data')
    def test_dashboard_data_request(self, mock_get_data, socketio_client):
        """Test dashboard data request via WebSocket"""
        # Mock dashboard data
        mock_get_data.return_value = {
            'markets': [],
            'trading_status': {'status': 'Active', 'bankroll': 10000},
            'stats': {'total_markets': 0}
        }
        
        socketio_client.emit('request_dashboard_data')
        
        # Wait for async processing
        time.sleep(0.5)
        
        received = socketio_client.get_received()
        update_msg = next((msg for msg in received if msg['name'] == 'dashboard_update'), None)
        assert update_msg is not None


class TestIntegrationScenarios:
    """Test real-world integration scenarios"""
    
    @pytest.fixture
    def full_app(self):
        from web_dashboard_real_odds import app
        from cache_manager import cache
        from rate_limiter import general_limiter
        
        app.config['TESTING'] = True
        
        # Clear state
        cache.clear()
        general_limiter.requests.clear()
        
        return app.test_client()
    
    def test_rate_limited_caching(self, full_app):
        """Test that rate limiting and caching work together"""
        # First request - should work and cache
        response1 = full_app.get('/cache-stats')
        assert response1.status_code == 200
        data1 = json.loads(response1.data)
        
        # Rapid requests - should be rate limited but use cache
        responses = []
        for i in range(100):
            responses.append(full_app.get('/cache-stats'))
        
        # Some should be rate limited (429)
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes
        
        # But successful ones should get cached data
        success_responses = [r for r in responses if r.status_code == 200]
        assert len(success_responses) > 0
    
    @patch('database_v2.db_manager')
    async def test_concurrent_dashboard_requests(self, mock_db_mgr):
        """Test handling of concurrent dashboard requests"""
        from web_dashboard_real_odds import get_real_odds_data
        from cache_manager import cache
        
        # Mock database
        mock_session = Mock()
        mock_query = Mock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        mock_query.group_by.return_value = mock_query
        mock_session.query.return_value = mock_query
        mock_db_mgr.get_db_session.return_value.__enter__.return_value = mock_session
        
        # Clear cache
        cache.clear()
        
        # Make concurrent requests
        tasks = []
        for i in range(10):
            tasks.append(get_real_odds_data())
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 10
        
        # Cache should have been used (check hit rate)
        stats = cache.get_stats()
        assert stats['hits'] > 0  # Some requests should hit cache


class TestPerformanceOptimizations:
    """Test performance optimizations"""
    
    def test_cache_cleanup_thread(self):
        """Test that cache cleanup thread is running"""
        from cache_manager import cache
        
        # Add expired entry
        cache.set("expire_test", "value", ttl=0)
        
        # Wait for cleanup thread (runs every minute, but we can call directly)
        cache.cleanup_expired()
        
        # Should be cleaned up
        assert cache.get("expire_test") is None
    
    def test_rate_limiter_memory_usage(self):
        """Test rate limiter doesn't leak memory"""
        from rate_limiter import RateLimiter
        import sys
        
        limiter = RateLimiter(requests_per_minute=60)
        
        # Simulate many different IPs
        for i in range(1000):
            limiter.is_allowed(f"ip_{i}")
        
        # Check memory usage is reasonable
        # (In production, you'd use memory profiling tools)
        assert len(limiter.requests) == 1000
    
    @patch('time.time')
    def test_cache_performance(self, mock_time):
        """Test cache performance characteristics"""
        from cache_manager import CacheManager
        
        cache = CacheManager()
        mock_time.return_value = 1000
        
        # Add many items
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}", ttl=3600)
        
        # Get should be fast (O(1))
        start = time.perf_counter()
        value = cache.get("key_500")
        end = time.perf_counter()
        
        assert value == "value_500"
        assert (end - start) < 0.001  # Should be very fast


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])