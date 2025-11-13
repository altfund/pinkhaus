"""
Rate limiting for Ominari DApp API endpoints
"""
from functools import wraps
from flask import request, jsonify
import time
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(deque)
        self.blocked_ips = set()
        
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        if identifier in self.blocked_ips:
            return False
            
        now = time.time()
        minute_ago = now - 60
        
        # Remove old requests
        while self.requests[identifier] and self.requests[identifier][0] < minute_ago:
            self.requests[identifier].popleft()
        
        # Check rate limit
        if len(self.requests[identifier]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {identifier}")
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True
    
    def block_ip(self, ip: str, duration: int = 3600):
        """Temporarily block an IP"""
        self.blocked_ips.add(ip)
        logger.warning(f"Blocked IP: {ip} for {duration}s")
        
        # Schedule unblock
        def unblock():
            time.sleep(duration)
            self.blocked_ips.discard(ip)
            logger.info(f"Unblocked IP: {ip}")
        
        import threading
        threading.Thread(target=unblock, daemon=True).start()


# Global rate limiter instances
general_limiter = RateLimiter(requests_per_minute=60)
api_limiter = RateLimiter(requests_per_minute=30)
websocket_limiter = RateLimiter(requests_per_minute=120)


def rate_limit(limiter=None, requests_per_minute=60):
    """Rate limiting decorator"""
    if limiter is None:
        limiter = RateLimiter(requests_per_minute)
        
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Get client identifier (IP address)
            identifier = request.remote_addr
            
            # Check rate limit
            if not limiter.is_allowed(identifier):
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Maximum {limiter.requests_per_minute} requests per minute'
                }), 429
            
            return f(*args, **kwargs)
        
        return wrapped
    return decorator


def ws_rate_limit(limiter=None):
    """WebSocket rate limiting decorator"""
    if limiter is None:
        limiter = websocket_limiter
        
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # For WebSocket, get client from request context
            try:
                from flask_socketio import request as ws_request
                identifier = ws_request.remote_addr
            except:
                identifier = "unknown"
            
            if not limiter.is_allowed(identifier):
                logger.warning(f"WebSocket rate limit exceeded for {identifier}")
                return {'error': 'Rate limit exceeded'}
            
            return f(*args, **kwargs)
        
        return wrapped
    return decorator


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load"""
    
    def __init__(self, base_rate: int = 60, min_rate: int = 10, max_rate: int = 200):
        self.base_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.current_rate = base_rate
        self.limiter = RateLimiter(self.current_rate)
        self.load_history = deque(maxlen=60)  # 1 minute of history
        
    def adjust_rate(self, cpu_usage: float, memory_usage: float):
        """Adjust rate based on system load"""
        load_score = (cpu_usage + memory_usage) / 2
        self.load_history.append(load_score)
        
        if len(self.load_history) < 10:
            return
        
        avg_load = sum(self.load_history) / len(self.load_history)
        
        if avg_load > 0.8:  # High load
            self.current_rate = max(self.min_rate, int(self.current_rate * 0.8))
        elif avg_load < 0.3:  # Low load
            self.current_rate = min(self.max_rate, int(self.current_rate * 1.2))
        else:
            # Gradually return to base rate
            if self.current_rate < self.base_rate:
                self.current_rate = min(self.base_rate, int(self.current_rate * 1.1))
            elif self.current_rate > self.base_rate:
                self.current_rate = max(self.base_rate, int(self.current_rate * 0.9))
        
        # Update limiter
        self.limiter.requests_per_minute = self.current_rate
        logger.debug(f"Adjusted rate limit to {self.current_rate} requests/min (load: {avg_load:.2f})")


# Middleware for global rate limiting
def setup_rate_limiting(app):
    """Setup global rate limiting for Flask app"""
    
    @app.before_request
    def check_rate_limit():
        # Skip rate limiting for static files and health checks
        if request.path.startswith('/static') or request.path in ['/health', '/metrics']:
            return
        
        identifier = request.remote_addr
        
        # Use different limiters for different endpoints
        if request.path.startswith('/api/'):
            limiter = api_limiter
        else:
            limiter = general_limiter
        
        if not limiter.is_allowed(identifier):
            logger.warning(f"Rate limit exceeded for {identifier} on {request.path}")
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': 'Please slow down your requests'
            }), 429
    
    return app