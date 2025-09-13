#!/usr/bin/env python3
"""
Prometheus metrics for Ominari Trading System
"""

from prometheus_client import Counter, Gauge, Histogram, Summary
import time
from functools import wraps

# API metrics
api_requests = Counter(
    'ominari_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

api_request_duration = Histogram(
    'ominari_api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

api_errors = Counter(
    'ominari_api_errors_total',
    'Total API errors',
    ['method', 'endpoint', 'error_type']
)

# Trading metrics
portfolio_value = Gauge(
    'ominari_portfolio_value',
    'Current portfolio value',
    ['session_id']
)

total_pnl = Gauge(
    'ominari_total_pnl',
    'Total P&L',
    ['session_id']
)

open_positions = Gauge(
    'ominari_open_positions',
    'Number of open positions',
    ['session_id']
)

win_rate = Gauge(
    'ominari_win_rate',
    'Win rate percentage',
    ['session_id']
)

# System metrics
active_markets = Gauge(
    'ominari_active_markets',
    'Number of active markets'
)

database_query_duration = Histogram(
    'ominari_database_query_duration_seconds',
    'Database query duration',
    ['query_type']
)

blockchain_rpc_calls = Counter(
    'ominari_blockchain_rpc_calls_total',
    'Total blockchain RPC calls',
    ['network', 'method', 'status']
)

# Decorators for easy metric collection
def track_api_request(endpoint):
    """Decorator to track API requests."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 200
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 500
                api_errors.labels(
                    method='GET',
                    endpoint=endpoint,
                    error_type=type(e).__name__
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                api_requests.labels(
                    method='GET',
                    endpoint=endpoint,
                    status=status
                ).inc()
                api_request_duration.labels(
                    method='GET',
                    endpoint=endpoint
                ).observe(duration)
        
        return wrapper
    return decorator


def track_database_query(query_type):
    """Decorator to track database query performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                database_query_duration.labels(
                    query_type=query_type
                ).observe(duration)
        
        return wrapper
    return decorator


def update_trading_metrics(session_data):
    """Update trading-related metrics."""
    portfolio_value.labels(session_id=session_data['id']).set(
        session_data['portfolio_value']
    )
    total_pnl.labels(session_id=session_data['id']).set(
        session_data['total_pnl']
    )
    open_positions.labels(session_id=session_data['id']).set(
        session_data['open_positions']
    )
    win_rate.labels(session_id=session_data['id']).set(
        session_data['win_rate']
    )
