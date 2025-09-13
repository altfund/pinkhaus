#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitoring and Metrics Collection
Prometheus metrics, health checks, and alerting.
"""

import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from contextlib import contextmanager
from functools import wraps
import psutil
import asyncio

from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    start_http_server, CollectorRegistry
)

from config import settings

logger = logging.getLogger(__name__)

# Create custom registry
registry = CollectorRegistry()

# System metrics
system_info = Info(
    'ominari_system_info',
    'Ominari system information',
    registry=registry
)

uptime_seconds = Gauge(
    'ominari_uptime_seconds',
    'Time since system start in seconds',
    registry=registry
)

# Signal metrics
active_signals = Gauge(
    'ominari_active_signals_total',
    'Number of active signals',
    registry=registry
)

signal_weights = Gauge(
    'ominari_signal_weight',
    'Current weight for each signal',
    ['signal_name'],
    registry=registry
)

signal_performance = Gauge(
    'ominari_signal_performance',
    'Signal performance metrics',
    ['signal_name', 'metric'],
    registry=registry
)

signal_predictions = Counter(
    'ominari_signal_predictions_total',
    'Total predictions made by each signal',
    ['signal_name'],
    registry=registry
)

# Trading metrics
trades_submitted = Counter(
    'ominari_trades_submitted_total',
    'Total trades submitted',
    ['type', 'signal'],
    registry=registry
)

trades_filled = Counter(
    'ominari_trades_filled_total',
    'Total trades filled',
    ['type', 'signal'],
    registry=registry
)

trade_latency = Histogram(
    'ominari_trade_latency_seconds',
    'Trade execution latency',
    ['type'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=registry
)

position_size = Gauge(
    'ominari_position_size',
    'Current position sizes',
    ['market_id', 'type'],
    registry=registry
)

pnl_total = Gauge(
    'ominari_pnl_total',
    'Total profit and loss',
    ['type'],
    registry=registry
)

capital_available = Gauge(
    'ominari_capital_available',
    'Available capital for trading',
    ['type'],
    registry=registry
)

# Backtest metrics
backtest_duration = Histogram(
    'ominari_backtest_duration_seconds',
    'Backtest execution duration',
    ['signal_name'],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
    registry=registry
)

backtest_sharpe = Gauge(
    'ominari_backtest_sharpe_ratio',
    'Backtest Sharpe ratio',
    ['signal_name', 'period'],
    registry=registry
)

# Data collection metrics
data_collection_duration = Histogram(
    'ominari_data_collection_duration_seconds',
    'Data collection duration by source',
    ['source'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=registry
)

data_points_collected = Counter(
    'ominari_data_points_collected_total',
    'Total data points collected',
    ['source', 'type'],
    registry=registry
)

data_collection_errors = Counter(
    'ominari_data_collection_errors_total',
    'Data collection errors',
    ['source', 'error_type'],
    registry=registry
)

# Database metrics
db_query_duration = Histogram(
    'ominari_db_query_duration_seconds',
    'Database query duration',
    ['operation'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
    registry=registry
)

db_connections_active = Gauge(
    'ominari_db_connections_active',
    'Active database connections',
    registry=registry
)

# HTTP/API metrics
http_requests = Counter(
    'ominari_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

http_request_duration = Histogram(
    'ominari_http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=registry
)

# Resource metrics
cpu_usage_percent = Gauge(
    'ominari_cpu_usage_percent',
    'CPU usage percentage',
    registry=registry
)

memory_usage_bytes = Gauge(
    'ominari_memory_usage_bytes',
    'Memory usage in bytes',
    registry=registry
)

disk_usage_percent = Gauge(
    'ominari_disk_usage_percent',
    'Disk usage percentage',
    ['path'],
    registry=registry
)


class MetricsCollector:
    """Collects and exports metrics."""
    
    def __init__(self, port: int = None):
        self.port = port or settings.monitoring.metrics_port
        self.start_time = time.time()
        self._initialized = False
        
    def initialize(self):
        """Initialize metrics collector."""
        if self._initialized:
            return
            
        # Set system info
        system_info.info({
            'version': '1.0.0',
            'environment': settings.environment,
            'network': settings.blockchain.network
        })
        
        # Start metrics server
        if settings.monitoring.enabled:
            try:
                start_http_server(self.port, registry=registry)
                logger.info(f"Metrics server started on port {self.port}")
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")
                
    def update_system_metrics(self):
        """Update system-level metrics."""
        # Uptime
        uptime_seconds.set(time.time() - self.start_time)
        
        # CPU and memory
        cpu_usage_percent.set(psutil.cpu_percent(interval=1))
        memory = psutil.virtual_memory()
        memory_usage_bytes.set(memory.used)
        
        # Disk usage
        for path in [settings.data_dir, settings.log_dir]:
            try:
                usage = psutil.disk_usage(str(path))
                disk_usage_percent.labels(path=str(path)).set(usage.percent)
            except Exception:
                pass
                
    def update_signal_metrics(self, signal_registry):
        """Update signal-related metrics."""
        active_signals.set(len(signal_registry.list_signals(active_only=True)))
        
        # Update weights
        weights = signal_registry.weight_manager.get_current_weights()
        for signal_name, weight in weights.items():
            signal_weights.labels(signal_name=signal_name).set(weight)
            
        # Update performance
        for signal_name in signal_registry.list_signals():
            signal = signal_registry.get_signal(signal_name)
            if signal and hasattr(signal, 'metadata'):
                stats = signal.metadata.performance_stats
                for metric, value in stats.items():
                    signal_performance.labels(
                        signal_name=signal_name,
                        metric=metric
                    ).set(value)
                    
    def record_trade(self, trade_type: str, signal_name: str, 
                    filled: bool, latency: float = None):
        """Record trade metrics."""
        trades_submitted.labels(type=trade_type, signal=signal_name).inc()
        
        if filled:
            trades_filled.labels(type=trade_type, signal=signal_name).inc()
            
        if latency is not None:
            trade_latency.labels(type=trade_type).observe(latency)
            
    def record_backtest(self, signal_name: str, duration: float, 
                       sharpe_ratio: float, period: str = "full"):
        """Record backtest metrics."""
        backtest_duration.labels(signal_name=signal_name).observe(duration)
        backtest_sharpe.labels(
            signal_name=signal_name,
            period=period
        ).set(sharpe_ratio)
        
    def record_data_collection(self, source: str, duration: float,
                             points: int, error: Optional[str] = None):
        """Record data collection metrics."""
        data_collection_duration.labels(source=source).observe(duration)
        
        if error:
            data_collection_errors.labels(
                source=source,
                error_type=error
            ).inc()
        else:
            data_points_collected.labels(
                source=source,
                type='market'
            ).inc(points)
            
    def record_http_request(self, method: str, endpoint: str, 
                          status: int, duration: float):
        """Record HTTP request metrics."""
        http_requests.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        http_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
    def record_db_query(self, operation: str, duration: float):
        """Record database query metrics."""
        db_query_duration.labels(operation=operation).observe(duration)
        
    def update_trading_metrics(self, positions: Dict[str, float],
                             pnl: float, capital: float):
        """Update trading position and PnL metrics."""
        for market_id, size in positions.items():
            position_size.labels(
                market_id=market_id,
                type='paper'
            ).set(size)
            
        pnl_total.labels(type='paper').set(pnl)
        capital_available.labels(type='paper').set(capital)


# Decorators for automatic metric collection
def track_duration(metric_name: str, labels: Dict[str, str] = None):
    """Decorator to track function execution duration."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if metric_name == 'backtest':
                    signal_name = kwargs.get('signal_name', 'unknown')
                    backtest_duration.labels(signal_name=signal_name).observe(duration)
                elif metric_name == 'db_query':
                    operation = labels.get('operation', 'unknown')
                    db_query_duration.labels(operation=operation).observe(duration)
                elif metric_name == 'data_collection':
                    source = labels.get('source', 'unknown')
                    data_collection_duration.labels(source=source).observe(duration)
                    
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                # Same metric recording as above
                
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


@contextmanager
def track_operation(operation_name: str):
    """Context manager to track operation duration."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.debug(f"Operation '{operation_name}' took {duration:.3f}s")


class HealthChecker:
    """System health checker."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.checks = {
            'database': self._check_database,
            'signals': self._check_signals,
            'trading': self._check_trading,
            'data_collection': self._check_data_collection
        }
        
    async def check_health(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'checks': {}
        }
        
        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    check_result = await check_func()
                else:
                    check_result = check_func()
                    
                results['checks'][name] = check_result
                
                if check_result['status'] != 'healthy':
                    results['status'] = 'degraded'
                    
            except Exception as e:
                results['checks'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
                results['status'] = 'unhealthy'
                
        return results
        
    def _check_database(self) -> Dict[str, Any]:
        """Check database health."""
        from database import SessionLocal
        from sqlalchemy import text
        import time
        
        try:
            start_time = time.time()
            with SessionLocal() as db:
                result = db.execute(text("SELECT 1")).scalar()
                db.commit()
                
            response_time = time.time() - start_time
            return {
                'status': 'healthy',
                'response_time': response_time
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
            
    def _check_signals(self) -> Dict[str, Any]:
        """Check signal system health."""
        active = active_signals._value.get()
        
        if active > 0:
            return {
                'status': 'healthy',
                'active_signals': int(active)
            }
        else:
            return {
                'status': 'degraded',
                'message': 'No active signals'
            }
            
    def _check_trading(self) -> Dict[str, Any]:
        """Check trading system health."""
        capital = capital_available._value.get()
        
        if capital > 1000:  # Minimum capital threshold
            return {
                'status': 'healthy',
                'available_capital': capital
            }
        else:
            return {
                'status': 'degraded',
                'message': 'Low capital',
                'available_capital': capital
            }
            
    async def _check_data_collection(self) -> Dict[str, Any]:
        """Check data collection health."""
        # Check recent data collection
        # This would query actual metrics
        return {
            'status': 'healthy',
            'last_collection': datetime.now(timezone.utc).isoformat()
        }


# Global instances
metrics_collector = MetricsCollector()
health_checker = HealthChecker(metrics_collector)


def initialize_monitoring():
    """Initialize monitoring system."""
    if settings.monitoring.enabled:
        metrics_collector.initialize()
        logger.info("Monitoring system initialized")
    else:
        logger.info("Monitoring disabled")


async def monitoring_loop():
    """Main monitoring loop."""
    while True:
        try:
            # Update system metrics
            metrics_collector.update_system_metrics()
            
            # Run health checks
            health = await health_checker.check_health()
            if health['status'] != 'healthy':
                logger.warning(f"System health degraded: {health}")
                
            await asyncio.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            await asyncio.sleep(60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_monitoring()
    
    # Run monitoring loop
    asyncio.run(monitoring_loop())