#!/usr/bin/env python3
"""
Comprehensive Telemetry System for Ominari Trading.
Implements OpenTelemetry for metrics, traces, and logs.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from functools import wraps
from datetime import datetime, timezone

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace.status import Status, StatusCode

logger = logging.getLogger(__name__)


class TelemetryConfig:
    """Telemetry configuration."""
    
    def __init__(self):
        self.service_name = os.getenv("OTEL_SERVICE_NAME", "ominari-trading")
        self.service_version = os.getenv("OTEL_SERVICE_VERSION", "1.0.0")
        self.environment = os.getenv("OTEL_ENVIRONMENT", "production")
        
        # Exporters
        self.otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
        self.export_traces = os.getenv("OTEL_EXPORT_TRACES", "true").lower() == "true"
        self.export_metrics = os.getenv("OTEL_EXPORT_METRICS", "true").lower() == "true"
        self.export_logs = os.getenv("OTEL_EXPORT_LOGS", "true").lower() == "true"
        
        # Prometheus metrics
        self.prometheus_port = int(os.getenv("PROMETHEUS_PORT", "9090"))
        self.enable_prometheus = os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true"
        
        # Console export for debugging
        self.console_export = os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true"


class TelemetryManager:
    """Manages all telemetry components."""
    
    def __init__(self, config: Optional[TelemetryConfig] = None):
        self.config = config or TelemetryConfig()
        self.tracer = None
        self.meter = None
        self._metrics = {}
        self._initialized = False
        
    def initialize(self):
        """Initialize telemetry providers."""
        if self._initialized:
            return
            
        # Create resource
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: self.config.service_name,
            ResourceAttributes.SERVICE_VERSION: self.config.service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.environment,
            "service.instance.id": os.getenv("HOSTNAME", "localhost"),
        })
        
        # Setup tracing
        if self.config.export_traces:
            self._setup_tracing(resource)
            
        # Setup metrics
        if self.config.export_metrics:
            self._setup_metrics(resource)
            
        # Instrument libraries
        self._instrument_libraries()
        
        # Setup custom metrics
        self._setup_custom_metrics()
        
        self._initialized = True
        logger.info(f"Telemetry initialized for {self.config.service_name}")
        
    def _setup_tracing(self, resource: Resource):
        """Setup distributed tracing."""
        # Create tracer provider
        provider = TracerProvider(resource=resource)
        
        # Add exporters
        if self.config.console_export:
            provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )
        
        if self.config.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint,
                insecure=True
            )
            provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
        
        # Set global provider
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(__name__)
        
    def _setup_metrics(self, resource: Resource):
        """Setup metrics collection."""
        readers = []
        
        # Console exporter for debugging
        if self.config.console_export:
            readers.append(
                PeriodicExportingMetricReader(
                    ConsoleMetricExporter(),
                    export_interval_millis=5000
                )
            )
        
        # OTLP exporter
        if self.config.otlp_endpoint:
            readers.append(
                PeriodicExportingMetricReader(
                    OTLPMetricExporter(
                        endpoint=self.config.otlp_endpoint,
                        insecure=True
                    ),
                    export_interval_millis=10000
                )
            )
        
        # Prometheus exporter
        if self.config.enable_prometheus:
            readers.append(
                PrometheusMetricReader(
                    port=self.config.prometheus_port
                )
            )
        
        # Create meter provider
        provider = MeterProvider(
            resource=resource,
            metric_readers=readers
        )
        
        # Set global provider
        metrics.set_meter_provider(provider)
        self.meter = metrics.get_meter(__name__)
        
    def _instrument_libraries(self):
        """Instrument common libraries."""
        # SQLAlchemy
        try:
            from database_v2 import engine
            SQLAlchemyInstrumentor().instrument(
                engine=engine,
                service="ominari-db"
            )
        except:
            logger.warning("Failed to instrument SQLAlchemy")
        
        # Requests
        RequestsInstrumentor().instrument(
            service="ominari-http"
        )
        
        # Logging
        if self.config.export_logs:
            LoggingInstrumentor().instrument()
            
    def _setup_custom_metrics(self):
        """Setup custom business metrics."""
        if not self.meter:
            return
            
        # Trading metrics
        self._metrics['bets_placed'] = self.meter.create_counter(
            name="trading.bets.placed",
            description="Total number of bets placed",
            unit="1"
        )
        
        self._metrics['bets_won'] = self.meter.create_counter(
            name="trading.bets.won",
            description="Total number of winning bets",
            unit="1"
        )
        
        self._metrics['bets_lost'] = self.meter.create_counter(
            name="trading.bets.lost",
            description="Total number of losing bets",
            unit="1"
        )
        
        self._metrics['total_staked'] = self.meter.create_counter(
            name="trading.stake.total",
            description="Total amount staked",
            unit="USD"
        )
        
        self._metrics['total_pnl'] = self.meter.create_up_down_counter(
            name="trading.pnl.total",
            description="Total profit and loss",
            unit="USD"
        )
        
        # Risk metrics
        self._metrics['current_exposure'] = self.meter.create_observable_gauge(
            name="risk.exposure.current",
            description="Current portfolio exposure",
            unit="USD",
            callbacks=[self._get_current_exposure]
        )
        
        self._metrics['current_drawdown'] = self.meter.create_observable_gauge(
            name="risk.drawdown.current",
            description="Current drawdown percentage",
            unit="%",
            callbacks=[self._get_current_drawdown]
        )
        
        self._metrics['open_positions'] = self.meter.create_observable_gauge(
            name="trading.positions.open",
            description="Number of open positions",
            unit="1",
            callbacks=[self._get_open_positions]
        )
        
        # Signal metrics
        self._metrics['signal_accuracy'] = self.meter.create_histogram(
            name="signals.accuracy",
            description="Signal prediction accuracy",
            unit="ratio"
        )
        
        self._metrics['signal_latency'] = self.meter.create_histogram(
            name="signals.latency",
            description="Signal generation latency",
            unit="ms"
        )
        
        # System metrics
        self._metrics['db_query_duration'] = self.meter.create_histogram(
            name="db.query.duration",
            description="Database query duration",
            unit="ms"
        )
        
        self._metrics['api_request_duration'] = self.meter.create_histogram(
            name="api.request.duration",
            description="API request duration",
            unit="ms"
        )
        
    def _get_current_exposure(self, options):
        """Callback for current exposure metric."""
        # Would fetch from database
        yield metrics.Observation(0.0, {"session": "current"})
        
    def _get_current_drawdown(self, options):
        """Callback for current drawdown metric."""
        # Would fetch from database
        yield metrics.Observation(0.0, {"session": "current"})
        
    def _get_open_positions(self, options):
        """Callback for open positions metric."""
        # Would fetch from database
        yield metrics.Observation(0, {"session": "current"})
    
    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a traced span."""
        if not self.tracer:
            yield
            return
            
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                span.set_attributes(attributes)
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)
                raise
                
    def record_bet_placed(self, amount: float, market: str, strategy: str):
        """Record a bet placement."""
        if 'bets_placed' in self._metrics:
            self._metrics['bets_placed'].add(
                1,
                {"market": market, "strategy": strategy}
            )
        if 'total_staked' in self._metrics:
            self._metrics['total_staked'].add(
                amount,
                {"market": market, "strategy": strategy}
            )
            
    def record_bet_settled(self, pnl: float, won: bool, market: str, strategy: str):
        """Record a bet settlement."""
        labels = {"market": market, "strategy": strategy}
        
        if won and 'bets_won' in self._metrics:
            self._metrics['bets_won'].add(1, labels)
        elif not won and 'bets_lost' in self._metrics:
            self._metrics['bets_lost'].add(1, labels)
            
        if 'total_pnl' in self._metrics:
            self._metrics['total_pnl'].add(pnl, labels)
            
    def record_signal_accuracy(self, accuracy: float, signal_name: str):
        """Record signal accuracy."""
        if 'signal_accuracy' in self._metrics:
            self._metrics['signal_accuracy'].record(
                accuracy,
                {"signal": signal_name}
            )
            
    def record_latency(self, duration_ms: float, operation: str):
        """Record operation latency."""
        if operation == "signal" and 'signal_latency' in self._metrics:
            self._metrics['signal_latency'].record(duration_ms)
        elif operation == "db" and 'db_query_duration' in self._metrics:
            self._metrics['db_query_duration'].record(duration_ms)
        elif operation == "api" and 'api_request_duration' in self._metrics:
            self._metrics['api_request_duration'].record(duration_ms)


# Global telemetry instance
_telemetry = TelemetryManager()


def initialize_telemetry(config: Optional[TelemetryConfig] = None):
    """Initialize global telemetry."""
    global _telemetry
    if config:
        _telemetry = TelemetryManager(config)
    _telemetry.initialize()
    return _telemetry


def get_telemetry() -> TelemetryManager:
    """Get global telemetry instance."""
    return _telemetry


# Decorators for instrumentation
def traced(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """Decorator to trace function execution."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"
            span_attrs = attributes or {}
            
            with _telemetry.span(span_name, span_attrs):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def timed(metric_name: str = "db"):
    """Decorator to time function execution."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.time() - start) * 1000
                _telemetry.record_latency(duration_ms, metric_name)
        return wrapper
    return decorator


# Context managers for common operations
@contextmanager
def trace_db_operation(operation: str, table: str):
    """Trace database operation."""
    with _telemetry.span(
        f"db.{operation}",
        {"db.table": table, "db.operation": operation}
    ):
        start = time.time()
        yield
        duration_ms = (time.time() - start) * 1000
        _telemetry.record_latency(duration_ms, "db")


@contextmanager
def trace_signal_generation(signal_name: str):
    """Trace signal generation."""
    with _telemetry.span(
        f"signal.generate",
        {"signal.name": signal_name}
    ):
        start = time.time()
        yield
        duration_ms = (time.time() - start) * 1000
        _telemetry.record_latency(duration_ms, "signal")


@contextmanager
def trace_api_call(endpoint: str, method: str):
    """Trace API call."""
    with _telemetry.span(
        f"api.{method}",
        {"api.endpoint": endpoint, "api.method": method}
    ):
        start = time.time()
        yield
        duration_ms = (time.time() - start) * 1000
        _telemetry.record_latency(duration_ms, "api")


def demonstrate_telemetry():
    """Demonstrate telemetry features."""
    # Initialize with console export
    config = TelemetryConfig()
    config.console_export = True
    initialize_telemetry(config)
    
    telemetry = get_telemetry()
    
    print("=== Telemetry Demonstration ===\n")
    
    # Trace a betting operation
    with telemetry.span("demo.place_bet", {"demo": True}):
        print("Simulating bet placement...")
        time.sleep(0.1)
        
        # Record metrics
        telemetry.record_bet_placed(
            amount=100.0,
            market="Soccer",
            strategy="kelly"
        )
        
    # Trace signal generation
    with trace_signal_generation("implied_probability"):
        print("Generating signal...")
        time.sleep(0.05)
        telemetry.record_signal_accuracy(0.65, "implied_probability")
    
    # Trace database operation
    with trace_db_operation("select", "positions"):
        print("Querying database...")
        time.sleep(0.02)
    
    # Simulate bet settlement
    telemetry.record_bet_settled(
        pnl=15.0,
        won=True,
        market="Soccer",
        strategy="kelly"
    )
    
    print("\nTelemetry data exported to configured backends")
    print(f"Prometheus metrics available at http://localhost:{config.prometheus_port}/metrics")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_telemetry()