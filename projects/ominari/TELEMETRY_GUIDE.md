# Telemetry and Monitoring Guide - Ominari Trading System

## Overview

The Ominari trading system now includes comprehensive monitoring and telemetry using OpenTelemetry, Prometheus, and Grafana. This provides:

- **Distributed Tracing**: Track requests across components
- **Metrics Collection**: Business and technical metrics
- **Log Aggregation**: Structured logging with context
- **Alerting**: Proactive notifications for issues
- **Visualization**: Real-time dashboards

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Trading System │────▶│ OpenTelemetry│────▶│ Prometheus  │
│   (Instrumented)│     │   Collector  │     │   (Metrics) │
└─────────────────┘     └──────────────┘     └─────────────┘
                               │                      │
                               │                      ▼
                               │              ┌─────────────┐
                               │              │   Grafana   │
                               │              │(Dashboards) │
                               │              └─────────────┘
                               ▼
                        ┌──────────────┐      ┌─────────────┐
                        │    Jaeger    │      │Alertmanager │
                        │   (Traces)   │      │  (Alerts)   │
                        └──────────────┘      └─────────────┘
```

## Quick Start

### 1. Enable Telemetry in Code

```python
# In your main application
from telemetry import initialize_telemetry
from instrumented_components import setup_instrumentation

# Initialize telemetry
telemetry = initialize_telemetry()

# Setup component instrumentation
setup_instrumentation()
```

### 2. Start Monitoring Stack

```bash
# Generate monitoring configuration
python monitoring_stack.py

# Start the stack
cd monitoring
./setup.sh
```

### 3. Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9091
- **Alertmanager**: http://localhost:9093

## Metrics Reference

### Trading Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `trading.bets.placed` | Counter | Total bets placed | market, strategy |
| `trading.bets.won` | Counter | Winning bets | market, strategy |
| `trading.bets.lost` | Counter | Losing bets | market, strategy |
| `trading.stake.total` | Counter | Total amount staked | market, strategy |
| `trading.pnl.total` | UpDownCounter | Total P&L | market, strategy |
| `trading.positions.open` | Gauge | Current open positions | session |

### Risk Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `risk.exposure.current` | Gauge | Current portfolio exposure | session |
| `risk.drawdown.current` | Gauge | Current drawdown % | session |
| `risk.violations.total` | Counter | Risk limit violations | type, severity |

### Signal Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `signals.accuracy` | Histogram | Signal prediction accuracy | signal |
| `signals.latency` | Histogram | Signal generation time | signal |
| `signals.weight.current` | Gauge | Current signal weight | signal |

### System Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `db.query.duration` | Histogram | Database query time | operation, table |
| `api.request.duration` | Histogram | API request time | endpoint, method |
| `system.health` | Gauge | System health (0-1) | component |

## Instrumentation Examples

### 1. Trace a Function

```python
from telemetry import traced

@traced("my_operation")
def process_data(data):
    # Your code here
    return result
```

### 2. Record Custom Metrics

```python
from telemetry import get_telemetry

telemetry = get_telemetry()

# Record a bet
telemetry.record_bet_placed(
    amount=100.0,
    market="Soccer",
    strategy="kelly"
)

# Record settlement
telemetry.record_bet_settled(
    pnl=15.0,
    won=True,
    market="Soccer",
    strategy="kelly"
)
```

### 3. Add Span Context

```python
from telemetry import get_telemetry

telemetry = get_telemetry()

with telemetry.span("complex_operation", {
    "user_id": "123",
    "operation_type": "batch_process"
}) as span:
    # Your operation
    span.set_attribute("items_processed", 100)
    span.add_event("milestone_reached")
```

### 4. Time Operations

```python
from telemetry import timed

@timed("db")  # Records to db.query.duration
def query_database():
    # Database operation
    pass
```

## Alert Configuration

### Default Alerts

1. **Risk Alerts**
   - High Drawdown (>10%)
   - Critical Drawdown (>15%)
   - Daily Loss Limit

2. **Trading Alerts**
   - Low Win Rate (<40%)
   - No Recent Bets
   - High Bet Rate

3. **System Alerts**
   - Slow Database Queries
   - High API Latency
   - System Down

### Custom Alert Example

```yaml
- alert: CustomMetric
  expr: 'my_custom_metric > 100'
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Custom metric high: {{ $value }}"
```

## Dashboard Customization

### Add Custom Panel

```json
{
  "title": "My Custom Metric",
  "type": "timeseries",
  "targets": [{
    "expr": "rate(my_custom_metric[5m])"
  }],
  "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
}
```

### Query Examples

```promql
# Win rate over last hour
rate(trading_bets_won[1h]) / rate(trading_bets_placed[1h])

# P95 signal latency
histogram_quantile(0.95, signals_latency_bucket)

# Daily P&L
increase(trading_pnl_total[1d])

# Average exposure by session
avg(risk_exposure_current) by (session)
```

## Production Configuration

### Environment Variables

```bash
# OpenTelemetry Configuration
export OTEL_SERVICE_NAME="ominari-trading"
export OTEL_SERVICE_VERSION="1.0.0"
export OTEL_ENVIRONMENT="production"
export OTEL_EXPORTER_OTLP_ENDPOINT="localhost:4317"

# Feature Flags
export OTEL_EXPORT_TRACES="true"
export OTEL_EXPORT_METRICS="true"
export OTEL_EXPORT_LOGS="true"

# Prometheus
export PROMETHEUS_PORT="9090"
export ENABLE_PROMETHEUS="true"

# Debug
export OTEL_CONSOLE_EXPORT="false"
```

### Performance Considerations

1. **Sampling**: For high-volume systems, implement trace sampling
2. **Batching**: Metrics are batched and exported periodically
3. **Cardinality**: Limit label values to prevent metric explosion
4. **Retention**: Configure appropriate data retention policies

### Security

1. **Authentication**: Enable auth on Grafana/Prometheus
2. **TLS**: Use secure connections for exporters
3. **Secrets**: Store credentials securely
4. **Network**: Restrict access to monitoring endpoints

## Troubleshooting

### No Metrics Appearing

1. Check telemetry is initialized:
   ```python
   telemetry = get_telemetry()
   print(telemetry._initialized)  # Should be True
   ```

2. Verify Prometheus is scraping:
   - Visit http://localhost:9091/targets
   - Check target status is "UP"

3. Check for errors in logs

### Missing Traces

1. Verify OTLP endpoint is accessible
2. Check trace sampling configuration
3. Look for errors in span processors

### High Cardinality

Symptoms: Prometheus memory usage high

Solutions:
1. Reduce label combinations
2. Aggregate before exporting
3. Use recording rules

## Best Practices

1. **Meaningful Metrics**: Track business-relevant metrics
2. **Consistent Labels**: Use standard label names
3. **Appropriate Granularity**: Balance detail vs overhead
4. **Error Handling**: Always record errors in spans
5. **Context Propagation**: Pass trace context between services
6. **Alert Fatigue**: Tune alerts to reduce noise
7. **Dashboard Organization**: Group related metrics
8. **Documentation**: Document custom metrics

## Integration with Existing Tools

### Datadog
```python
from opentelemetry.exporter.datadog import DatadogExporter
```

### New Relic
```python
from opentelemetry.exporter.newrelic import NewRelicExporter
```

### CloudWatch
```python
from opentelemetry.exporter.cloudwatch import CloudWatchExporter
```

## Conclusion

The telemetry system provides comprehensive observability for the Ominari trading system. It enables:

- Real-time monitoring of trading performance
- Proactive alerting for risk conditions
- Performance optimization through tracing
- Historical analysis of system behavior

Regular monitoring and alert tuning ensure optimal system performance and risk management.