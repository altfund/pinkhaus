#!/usr/bin/env python3
"""
Setup script for Prometheus and Grafana monitoring stack
"""

import os
import json
import yaml
import subprocess
import time
from pathlib import Path

def create_prometheus_config():
    """Create Prometheus configuration."""
    config = {
        'global': {
            'scrape_interval': '15s',
            'evaluation_interval': '15s',
            'external_labels': {
                'monitor': 'ominari-monitor'
            }
        },
        'alerting': {
            'alertmanagers': [{
                'static_configs': [{
                    'targets': ['localhost:9093']
                }]
            }]
        },
        'rule_files': [
            'alerts/*.yml'
        ],
        'scrape_configs': [
            {
                'job_name': 'prometheus',
                'static_configs': [{
                    'targets': ['localhost:9090']
                }]
            },
            {
                'job_name': 'ominari_api',
                'static_configs': [{
                    'targets': ['localhost:8888']
                }],
                'metrics_path': '/metrics'
            },
            {
                'job_name': 'node_exporter',
                'static_configs': [{
                    'targets': ['localhost:9100']
                }]
            },
            {
                'job_name': 'postgres_exporter',
                'static_configs': [{
                    'targets': ['localhost:9187']
                }]
            }
        ]
    }
    
    # Create monitoring directory
    Path('monitoring').mkdir(exist_ok=True)
    
    # Write Prometheus config
    with open('monitoring/prometheus.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Created monitoring/prometheus.yml")


def create_alert_rules():
    """Create Prometheus alert rules."""
    alerts = {
        'groups': [
            {
                'name': 'ominari_alerts',
                'interval': '30s',
                'rules': [
                    {
                        'alert': 'APIDown',
                        'expr': 'up{job="ominari_api"} == 0',
                        'for': '1m',
                        'labels': {
                            'severity': 'critical'
                        },
                        'annotations': {
                            'summary': 'Ominari API is down',
                            'description': 'API has been down for more than 1 minute'
                        }
                    },
                    {
                        'alert': 'HighErrorRate',
                        'expr': 'rate(ominari_api_errors_total[5m]) > 0.05',
                        'for': '5m',
                        'labels': {
                            'severity': 'warning'
                        },
                        'annotations': {
                            'summary': 'High API error rate',
                            'description': 'Error rate is above 5% for 5 minutes'
                        }
                    },
                    {
                        'alert': 'DatabaseSlow',
                        'expr': 'ominari_database_query_duration_seconds > 1',
                        'for': '5m',
                        'labels': {
                            'severity': 'warning'
                        },
                        'annotations': {
                            'summary': 'Database queries are slow',
                            'description': 'Database queries taking more than 1 second'
                        }
                    },
                    {
                        'alert': 'HighMemoryUsage',
                        'expr': 'node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes < 0.1',
                        'for': '5m',
                        'labels': {
                            'severity': 'warning'
                        },
                        'annotations': {
                            'summary': 'High memory usage',
                            'description': 'Less than 10% memory available'
                        }
                    },
                    {
                        'alert': 'DiskSpaceLow',
                        'expr': 'node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes < 0.1',
                        'for': '5m',
                        'labels': {
                            'severity': 'critical'
                        },
                        'annotations': {
                            'summary': 'Low disk space',
                            'description': 'Less than 10% disk space available'
                        }
                    }
                ]
            }
        ]
    }
    
    # Create alerts directory
    Path('monitoring/alerts').mkdir(exist_ok=True)
    
    # Write alert rules
    with open('monitoring/alerts/ominari.yml', 'w') as f:
        yaml.dump(alerts, f, default_flow_style=False)
    
    print("✅ Created monitoring/alerts/ominari.yml")


def create_grafana_dashboards():
    """Create Grafana dashboard configuration."""
    dashboard = {
        "dashboard": {
            "id": None,
            "title": "Ominari Trading System",
            "timezone": "browser",
            "schemaVersion": 16,
            "panels": [
                {
                    "id": 1,
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "type": "graph",
                    "title": "API Request Rate",
                    "targets": [{
                        "expr": "rate(ominari_api_requests_total[5m])",
                        "legendFormat": "{{method}} {{endpoint}}"
                    }]
                },
                {
                    "id": 2,
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    "type": "graph",
                    "title": "API Response Time",
                    "targets": [{
                        "expr": "histogram_quantile(0.95, rate(ominari_api_request_duration_seconds_bucket[5m]))",
                        "legendFormat": "95th percentile"
                    }]
                },
                {
                    "id": 3,
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                    "type": "graph",
                    "title": "Trading Performance",
                    "targets": [
                        {
                            "expr": "ominari_portfolio_value",
                            "legendFormat": "Portfolio Value"
                        },
                        {
                            "expr": "ominari_total_pnl",
                            "legendFormat": "Total P&L"
                        }
                    ]
                },
                {
                    "id": 4,
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                    "type": "graph",
                    "title": "System Resources",
                    "targets": [
                        {
                            "expr": "100 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes * 100)",
                            "legendFormat": "Memory Usage %"
                        },
                        {
                            "expr": "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
                            "legendFormat": "CPU Usage %"
                        }
                    ]
                },
                {
                    "id": 5,
                    "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16},
                    "type": "stat",
                    "title": "Open Positions",
                    "targets": [{
                        "expr": "ominari_open_positions"
                    }]
                },
                {
                    "id": 6,
                    "gridPos": {"h": 4, "w": 6, "x": 6, "y": 16},
                    "type": "stat",
                    "title": "Win Rate",
                    "targets": [{
                        "expr": "ominari_win_rate * 100"
                    }]
                },
                {
                    "id": 7,
                    "gridPos": {"h": 4, "w": 6, "x": 12, "y": 16},
                    "type": "stat",
                    "title": "Active Markets",
                    "targets": [{
                        "expr": "ominari_active_markets"
                    }]
                },
                {
                    "id": 8,
                    "gridPos": {"h": 4, "w": 6, "x": 18, "y": 16},
                    "type": "stat",
                    "title": "Database Size",
                    "targets": [{
                        "expr": "pg_database_size_bytes{datname=\"ominari\"}"
                    }]
                }
            ]
        }
    }
    
    # Create dashboards directory
    Path('monitoring/dashboards').mkdir(exist_ok=True)
    
    # Write dashboard
    with open('monitoring/dashboards/ominari.json', 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print("✅ Created monitoring/dashboards/ominari.json")


def create_docker_compose():
    """Create docker-compose.yml for monitoring stack."""
    compose = {
        'version': '3.8',
        'services': {
            'prometheus': {
                'image': 'prom/prometheus:latest',
                'container_name': 'ominari-prometheus',
                'ports': ['9090:9090'],
                'volumes': [
                    './monitoring/prometheus.yml:/etc/prometheus/prometheus.yml',
                    './monitoring/alerts:/etc/prometheus/alerts',
                    'prometheus_data:/prometheus'
                ],
                'command': [
                    '--config.file=/etc/prometheus/prometheus.yml',
                    '--storage.tsdb.path=/prometheus',
                    '--web.console.libraries=/usr/share/prometheus/console_libraries',
                    '--web.console.templates=/usr/share/prometheus/consoles',
                    '--web.enable-lifecycle'
                ],
                'restart': 'unless-stopped'
            },
            'grafana': {
                'image': 'grafana/grafana:latest',
                'container_name': 'ominari-grafana',
                'ports': ['3000:3000'],
                'volumes': [
                    'grafana_data:/var/lib/grafana',
                    './monitoring/dashboards:/var/lib/grafana/dashboards',
                    './monitoring/grafana-datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml',
                    './monitoring/grafana-dashboards.yml:/etc/grafana/provisioning/dashboards/dashboards.yml'
                ],
                'environment': {
                    'GF_SECURITY_ADMIN_PASSWORD': 'ominari123',
                    'GF_USERS_ALLOW_SIGN_UP': 'false'
                },
                'restart': 'unless-stopped',
                'depends_on': ['prometheus']
            },
            'node_exporter': {
                'image': 'prom/node-exporter:latest',
                'container_name': 'ominari-node-exporter',
                'ports': ['9100:9100'],
                'volumes': [
                    '/proc:/host/proc:ro',
                    '/sys:/host/sys:ro',
                    '/:/rootfs:ro'
                ],
                'command': [
                    '--path.procfs=/host/proc',
                    '--path.sysfs=/host/sys',
                    '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
                ],
                'restart': 'unless-stopped'
            },
            'alertmanager': {
                'image': 'prom/alertmanager:latest',
                'container_name': 'ominari-alertmanager',
                'ports': ['9093:9093'],
                'volumes': [
                    './monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml',
                    'alertmanager_data:/alertmanager'
                ],
                'command': [
                    '--config.file=/etc/alertmanager/alertmanager.yml',
                    '--storage.path=/alertmanager'
                ],
                'restart': 'unless-stopped'
            }
        },
        'volumes': {
            'prometheus_data': {},
            'grafana_data': {},
            'alertmanager_data': {}
        },
        'networks': {
            'default': {
                'name': 'ominari-monitoring'
            }
        }
    }
    
    # Write docker-compose file
    with open('monitoring/docker-compose.yml', 'w') as f:
        yaml.dump(compose, f, default_flow_style=False)
    
    print("✅ Created monitoring/docker-compose.yml")


def create_grafana_configs():
    """Create Grafana datasource and dashboard provisioning configs."""
    # Datasources config
    datasources = {
        'apiVersion': 1,
        'datasources': [{
            'name': 'Prometheus',
            'type': 'prometheus',
            'access': 'proxy',
            'url': 'http://prometheus:9090',
            'isDefault': True
        }]
    }
    
    with open('monitoring/grafana-datasources.yml', 'w') as f:
        yaml.dump(datasources, f)
    
    # Dashboard provisioning config
    dashboards = {
        'apiVersion': 1,
        'providers': [{
            'name': 'default',
            'orgId': 1,
            'folder': '',
            'type': 'file',
            'disableDeletion': False,
            'updateIntervalSeconds': 10,
            'options': {
                'path': '/var/lib/grafana/dashboards'
            }
        }]
    }
    
    with open('monitoring/grafana-dashboards.yml', 'w') as f:
        yaml.dump(dashboards, f)
    
    print("✅ Created Grafana provisioning configs")


def create_alertmanager_config():
    """Create Alertmanager configuration."""
    config = {
        'global': {
            'resolve_timeout': '5m'
        },
        'route': {
            'group_by': ['alertname', 'cluster', 'service'],
            'group_wait': '10s',
            'group_interval': '10s',
            'repeat_interval': '12h',
            'receiver': 'default'
        },
        'receivers': [{
            'name': 'default',
            'webhook_configs': [{
                'url': 'http://localhost:8888/api/alerts',
                'send_resolved': True
            }]
        }]
    }
    
    with open('monitoring/alertmanager.yml', 'w') as f:
        yaml.dump(config, f)
    
    print("✅ Created monitoring/alertmanager.yml")


def create_metrics_module():
    """Create Python module for Prometheus metrics."""
    code = '''#!/usr/bin/env python3
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
'''
    
    with open('monitoring/metrics.py', 'w') as f:
        f.write(code)
    
    print("✅ Created monitoring/metrics.py")


def create_setup_script():
    """Create setup script for monitoring."""
    script = '''#!/bin/bash
# Setup script for Ominari monitoring

echo "=== Ominari Monitoring Setup ==="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not installed. Please install docker-compose first."
    exit 1
fi

# Start monitoring stack
echo "Starting monitoring stack..."
cd monitoring
docker-compose up -d

# Wait for services to start
echo "Waiting for services to start..."
sleep 10

# Check service status
echo ""
echo "=== Service Status ==="
docker-compose ps

echo ""
echo "=== Access URLs ==="
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000 (admin/ominari123)"
echo "Alertmanager: http://localhost:9093"

echo ""
echo "=== Next Steps ==="
echo "1. Update web_monitor_auth.py to expose /metrics endpoint"
echo "2. Import dashboard in Grafana"
echo "3. Configure alert notifications in Alertmanager"

echo ""
echo "✅ Monitoring setup complete!"
'''
    
    with open('monitoring/setup.sh', 'w') as f:
        f.write(script)
    
    # Make executable
    os.chmod('monitoring/setup.sh', 0o755)
    
    print("✅ Created monitoring/setup.sh")


def main():
    """Set up complete monitoring stack."""
    print("=== Setting up Prometheus & Grafana Monitoring ===\n")
    
    # Create all configurations
    create_prometheus_config()
    create_alert_rules()
    create_grafana_dashboards()
    create_docker_compose()
    create_grafana_configs()
    create_alertmanager_config()
    create_metrics_module()
    create_setup_script()
    
    print("\n=== Setup Complete ===")
    print("\nTo start the monitoring stack:")
    print("  cd monitoring && ./setup.sh")
    
    print("\nTo integrate with your API:")
    print("  1. Add prometheus_client to dependencies")
    print("  2. Import metrics from monitoring/metrics.py")
    print("  3. Add /metrics endpoint to web_monitor_auth.py")
    
    print("\nExample integration:")
    print("""
from prometheus_client import generate_latest
from monitoring.metrics import track_api_request, update_trading_metrics

@app.route('/metrics')
def metrics():
    return generate_latest()

@app.route('/api/status')
@track_api_request('/api/status')
def status():
    # Your existing code
    pass
""")


if __name__ == "__main__":
    main()