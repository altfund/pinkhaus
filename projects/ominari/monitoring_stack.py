#!/usr/bin/env python3
"""
Monitoring Stack Configuration and Management.
Sets up Prometheus, Grafana, and alerts.
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    expression: str
    duration: str = "5m"
    severity: str = "warning"
    description: str = ""
    annotations: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class PrometheusConfig:
    """Prometheus configuration."""
    scrape_interval: str = "15s"
    evaluation_interval: str = "15s"
    retention_time: str = "30d"
    storage_path: str = "./prometheus_data"
    port: int = 9090
    
    def generate_config(self) -> Dict:
        """Generate prometheus.yml configuration."""
        return {
            'global': {
                'scrape_interval': self.scrape_interval,
                'evaluation_interval': self.evaluation_interval
            },
            'storage': {
                'tsdb': {
                    'path': self.storage_path,
                    'retention.time': self.retention_time
                }
            },
            'scrape_configs': [
                {
                    'job_name': 'ominari-trading',
                    'static_configs': [{
                        'targets': [f'localhost:{self.port}']
                    }]
                },
                {
                    'job_name': 'ominari-database',
                    'static_configs': [{
                        'targets': ['localhost:9187']  # Postgres exporter
                    }]
                }
            ],
            'rule_files': [
                'alerts/*.yml'
            ]
        }


def create_alert_rules() -> List[AlertRule]:
    """Create default alert rules."""
    return [
        # Risk alerts
        AlertRule(
            name="HighDrawdown",
            expression='risk_drawdown_current > 0.10',
            duration="5m",
            severity="warning",
            description="Portfolio drawdown exceeds 10%",
            annotations={
                "summary": "High drawdown detected: {{ $value | humanizePercentage }}"
            }
        ),
        AlertRule(
            name="CriticalDrawdown",
            expression='risk_drawdown_current > 0.15',
            duration="1m",
            severity="critical",
            description="Portfolio drawdown exceeds 15% - kill switch threshold",
            annotations={
                "summary": "CRITICAL: Drawdown at {{ $value | humanizePercentage }}"
            }
        ),
        AlertRule(
            name="DailyLossLimit",
            expression='increase(trading_pnl_total[1d]) < -500',
            duration="1m",
            severity="warning",
            description="Daily loss exceeds $500",
            annotations={
                "summary": "Daily loss: ${{ $value | humanize }}"
            }
        ),
        
        # Trading alerts
        AlertRule(
            name="LowWinRate",
            expression='rate(trading_bets_won[1h]) / rate(trading_bets_placed[1h]) < 0.40',
            duration="30m",
            severity="warning",
            description="Win rate below 40% for 30 minutes",
            annotations={
                "summary": "Low win rate: {{ $value | humanizePercentage }}"
            }
        ),
        AlertRule(
            name="NoRecentBets",
            expression='rate(trading_bets_placed[30m]) == 0',
            duration="30m",
            severity="info",
            description="No bets placed in last 30 minutes",
            annotations={
                "summary": "Trading appears to be idle"
            }
        ),
        AlertRule(
            name="HighBetRate",
            expression='rate(trading_bets_placed[5m]) > 10',
            duration="5m",
            severity="warning",
            description="Placing more than 10 bets per 5 minutes",
            annotations={
                "summary": "High betting rate: {{ $value | humanize }} bets/5m"
            }
        ),
        
        # Signal alerts
        AlertRule(
            name="SignalLatencyHigh",
            expression='histogram_quantile(0.95, signals_latency_bucket) > 1000',
            duration="10m",
            severity="warning",
            description="Signal generation P95 latency above 1 second",
            annotations={
                "summary": "Signal latency P95: {{ $value | humanizeDuration }}"
            }
        ),
        AlertRule(
            name="SignalAccuracyLow",
            expression='avg_over_time(signals_accuracy[1h]) < 0.52',
            duration="1h",
            severity="info",
            description="Average signal accuracy below 52%",
            annotations={
                "summary": "Signal accuracy: {{ $value | humanizePercentage }}"
            }
        ),
        
        # System alerts
        AlertRule(
            name="DatabaseSlowQueries",
            expression='histogram_quantile(0.95, db_query_duration_bucket) > 500',
            duration="10m",
            severity="warning",
            description="Database P95 query time above 500ms",
            annotations={
                "summary": "Slow DB queries: P95 {{ $value | humanizeDuration }}"
            }
        ),
        AlertRule(
            name="APILatencyHigh",
            expression='histogram_quantile(0.95, api_request_duration_bucket) > 2000',
            duration="5m",
            severity="warning",
            description="API P95 latency above 2 seconds",
            annotations={
                "summary": "API latency P95: {{ $value | humanizeDuration }}"
            }
        ),
        AlertRule(
            name="SystemDown",
            expression='up{job="ominari-trading"} == 0',
            duration="1m",
            severity="critical",
            description="Trading system is down",
            annotations={
                "summary": "CRITICAL: Trading system not responding"
            }
        )
    ]


def create_grafana_dashboard() -> Dict:
    """Create Grafana dashboard configuration."""
    return {
        "dashboard": {
            "title": "Ominari Trading System",
            "timezone": "utc",
            "refresh": "10s",
            "panels": [
                # Overview row
                {
                    "title": "System Overview",
                    "type": "row",
                    "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0}
                },
                
                # Key metrics
                {
                    "title": "Total P&L",
                    "type": "stat",
                    "targets": [{
                        "expr": "trading_pnl_total"
                    }],
                    "gridPos": {"h": 4, "w": 6, "x": 0, "y": 1}
                },
                {
                    "title": "Current Drawdown",
                    "type": "gauge",
                    "targets": [{
                        "expr": "risk_drawdown_current * 100"
                    }],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 10},
                                    {"color": "red", "value": 15}
                                ]
                            }
                        }
                    },
                    "gridPos": {"h": 4, "w": 6, "x": 6, "y": 1}
                },
                {
                    "title": "Win Rate",
                    "type": "stat",
                    "targets": [{
                        "expr": "rate(trading_bets_won[1h]) / rate(trading_bets_placed[1h]) * 100"
                    }],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent"
                        }
                    },
                    "gridPos": {"h": 4, "w": 6, "x": 12, "y": 1}
                },
                {
                    "title": "Open Positions",
                    "type": "stat",
                    "targets": [{
                        "expr": "trading_positions_open"
                    }],
                    "gridPos": {"h": 4, "w": 6, "x": 18, "y": 1}
                },
                
                # Trading activity
                {
                    "title": "Trading Activity",
                    "type": "row",
                    "gridPos": {"h": 1, "w": 24, "x": 0, "y": 5}
                },
                {
                    "title": "Bets Placed",
                    "type": "timeseries",
                    "targets": [{
                        "expr": "rate(trading_bets_placed[5m])"
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 6}
                },
                {
                    "title": "P&L Over Time",
                    "type": "timeseries",
                    "targets": [{
                        "expr": "trading_pnl_total"
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 6}
                },
                
                # Risk metrics
                {
                    "title": "Risk Management",
                    "type": "row",
                    "gridPos": {"h": 1, "w": 24, "x": 0, "y": 14}
                },
                {
                    "title": "Portfolio Exposure",
                    "type": "timeseries",
                    "targets": [{
                        "expr": "risk_exposure_current"
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 15}
                },
                {
                    "title": "Drawdown History",
                    "type": "timeseries",
                    "targets": [{
                        "expr": "risk_drawdown_current * 100"
                    }],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent"
                        }
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 15}
                },
                
                # Signal performance
                {
                    "title": "Signal Performance",
                    "type": "row",
                    "gridPos": {"h": 1, "w": 24, "x": 0, "y": 23}
                },
                {
                    "title": "Signal Accuracy by Type",
                    "type": "bargauge",
                    "targets": [{
                        "expr": "avg_over_time(signals_accuracy[1h]) by (signal)"
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24}
                },
                {
                    "title": "Signal Latency P95",
                    "type": "timeseries",
                    "targets": [{
                        "expr": "histogram_quantile(0.95, signals_latency_bucket)"
                    }],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "ms"
                        }
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24}
                }
            ]
        }
    }


def create_docker_compose() -> Dict:
    """Create docker-compose configuration for monitoring stack."""
    return {
        'version': '3.8',
        'services': {
            'prometheus': {
                'image': 'prom/prometheus:latest',
                'ports': ['9091:9090'],
                'volumes': [
                    './prometheus.yml:/etc/prometheus/prometheus.yml',
                    './alerts:/etc/prometheus/alerts',
                    './prometheus_data:/prometheus'
                ],
                'command': [
                    '--config.file=/etc/prometheus/prometheus.yml',
                    '--storage.tsdb.path=/prometheus',
                    '--web.console.libraries=/usr/share/prometheus/console_libraries',
                    '--web.console.templates=/usr/share/prometheus/consoles'
                ],
                'restart': 'unless-stopped'
            },
            'grafana': {
                'image': 'grafana/grafana:latest',
                'ports': ['3000:3000'],
                'volumes': [
                    './grafana_data:/var/lib/grafana',
                    './grafana/dashboards:/etc/grafana/provisioning/dashboards',
                    './grafana/datasources:/etc/grafana/provisioning/datasources'
                ],
                'environment': {
                    'GF_SECURITY_ADMIN_PASSWORD': 'admin',
                    'GF_USERS_ALLOW_SIGN_UP': 'false'
                },
                'restart': 'unless-stopped',
                'depends_on': ['prometheus']
            },
            'alertmanager': {
                'image': 'prom/alertmanager:latest',
                'ports': ['9093:9093'],
                'volumes': [
                    './alertmanager.yml:/etc/alertmanager/config.yml',
                    './alertmanager_data:/alertmanager'
                ],
                'command': [
                    '--config.file=/etc/alertmanager/config.yml',
                    '--storage.path=/alertmanager'
                ],
                'restart': 'unless-stopped'
            },
            'postgres_exporter': {
                'image': 'prometheuscommunity/postgres-exporter:latest',
                'ports': ['9187:9187'],
                'environment': {
                    'DATA_SOURCE_NAME': 'postgresql://user:password@host:5432/dbname?sslmode=disable'
                },
                'restart': 'unless-stopped'
            }
        },
        'volumes': {
            'prometheus_data': {},
            'grafana_data': {},
            'alertmanager_data': {}
        }
    }


def setup_monitoring_stack(output_dir: str = "./monitoring"):
    """Setup complete monitoring stack configuration."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/alerts", exist_ok=True)
    os.makedirs(f"{output_dir}/grafana/dashboards", exist_ok=True)
    os.makedirs(f"{output_dir}/grafana/datasources", exist_ok=True)
    
    # Prometheus config
    prometheus_config = PrometheusConfig()
    with open(f"{output_dir}/prometheus.yml", 'w') as f:
        yaml.dump(prometheus_config.generate_config(), f)
    
    # Alert rules
    alert_rules = create_alert_rules()
    alerts_config = {
        'groups': [{
            'name': 'ominari_trading',
            'interval': '30s',
            'rules': [
                {
                    'alert': rule.name,
                    'expr': rule.expression,
                    'for': rule.duration,
                    'labels': {
                        'severity': rule.severity,
                        **rule.labels
                    },
                    'annotations': {
                        'description': rule.description,
                        **rule.annotations
                    }
                }
                for rule in alert_rules
            ]
        }]
    }
    
    with open(f"{output_dir}/alerts/trading.yml", 'w') as f:
        yaml.dump(alerts_config, f)
    
    # Alertmanager config
    alertmanager_config = {
        'route': {
            'group_by': ['alertname', 'severity'],
            'group_wait': '10s',
            'group_interval': '10s',
            'repeat_interval': '1h',
            'receiver': 'web.hook'
        },
        'receivers': [{
            'name': 'web.hook',
            'webhook_configs': [{
                'url': 'http://localhost:5001/alerts'
            }]
        }]
    }
    
    with open(f"{output_dir}/alertmanager.yml", 'w') as f:
        yaml.dump(alertmanager_config, f)
    
    # Grafana datasource
    datasource_config = {
        'apiVersion': 1,
        'datasources': [{
            'name': 'Prometheus',
            'type': 'prometheus',
            'access': 'proxy',
            'url': 'http://prometheus:9090',
            'basicAuth': False,
            'isDefault': True,
            'editable': True
        }]
    }
    
    with open(f"{output_dir}/grafana/datasources/prometheus.yml", 'w') as f:
        yaml.dump(datasource_config, f)
    
    # Grafana dashboard
    dashboard = create_grafana_dashboard()
    dashboard_provision = {
        'apiVersion': 1,
        'providers': [{
            'name': 'default',
            'orgId': 1,
            'folder': '',
            'type': 'file',
            'disableDeletion': False,
            'updateIntervalSeconds': 10,
            'options': {
                'path': '/etc/grafana/provisioning/dashboards'
            }
        }]
    }
    
    with open(f"{output_dir}/grafana/dashboards/dashboard.yml", 'w') as f:
        yaml.dump(dashboard_provision, f)
        
    with open(f"{output_dir}/grafana/dashboards/ominari.json", 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    # Docker compose
    docker_config = create_docker_compose()
    with open(f"{output_dir}/docker-compose.yml", 'w') as f:
        yaml.dump(docker_config, f)
    
    # Setup script
    setup_script = """#!/bin/bash
# Ominari Monitoring Stack Setup

echo "Setting up Ominari monitoring stack..."

# Create data directories
mkdir -p prometheus_data grafana_data alertmanager_data

# Set permissions
chmod 777 prometheus_data grafana_data alertmanager_data

# Start services
docker-compose up -d

echo "Monitoring stack started!"
echo "Prometheus: http://localhost:9091"
echo "Grafana: http://localhost:3000 (admin/admin)"
echo "Alertmanager: http://localhost:9093"
"""
    
    with open(f"{output_dir}/setup.sh", 'w') as f:
        f.write(setup_script)
    os.chmod(f"{output_dir}/setup.sh", 0o755)
    
    print(f"Monitoring stack configuration created in {output_dir}/")
    print("\nTo start the monitoring stack:")
    print(f"  cd {output_dir}")
    print("  ./setup.sh")


if __name__ == "__main__":
    setup_monitoring_stack()