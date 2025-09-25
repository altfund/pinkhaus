# Ominari Deployment Guide

This guide covers deploying the Ominari Blockchain Trading System locally and in production environments.

## Table of Contents

1. [Local Development Deployment](#local-development-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Production Deployment](#production-deployment)
4. [Monitoring Setup](#monitoring-setup)
5. [Troubleshooting](#troubleshooting)

## Local Development Deployment

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+ (optional)
- Docker & Docker Compose
- 8GB RAM minimum
- 50GB disk space

### Step 1: Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd ominari

# Create Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install uv
uv sync

# Copy environment template
cp .env.example .env
```

### Step 2: Configure Environment

Edit `.env` file:

```bash
# Database
DATABASE_URL=postgresql://ominari_user:ominari_2025_secure@localhost:5432/ominari_live
HISTORICAL_DB_PATH=sport_odds.db

# Blockchain RPC (use your own endpoints for production)
OPTIMISM_RPC_URL=https://mainnet.optimism.io
ARBITRUM_RPC_URL=https://arb1.arbitrum.io/rpc

# Redis
REDIS_URL=redis://localhost:6379/0

# Trading Configuration
PAPER_TRADING_BANKROLL=10000
MAX_POSITION_SIZE=0.25
MIN_BET_SIZE=1.0
COMMISSION_RATE=0.002

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Monitoring
MONITORING_ENABLED=true
METRICS_PORT=9090
```

### Step 3: Start PostgreSQL

Option 1: Using system PostgreSQL:
```bash
# Create database and user
sudo -u postgres psql
CREATE DATABASE ominari_live;
CREATE USER ominari_user WITH PASSWORD 'ominari_2025_secure';
GRANT ALL PRIVILEGES ON DATABASE ominari_live TO ominari_user;
\q
```

Option 2: Using Docker:
```bash
docker run -d \
  --name ominari-postgres \
  -e POSTGRES_DB=ominari_live \
  -e POSTGRES_USER=ominari_user \
  -e POSTGRES_PASSWORD=ominari_2025_secure \
  -p 5432:5432 \
  postgres:15-alpine
```

### Step 4: Run Database Migrations

```bash
# Initialize database schema
alembic upgrade head

# Migrate historical data (if available)
python migrate_blockchain_to_postgres.py
```

### Step 5: Start Services

Start each service in a separate terminal:

```bash
# Terminal 1: Web Dashboard
python web_monitor_unified.py

# Terminal 2: Blockchain Sync Daemon
python blockchain_reader.py --daemon

# Terminal 3: Paper Trading Engine
python paper_trading_engine.py

# Terminal 4: API Server (optional)
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Step 6: Verify Installation

1. Open web dashboard: http://localhost:8888
2. Check health endpoint: http://localhost:8000/health
3. View logs for any errors

## Docker Deployment

### Quick Start

```bash
# Start all services
docker-compose up -d

# View status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Service URLs

- Web Dashboard: http://localhost:8888
- Trading API: http://localhost:8000
- Prometheus: http://localhost:9091
- Grafana: http://localhost:3000 (admin/ominari_admin_2025)
- PostgreSQL: localhost:5432
- Redis: localhost:6379

### Docker Compose Services

The `docker-compose.yml` includes:

1. **postgres**: PostgreSQL database
2. **redis**: Redis cache
3. **ominari-trading**: Main trading system
4. **ominari-blockchain**: Blockchain sync daemon
5. **ominari-paper**: Paper trading engine
6. **ominari-web**: Web dashboard
7. **prometheus**: Metrics collection
8. **grafana**: Visualization dashboards

### Customizing Docker Deployment

Create `docker-compose.override.yml` for local overrides:

```yaml
version: '3.8'

services:
  ominari-trading:
    environment:
      - LOG_LEVEL=DEBUG
      - PAPER_TRADING_BANKROLL=50000
    volumes:
      - ./custom_config:/app/config

  ominari-web:
    ports:
      - "8889:9090"  # Different port
```

## Production Deployment

### Pre-deployment Checklist

- [ ] SSL certificates configured
- [ ] Firewall rules set
- [ ] Database backups configured
- [ ] Monitoring alerts set up
- [ ] Log rotation configured
- [ ] Resource limits defined
- [ ] Security scan completed

### Step 1: Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y docker.io docker-compose nginx certbot python3-certbot-nginx

# Add user to docker group
sudo usermod -aG docker $USER

# Setup firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### Step 2: Configure Production Environment

Create `.env.production`:

```bash
# Production settings
ENVIRONMENT=production
LOG_LEVEL=INFO

# Use managed database
DATABASE_URL=postgresql://user:pass@managed-db.amazonaws.com:5432/ominari

# Production RPC endpoints
OPTIMISM_RPC_URL=https://your-node.optimism.io
ARBITRUM_RPC_URL=https://your-node.arbitrum.io

# Security
SECRET_KEY=<generate-strong-secret>
API_KEY=<generate-api-key>

# Resource limits
MAX_WORKERS=4
CONNECTION_POOL_SIZE=20
```

### Step 3: Deploy with Docker

```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d

# Or use deployment script
./deploy.sh production deploy
```

### Step 4: Configure Nginx

Create `/etc/nginx/sites-available/ominari`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8888;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable site and get SSL:
```bash
sudo ln -s /etc/nginx/sites-available/ominari /etc/nginx/sites-enabled/
sudo certbot --nginx -d your-domain.com
sudo systemctl reload nginx
```

### Step 5: Setup Monitoring

Configure alerts in `monitoring/alerts/trading_alerts.yml`:

```yaml
groups:
  - name: trading_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High error rate detected
          
      - alert: DatabaseDown
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: PostgreSQL database is down
```

### Step 6: Configure Backups

Create backup script `/opt/ominari/backup.sh`:

```bash
#!/bin/bash
BACKUP_DIR="/backups/ominari"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup database
docker-compose exec -T postgres pg_dump -U ominari_user ominari_live | \
  gzip > "$BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz"

# Backup configuration
tar czf "$BACKUP_DIR/config_backup_$TIMESTAMP.tar.gz" .env* *.yml

# Keep only last 30 days
find "$BACKUP_DIR" -type f -mtime +30 -delete

# Upload to S3 (optional)
aws s3 sync "$BACKUP_DIR" s3://your-backup-bucket/ominari/
```

Add to crontab:
```bash
0 2 * * * /opt/ominari/backup.sh
```

## Monitoring Setup

### Prometheus Configuration

Already configured in `monitoring/prometheus.yml`. Key metrics:

- `ominari_markets_total`: Total markets by sport/status
- `ominari_signals_generated`: Signal generation count
- `ominari_paper_trades_total`: Paper trading activity
- `ominari_pnl_total`: Cumulative P&L

### Grafana Dashboards

Import dashboards from `monitoring/grafana/dashboards/`:

1. **Trading Overview**: Overall system health
2. **Signal Performance**: Signal accuracy and returns
3. **Paper Trading**: Detailed P&L analysis
4. **System Metrics**: Resource usage

### Setting Up Alerts

Configure Alertmanager in `monitoring/alertmanager.yml`:

```yaml
global:
  slack_api_url: 'YOUR_SLACK_WEBHOOK'

route:
  receiver: 'slack-notifications'
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h

receivers:
  - name: 'slack-notifications'
    slack_configs:
      - channel: '#ominari-alerts'
        title: 'Ominari Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

## Troubleshooting

### Common Deployment Issues

#### 1. Database Connection Failed

**Symptom**: `psycopg2.OperationalError: could not connect to server`

**Solution**:
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check connection string
docker-compose exec ominari-trading python -c "
from database_v2 import db_manager
print(db_manager.engine.url)
"

# Test connection
docker-compose exec postgres pg_isready -U ominari_user
```

#### 2. Blockchain RPC Errors

**Symptom**: `Error fetching block: HTTPError 429`

**Solution**:
- Use private RPC endpoints
- Configure rate limiting
- Add multiple fallback endpoints in `rpc_config.py`

#### 3. Redis Connection Refused

**Symptom**: `Redis not available: Connection refused`

**Solution**:
```bash
# Start Redis if not running
docker-compose up -d redis

# Check Redis is accessible
docker-compose exec redis redis-cli ping
```

#### 4. Port Already in Use

**Symptom**: `bind: address already in use`

**Solution**:
```bash
# Find process using port
sudo lsof -i :8888

# Kill process or change port in docker-compose.yml
```

#### 5. Insufficient Resources

**Symptom**: Container keeps restarting

**Solution**:
```bash
# Check resource usage
docker stats

# Increase limits in docker-compose.yml:
services:
  ominari-trading:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

### Performance Tuning

#### Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX idx_market_sport ON market(sport);
CREATE INDEX idx_market_starts_at ON market(starts_at);
CREATE INDEX idx_odd_timestamp ON odd(timestamp);

-- Vacuum and analyze
VACUUM ANALYZE;
```

#### Redis Optimization

```bash
# Configure Redis for production
redis-cli CONFIG SET maxmemory 2gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET save ""  # Disable persistence for cache
```

### Health Checks

Run system health check:
```bash
python scripts/health_check.py
```

Expected output:
```
✅ Database: Connected (45ms)
✅ Redis: Connected (2ms)
✅ Blockchain RPC: Optimism OK, Arbitrum OK
✅ Web Dashboard: Running on port 8888
✅ Paper Trading: Active (3 open positions)
✅ Disk Space: 45GB free
✅ Memory: 4.2GB/8GB used
```

## Maintenance

### Regular Tasks

Daily:
- Check error logs: `docker-compose logs --since 24h | grep ERROR`
- Monitor disk space: `df -h`
- Check database size: `docker-compose exec postgres psql -U ominari_user -c "\l+"`

Weekly:
- Run database vacuum: `docker-compose exec postgres vacuumdb -U ominari_user -z ominari_live`
- Review performance metrics in Grafana
- Update dependencies: `docker-compose pull`

Monthly:
- Test backup restoration
- Review and rotate logs
- Security updates: `sudo apt update && sudo apt upgrade`

### Scaling Considerations

For high-volume deployments:

1. **Database**: Use managed PostgreSQL with read replicas
2. **Redis**: Use Redis Cluster or ElastiCache
3. **Application**: Run multiple instances behind load balancer
4. **Blockchain**: Use dedicated nodes or node providers
5. **Monitoring**: Use managed Prometheus/Grafana or DataDog

## Support

For deployment issues:
1. Check logs: `docker-compose logs -f <service>`
2. Review documentation in `/docs`
3. Check GitHub issues
4. Contact support with deployment logs