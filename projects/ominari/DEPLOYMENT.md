# Ominari Blockchain Trading System - Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available
- 50GB disk space (for PostgreSQL + Redis + logs)

### Production Deployment
```bash
# 1. Clone and prepare
git clone <repository>
cd ominari

# 2. Build and start all services
docker-compose up -d

# 3. Check service status
docker-compose ps
docker-compose logs -f ominari-trading
```

### Development Deployment
```bash
# Start with development overrides
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Include development tools (PgAdmin, Redis Commander)
docker-compose --profile dev-tools up -d
```

## 📦 Container Services

### Core Services

1. **ominari-trading** (Port 8000)
   - Main trading engine and API
   - Handles signal processing and trade execution
   - Web interface for monitoring

2. **ominari-blockchain** 
   - Blockchain sync daemon
   - Continuously discovers new markets on Optimism/Arbitrum
   - Updates market data and odds

3. **ominari-paper**
   - Paper trading engine
   - Tests strategies with virtual money
   - Performance tracking and reporting

4. **ominari-web** (Port 9090)
   - Web monitoring dashboard
   - Real-time system status
   - Trading performance metrics

### Infrastructure Services

5. **postgres** (Port 5432)
   - Primary database for live data
   - Optimized for high-frequency trading data
   - Automatic backups and migrations

6. **redis** (Port 6379)
   - High-performance caching layer
   - 10-1000x query speedup
   - Session and signal caching

7. **prometheus** (Port 9091)
   - Metrics collection and monitoring
   - Custom trading metrics
   - Alert generation

8. **grafana** (Port 3000)
   - Monitoring dashboards
   - Performance visualization
   - Trading analytics

## 🔧 Configuration

### Environment Variables

#### Trading System
```bash
DATABASE_URL=postgresql://ominari_user:ominari_2025_secure@postgres:5432/ominari_live
REDIS_URL=redis://redis:6379/0
HISTORICAL_DB_PATH=/app/sport_odds.db
PAPER_TRADING_BANKROLL=10000
```

#### Blockchain Integration  
```bash
OPTIMISM_RPC_URL=https://mainnet.optimism.io
ARBITRUM_RPC_URL=https://arb1.arbitrum.io/rpc
BLOCKCHAIN_SYNC_INTERVAL=60
```

### Service Configuration

#### Different Service Modes
```bash
# Trading system
SERVICE=trading

# Blockchain sync daemon
SERVICE=blockchain-sync

# Paper trading
SERVICE=paper-trading

# Web monitor
SERVICE=web-monitor

# API server only
SERVICE=api

# Backtesting
SERVICE=backtest
```

## 🛠️ Management Commands

### Service Management
```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d ominari-trading

# View logs
docker-compose logs -f ominari-trading
docker-compose logs --tail 100 ominari-blockchain

# Restart service
docker-compose restart ominari-trading

# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ DATA LOSS)
docker-compose down -v
```

### Database Management
```bash
# Run database migrations
docker-compose exec ominari-trading python -c "
from alembic.config import Config
from alembic import command
alembic_cfg = Config('alembic.ini')
command.upgrade(alembic_cfg, 'head')
"

# PostgreSQL shell access
docker-compose exec postgres psql -U ominari_user -d ominari_live

# Create database backup
docker-compose exec postgres pg_dump -U ominari_user ominari_live > backup.sql

# Restore database
docker-compose exec -T postgres psql -U ominari_user -d ominari_live < backup.sql
```

### Monitoring Access

#### Grafana Dashboard
- URL: http://localhost:3000
- Username: admin
- Password: ominari_admin_2025

#### Prometheus Metrics
- URL: http://localhost:9091
- Trading metrics: http://localhost:8000/metrics
- Direct Prometheus query interface

#### Development Tools (with --profile dev-tools)
- PgAdmin: http://localhost:5050
- Redis Commander: http://localhost:8081

## 🔍 Health Checks

### Service Health Monitoring
```bash
# Check all service status
docker-compose ps

# Individual service health
docker-compose exec ominari-trading curl http://localhost:8000/health

# Database connectivity
docker-compose exec postgres pg_isready -U ominari_user

# Redis connectivity  
docker-compose exec redis redis-cli ping
```

### Log Monitoring
```bash
# Follow trading system logs
docker-compose logs -f ominari-trading

# Check blockchain sync progress
docker-compose logs -f ominari-blockchain | grep "Processed"

# Monitor all services
docker-compose logs -f
```

## 🚨 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs for errors
docker-compose logs ominari-trading

# Check resource usage
docker stats

# Verify dependencies are healthy
docker-compose exec postgres pg_isready
docker-compose exec redis redis-cli ping
```

#### Database Connection Issues
```bash
# Check PostgreSQL is accepting connections
docker-compose exec postgres psql -U ominari_user -d ominari_live -c "SELECT version();"

# Verify database exists
docker-compose exec postgres psql -U ominari_user -l
```

#### Blockchain Sync Issues
```bash
# Check RPC endpoint connectivity
docker-compose exec ominari-blockchain curl -X POST https://mainnet.optimism.io \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'

# Monitor sync progress
docker-compose logs -f ominari-blockchain
```

## 📊 Performance Optimization

### Resource Allocation
```yaml
# Add to docker-compose.yml services
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 1G
      cpus: '0.5'
```

### Database Tuning
```bash
# Check PostgreSQL performance
docker-compose exec postgres psql -U ominari_user -d ominari_live -c "
SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del 
FROM pg_stat_user_tables ORDER BY n_tup_ins DESC;
"
```

### Cache Optimization
```bash
# Check Redis memory usage
docker-compose exec redis redis-cli INFO memory

# Monitor cache hit rates
docker-compose exec redis redis-cli INFO stats | grep keyspace
```

## 🔐 Security

### Network Security
- All services run in isolated Docker network
- Only necessary ports exposed to host
- Internal service communication encrypted

### Database Security
- Strong passwords generated for all database users
- Connection encryption enabled
- Regular automated backups

### API Security
- Rate limiting enabled on all endpoints
- Authentication required for trading operations
- Input validation and sanitization

## 🔄 Updates and Maintenance

### Service Updates
```bash
# Pull latest images
docker-compose pull

# Rebuild custom images
docker-compose build --no-cache

# Rolling update (zero downtime)
docker-compose up -d --no-deps ominari-trading
```

### Database Maintenance
```bash
# Analyze query performance
docker-compose exec postgres psql -U ominari_user -d ominari_live -c "
SELECT query, mean_time, calls FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;
"

# Vacuum and analyze
docker-compose exec postgres psql -U ominari_user -d ominari_live -c "VACUUM ANALYZE;"
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale specific service
docker-compose up -d --scale ominari-blockchain=2

# Load balancing with nginx
# Add nginx service to docker-compose.yml
```

### Vertical Scaling
```bash
# Increase container resources
# Modify deploy.resources in docker-compose.yml
```

## 🎯 Production Checklist

- [ ] Set strong passwords in environment variables
- [ ] Configure proper logging levels
- [ ] Set up external monitoring (DataDog, etc.)
- [ ] Configure automated backups
- [ ] Test disaster recovery procedures
- [ ] Set up SSL certificates for external access
- [ ] Configure firewall rules
- [ ] Set up log rotation
- [ ] Test scaling procedures
- [ ] Document incident response procedures

---

🚀 **Ready for Production**: This deployment configuration is production-ready with monitoring, caching, and high availability features.