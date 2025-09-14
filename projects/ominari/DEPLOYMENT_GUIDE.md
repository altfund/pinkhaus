# Ominari Unified Dashboard - Deployment Guide

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- **Python 3.13+** installed
- **Git** for cloning repository
- **PostgreSQL** running on port 5435 (or use Docker option below)

### 1. Clone and Setup
```bash
# Clone repository
git clone git@github.com:ScheierVentures/ominari.git
cd ominari/projects/ominari

# Install dependencies with uv
uv sync

# Copy environment configuration
cp .env.production .env
# Edit .env to match your setup (database, API keys, etc.)
```

### 2. Database Setup (Choose One)

#### Option A: Use Existing PostgreSQL
```bash
# If PostgreSQL already running on port 5435
python check_system_status.py

# If connected, you're ready to go!
```

#### Option B: Docker PostgreSQL (Recommended)
```bash
# Start optimized PostgreSQL container
docker-compose -f docker-compose.postgres-standalone.yml up -d

# Wait for health check
docker-compose -f docker-compose.postgres-standalone.yml ps

# Run database migrations
uv run alembic upgrade head
```

### 3. Launch Dashboard
```bash
# Start the enhanced unified dashboard
uv run python web_monitor.py

# Dashboard will be available at:
# http://localhost:8888/unified
```

### 4. Verify Deployment
```bash
# Check system status
python check_system_status.py

# Test API endpoints
curl http://localhost:8888/api/dashboard/unified | jq '.markets[:2]'

# Expected: Market data with Home/Draw/Away odds displaying
```

---

## 🏭 Production Deployment

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Memory**: 4GB+ RAM (8GB recommended)
- **Storage**: 50GB+ SSD for optimal PostgreSQL performance
- **Network**: Stable internet for blockchain data feeds
- **Docker**: 20.10+ and Docker Compose 2.0+

### Production Infrastructure Setup

#### 1. Database Infrastructure
```bash
# Production PostgreSQL with optimizations
docker-compose -f docker-compose.postgres-standalone.yml up -d

# Verify database performance
docker exec -it ominari-postgres-hybrid psql -U ominari_user -d ominari_production -c "
SELECT
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes
FROM pg_stat_user_tables
ORDER BY n_tup_ins DESC LIMIT 5;"
```

#### 2. Application Container
```bash
# Build production image
docker build -t ominari-dashboard:latest .

# Run with production environment
docker run -d \
  --name ominari-dashboard \
  --network ominari-network \
  -p 8888:8888 \
  -e PG_HOST=ominari-postgres-hybrid \
  -e PG_PORT=5432 \
  -e LOG_LEVEL=INFO \
  --restart unless-stopped \
  ominari-dashboard:latest
```

#### 3. Reverse Proxy (Nginx)
```nginx
# /etc/nginx/sites-available/ominari-dashboard
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8888;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;

        # WebSocket support for real-time updates
        proxy_read_timeout 86400;
    }
}
```

---

## 🔧 Detailed Configuration

### Environment Variables

Create a comprehensive `.env` file:
```bash
# === CORE DATABASE ===
PG_HOST=localhost
PG_PORT=5435
PG_USER=ominari_user
PG_PASSWORD=ominari_2025_secure
PG_DB=ominari_production

# === DASHBOARD CONFIGURATION ===
API_PORT=8888
API_HOST=0.0.0.0
ENABLE_CORS=true
DEBUG=false

# === TRADING SETTINGS ===
PAPER_TRADING_MODE=true  # CRITICAL: Start in paper mode
INITIAL_BANKROLL=10000
RISK_PRESET=conservative
MAX_POSITION_SIZE_PCT=0.02
MIN_EDGE_REQUIRED=0.015

# === BLOCKCHAIN RPC (Optional - uses free endpoints by default) ===
OPTIMISM_RPC_URL=https://mainnet.optimism.io
ARBITRUM_RPC_URL=https://arb1.arbitrum.io/rpc

# === API KEYS (Optional - improves data quality) ===
OVERTIME_API_KEY=your_overtime_key
ODDS_API_KEY=your_odds_api_key
ALCHEMY_API_KEY=your_alchemy_key

# === MONITORING ===
LOG_LEVEL=INFO
TELEMETRY_ENABLED=true
```

### Database Migration and Setup
```bash
# Initialize database schema
uv run alembic upgrade head

# Populate with sample data (optional)
uv run python -c "
from database_v2 import db_manager
from setup_postgresql_hybrid import populate_sample_data
populate_sample_data()
print('✅ Sample data populated')
"

# Verify data integrity
uv run python check_system_status.py
```

### Dashboard Options

The system provides two dashboard implementations:

#### 1. **Enhanced Dashboard** (Primary - Recommended)
- **File**: `web_monitor.py`
- **URL**: `http://localhost:8888/unified`
- **Features**:
  - Full PostgreSQL integration
  - Paper trading engine with Kelly optimization
  - Enhanced blockchain signal providers (1.5x/1.2x weights)
  - Real-time WebSocket updates
  - Comprehensive risk management

```bash
# Launch enhanced dashboard (default)
uv run python web_monitor.py
```

#### 2. **Unified Dashboard** (Alternative)
- **File**: `web_monitor_unified.py`
- **URL**: `http://localhost:8889/` (modify port to avoid conflict)
- **Features**:
  - Clean implementation from scheier
  - Dual-chain blockchain support
  - Simple paper trading
  - WebSocket real-time updates

```bash
# Modify port in web_monitor_unified.py first
sed -i 's/port=8888/port=8889/g' web_monitor_unified.py

# Launch alternative dashboard
uv run python web_monitor_unified.py
```

---

## 📊 Monitoring & Operations

### Health Checks
```bash
# Comprehensive system status
python check_system_status.py

# Expected output:
# ✅ PostgreSQL: CONNECTED
# ✅ SQLite: AVAILABLE
# Total Markets: 47,049
# Web Monitor: RUNNING
# Blockchain Coverage: Optimism (3), Arbitrum (0)
```

### Performance Monitoring
```bash
# Database performance
uv run python -c "
from database_v2 import db_manager
import time
start = time.time()
with db_manager.get_db_session() as db:
    result = db.execute('SELECT COUNT(*) FROM market').scalar()
    print(f'Query time: {time.time() - start:.3f}s')
    print(f'Total markets: {result}')
"

# Expected: Query time <0.1s for optimal performance
```

### API Endpoint Testing
```bash
# Test all critical endpoints
echo "Testing dashboard endpoints..."

curl -s http://localhost:8888/api/dashboard/unified | jq '.markets | length'
curl -s http://localhost:8888/api/trading/positions | jq '.positions | length'
curl -s http://localhost:8888/api/dashboard-data | jq '.performance.total_value'

# Expected responses with market data, positions, and portfolio values
```

### Log Management
```bash
# View application logs
tail -f logs/ominari.log

# Monitor for errors
grep -i "error\|exception\|failed" logs/ominari.log | tail -20

# WebSocket connection monitoring
grep -i "websocket\|socketio" logs/ominari.log | tail -10
```

---

## 🔒 Security Considerations

### Production Security Checklist
- [ ] **Environment Variables**: Secure storage of database passwords and API keys
- [ ] **Network Security**: Firewall rules limiting access to port 8888
- [ ] **Database Security**: PostgreSQL authentication and SSL connections
- [ ] **HTTPS Setup**: SSL certificates for production domains
- [ ] **Paper Trading**: Start with `PAPER_TRADING_MODE=true`
- [ ] **API Keys**: Secure storage and rotation of blockchain RPC keys
- [ ] **Backup Strategy**: Regular database backups and disaster recovery

### Secure Environment Setup
```bash
# Create secure environment file
sudo mkdir -p /opt/ominari/config
sudo touch /opt/ominari/config/.env.secure
sudo chmod 600 /opt/ominari/config/.env.secure

# Add secrets securely
echo "PG_PASSWORD=$(openssl rand -base64 32)" | sudo tee -a /opt/ominari/config/.env.secure
echo "SECRET_KEY=$(openssl rand -base64 32)" | sudo tee -a /opt/ominari/config/.env.secure
```

---

## 🔄 Data Migration & Updates

### Blockchain Data Activation
```bash
# Sync recent blockchain markets (increases data 17x)
uv run python blockchain_hybrid_sync.py

# Monitor sync progress
tail -f logs/blockchain_sync.log

# Verify expanded coverage
python check_system_status.py
# Expected: Optimism markets increased significantly
```

### Database Updates
```bash
# Update database schema
uv run alembic revision --autogenerate -m "Your update description"
uv run alembic upgrade head

# Backup before major updates
pg_dump -h localhost -p 5435 -U ominari_user -d ominari_production > backup_$(date +%Y%m%d).sql
```

---

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. **Dashboard Not Loading**
```bash
# Check if process is running
ps aux | grep web_monitor

# Check port availability
lsof -i :8888

# Restart dashboard
pkill -f web_monitor
uv run python web_monitor.py
```

#### 2. **Database Connection Issues**
```bash
# Test PostgreSQL connection
python -c "
from database_v2 import db_manager
try:
    with db_manager.get_db_session() as db:
        result = db.execute('SELECT 1').scalar()
        print('✅ Database connection successful')
except Exception as e:
    print(f'❌ Database error: {e}')
"

# Check PostgreSQL service
docker-compose -f docker-compose.postgres-standalone.yml ps
docker-compose -f docker-compose.postgres-standalone.yml logs postgres
```

#### 3. **Missing Market Data**
```bash
# Check data population
python -c "
from database_v2 import db_manager
from models import Market, Odd
with db_manager.get_db_session() as db:
    markets = db.query(Market).count()
    odds = db.query(Odd).count()
    print(f'Markets: {markets}, Odds: {odds}')
    if markets == 0:
        print('Run: uv run python setup_postgresql_hybrid.py')
"
```

#### 4. **WebSocket Connection Problems**
```bash
# Check WebSocket functionality
curl -s http://localhost:8888/socket.io/?transport=polling

# Monitor WebSocket logs
grep -i "websocket" logs/ominari.log | tail -10

# Test from browser console:
# var socket = io(); socket.on('connect', () => console.log('Connected!'));
```

#### 5. **Performance Issues**
```bash
# Database optimization
docker exec -it ominari-postgres-hybrid psql -U ominari_user -d ominari_production -c "
VACUUM ANALYZE;
REINDEX DATABASE ominari_production;
"

# Check memory usage
docker stats ominari-postgres-hybrid
free -h
```

---

## 📈 Advanced Features

### Blockchain Signal Activation
The system includes advanced blockchain signal providers with enhanced weights:

```bash
# Test blockchain signal providers
uv run python -c "
from signals import SIGNAL_PROVIDERS
print(f'Available signals: {[s.name for s in SIGNAL_PROVIDERS]}')
from blockchain_signal_provider import BlockchainEnhancedSignal
signal = BlockchainEnhancedSignal()
print(f'Networks supported: {signal.networks}')
"
```

### Paper Trading Configuration
```python
# Advanced paper trading settings in .env
KELLY_FRACTION=0.25  # Conservative 25% Kelly
MIN_BET=10
MIN_BET_PCT=0.001
BANKROLL=10000
CAP_PER_GAME=0.25
CAP_PER_BET=0.25
BIASES_FAVORITE=-0.01
BIASES_LONGSHOT=0.01
BIASES_DRAW=0.005
```

### Real-Time Market Evaluation
The enhanced dashboard provides real-time Kelly optimization:
- **Portfolio Value**: Live tracking of paper trading performance
- **Market Signals**: Enhanced blockchain signals with metadata enrichment
- **Risk Management**: Conservative Kelly fractions with position sizing
- **Edge Detection**: Automated opportunity identification and ranking

---

## 🎯 Deployment Checklist

### Pre-Deployment
- [ ] Clone repository and navigate to `projects/ominari/`
- [ ] Install uv package manager and Python 3.13+
- [ ] Configure `.env` with database and API settings
- [ ] Test database connection (PostgreSQL on port 5435)

### Development Deployment
- [ ] Run `uv sync` to install dependencies
- [ ] Start database: `docker-compose -f docker-compose.postgres-standalone.yml up -d`
- [ ] Run migrations: `uv run alembic upgrade head`
- [ ] Launch dashboard: `uv run python web_monitor.py`
- [ ] Verify: Access http://localhost:8888/unified

### Production Deployment
- [ ] Build Docker image: `docker build -t ominari-dashboard .`
- [ ] Setup reverse proxy (Nginx) with SSL certificates
- [ ] Configure secure environment variables and secrets
- [ ] Setup monitoring and log aggregation
- [ ] Configure backup and disaster recovery
- [ ] Test full system with health checks

### Post-Deployment Verification
- [ ] System status: `python check_system_status.py`
- [ ] API endpoints responding with market data
- [ ] WebSocket connections working for real-time updates
- [ ] Paper trading engine calculating positions correctly
- [ ] Database queries performing <100ms
- [ ] Monitoring and alerting functional

---

**🚀 Your Ominari Unified Dashboard is ready for deployment!**

**Primary Dashboard**: http://localhost:8888/unified
**System Status**: `python check_system_status.py`
**Documentation**: CODE_REVIEW_CONTEXT.md, INTEGRATION_STATUS_CLAUDE.md

---

*Last Updated: 2025-09-14*
*Deployment Guide Version: 1.0*
*Compatible with: Enhanced PostgreSQL Integration + Scheier Dual-Chain Improvements*