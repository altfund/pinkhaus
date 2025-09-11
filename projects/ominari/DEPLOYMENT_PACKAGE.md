# Ominari Trading System - New Machine Deployment Guide

## 🚀 Quick Start (Hybrid Architecture)

### Prerequisites
- Docker & Docker Compose
- Python 3.13+
- Git
- `uv` (Python package manager)

### 1. Clone and Setup
```bash
git clone <repository>
cd ominari
```

### 2. Install Dependencies
```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### 3. Environment Configuration
```bash
# Copy environment template
cp .env.template .env

# Edit configuration (required settings highlighted below)
nano .env
```

**Required Environment Variables:**
```bash
# Hybrid Database Configuration (NEW)
PG_HOST=localhost
PG_PORT=5435
PG_USER=ominari_user  
PG_PASSWORD=ominari_2025_secure
PG_DB=ominari_production

# Legacy SQLite (for historical data)
DATABASE_URL=sqlite:///sport_odds.db

# Trading Mode (for safety)
PAPER_TRADING_MODE=true
RISK_PRESET=conservative

# Basic API keys (optional for testing)
OVERTIME_API_KEY=your-key-here
ODDS_API_KEY=your-key-here
```

### 4. Start Infrastructure
```bash
# Start PostgreSQL and Redis
docker run -d --name ominari-postgres-hybrid -p 5435:5432 \
  -e POSTGRES_USER=ominari_user \
  -e POSTGRES_PASSWORD=ominari_2025_secure \
  -e POSTGRES_DB=ominari_production \
  -v ominari_postgres_data:/var/lib/postgresql/data \
  postgres:15-alpine

# Verify PostgreSQL is running  
docker ps | grep postgres-hybrid
```

### 5. Initialize Databases
```bash
# Setup PostgreSQL schema for blockchain data
uv run python setup_postgresql_hybrid.py

# Initialize SQLite (if needed)
uv run python create_migration_prerequisites_fast.py

# Test hybrid access
uv run python hybrid_database_access.py
```

### 6. Start Trading System
```bash
# Option 1: Full system with monitoring
uv run python ominari_unified.py

# Option 2: Basic paper trading
uv run python paper_trading_engine.py

# Option 3: Blockchain data collection only
uv run python blockchain_sync_daemon.py
```

## 🏗️ Architecture Overview

### Data Flow (Hybrid)
```
Historical Data (>6 months) → SQLite → Applications
New Data (<6 months) → Blockchain → PostgreSQL → Applications
                     ↓
              Unified Access Layer
                (hybrid_database_access.py)
```

### Key Components
- **SQLite:** Historical odds/markets (216GB)
- **PostgreSQL:** Real-time blockchain data
- **Hybrid Access:** Automatic routing layer
- **Docker Services:** PostgreSQL, monitoring stack

## 📁 Critical Files for New Machines

### Configuration
- `.env` - Environment variables
- `docker-compose.postgres-standalone.yml` - PostgreSQL setup
- `pyproject.toml` - Dependencies

### Hybrid Database System  
- `hybrid_database_access.py` - Unified data access
- `setup_postgresql_hybrid.py` - PostgreSQL schema
- `sqlite_optimization_strategy.py` - SQLite optimization

### Core Trading
- `ominari_unified.py` - Main trading system
- `signals.py` - Signal providers
- `kelly_multimarket.py` - Position sizing
- `paper_trading_engine.py` - Paper trading

### Blockchain Integration
- `blockchain_sync_daemon.py` - Data collection
- `multi_chain_data_manager.py` - Chain management
- `arbitrage_engine.py` - Cross-chain opportunities

## 🔧 System Requirements

### Minimum Specs
- **RAM:** 8GB (16GB recommended)
- **Storage:** 300GB+ (for full historical data)
- **CPU:** 4 cores (8 recommended)
- **Network:** Stable internet for blockchain RPC

### Recommended Specs
- **RAM:** 32GB
- **Storage:** 1TB SSD 
- **CPU:** 8+ cores
- **Network:** Low-latency connection

## 🗄️ Database Options

### Option 1: Full Historical Data
```bash
# Copy sport_odds.db from existing machine (216GB)
rsync -av --progress user@old-machine:/path/sport_odds.db ./
```

### Option 2: Fresh Start (Blockchain Only)
```bash
# Start with empty PostgreSQL, collect new data
# Historical analysis limited to recent data
```

### Option 3: Hybrid Migration
```bash
# Copy recent data subset, full blockchain sync
# Best for new deployments
```

## ⚙️ Deployment Modes

### Development
```bash
export DEBUG=true
export TESTING=true
export PAPER_TRADING_MODE=true
uv run python ominari_unified.py
```

### Staging  
```bash
export RISK_PRESET=conservative
export PAPER_TRADING_MODE=true
export TELEMETRY_ENABLED=true
uv run python ominari_unified.py
```

### Production
```bash
export RISK_PRESET=moderate
export PAPER_TRADING_MODE=false  # Only when ready!
export TELEMETRY_ENABLED=true
export TRADING_PRIVATE_KEY=your-key  # Secure storage!
uv run python ominari_unified.py
```

## 🔐 Security Checklist

### Environment Security
- [ ] `.env` file not in version control
- [ ] Private keys stored securely
- [ ] Database passwords complex
- [ ] API keys have minimal permissions
- [ ] Paper trading enabled initially

### Database Security  
- [ ] PostgreSQL password changed
- [ ] Database backups configured
- [ ] Access logging enabled
- [ ] Network access restricted

### Application Security
- [ ] HTTPS for web interfaces
- [ ] API authentication enabled
- [ ] Monitoring alerts configured
- [ ] Emergency shutdown procedures

## 📊 Health Checks

### Database Connectivity
```bash
# Test SQLite
uv run python -c "
import sqlite3
conn = sqlite3.connect('sport_odds.db')
print('SQLite rows:', conn.execute('SELECT COUNT(*) FROM market LIMIT 1').fetchone()[0])
conn.close()
"

# Test PostgreSQL
uv run python verify_postgresql_setup.py

# Test hybrid access
uv run python hybrid_database_access.py
```

### System Health
```bash
# Check containers
docker ps

# Check disk space
df -h

# Check memory
free -h

# Check processes
ps aux | grep -E "(ominari|python)"
```

## 🚨 Troubleshooting

### Common Issues

**PostgreSQL Connection Failed**
```bash
# Check container status
docker logs ominari-postgres-hybrid

# Restart container
docker restart ominari-postgres-hybrid
```

**SQLite Lock Errors**
```bash
# Check for hung processes
lsof sport_odds.db

# Restart with WAL mode
uv run python sqlite_optimization_strategy.py
```

**Dependencies Issues**
```bash
# Reinstall dependencies
uv sync --reinstall

# Check Python version
python --version  # Should be 3.13+
```

**Memory Issues**
```bash
# Reduce cache sizes in hybrid_database_access.py
# Monitor with: htop or top
```

## 📞 Support

### Log Files
- `logs/ominari.log` - Main application
- `hybrid_migration.log` - Database migration
- `docker logs ominari-postgres-hybrid` - PostgreSQL

### Key Commands
```bash
# System status
uv run python system_health_check.py

# Database stats
uv run python hybrid_database_access.py

# Paper trading test
uv run python paper_trading_engine.py --test

# Signal registry test
uv run python signal_registry.py --test
```

---

**Status:** ✅ Ready for deployment  
**Architecture:** Hybrid SQLite/PostgreSQL  
**Mode:** Paper trading enabled by default  
**Security:** Environment-based configuration