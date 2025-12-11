# Dependencies and Setup Guide

This document lists all dependencies and configuration needed for deploying to a new machine.

## What's Already in Version Control ✅

These items are committed and will be pulled automatically:

- All Python code
- `pyproject.toml` - Python dependency definitions
- `alembic/` - Database migrations
- Documentation (deployment guides, architecture docs)
- `.flox/env/manifest.toml` - Flox environment definition
- `.gitignore` - Prevents committing sensitive files

## What You Need to Install/Configure on New Machine ⚙️

### 1. System Dependencies

**Install uv package manager:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install flox (optional, for PostgreSQL):**
```bash
# If using flox for PostgreSQL management
curl -1sLf 'https://downloads.flox.dev/by-name/stable/x86_64-linux/flox' | bash
```

**OR install PostgreSQL directly:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install postgresql-15 postgresql-contrib

# macOS
brew install postgresql@15
```

### 2. Environment Variables (NOT in version control)

Create a `.env.production` file with these settings:

```bash
# PostgreSQL Database Connection
export PG_HOST=localhost  # Or IP of database machine
export PG_PORT=5999
export PG_USER=ominari_user
export PG_PASSWORD=ominari_2025_secure
export PG_DB=ominari_production

# Environment
export OMINARI_ENV=production

# Trading Mode
export ENABLE_MAINNET_TRADING=false  # Set true only for real money trading
export USE_MOCK_BLOCKCHAIN=true  # Set false when using real blockchain

# RPC URLs (optional - only needed for real blockchain trading)
# Get free API keys from Alchemy or Infura
# OPTIMISM_RPC_URL=https://opt-mainnet.g.alchemy.com/v2/YOUR_API_KEY
# ARBITRUM_RPC_URL=https://arb-mainnet.g.alchemy.com/v2/YOUR_API_KEY
```

**Important:** The `.env.*` files are in `.gitignore` - you must create them manually on each machine.

### 3. PostgreSQL Database Setup

**If using the same database (recommended for 2+ machine setup):**

On the database machine, configure PostgreSQL for network access:

```bash
# Edit postgresql.conf
listen_addresses = '*'

# Edit pg_hba.conf to allow other machines
host    all    ominari_user    192.168.1.0/24    md5

# Restart PostgreSQL
sudo systemctl restart postgresql
# OR if using flox:
flox activate  # PostgreSQL starts automatically
```

**If setting up a new database:**

```bash
# On database machine
psql -U postgres

CREATE USER ominari_user WITH PASSWORD 'ominari_2025_secure';
CREATE DATABASE ominari_production OWNER ominari_user;
\q

# Run migrations
cd projects/ominari
uv run alembic upgrade head
```

### 4. Session Files (NOT in version control)

The system uses JSON session files to track portfolio state. These are NOT in version control because they contain trading state.

**For production trading:**
- Session files are created automatically on first run
- File: `paper_trading_sessions_production.json`

**To start fresh:**
```bash
# The system will create a new session file automatically
# with initial bankroll when services start
```

**To share existing portfolio state between machines:**
```bash
# Copy from existing machine to new machine
scp paper_trading_sessions_production.json user@new-machine:/path/to/ominari/projects/ominari/
```

## Deployment Checklist for New Machine

### Single Machine Setup (All Services)

```bash
# 1. Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repository
git clone git@github.com:ScheierVentures/ominari.git
cd ominari/projects/ominari
git checkout feature/ominari-updates

# 3. Install Python dependencies
uv sync

# 4. Setup PostgreSQL (using flox)
flox activate
# PostgreSQL starts automatically on port 5999

# 5. Create environment variables
cp .env.example .env.production
# Edit .env.production with your settings

# 6. Run database migrations
uv run alembic upgrade head

# 7. Start services
uv run python market_data_heartbeat.py > logs/market_data_heartbeat.log 2>&1 &
uv run python carver_heartbeat_with_backtest.py > logs/carver_heartbeat.log 2>&1 &
uv run python execute_trading_solution.py > logs/execute_trading.log 2>&1 &
```

### Multi-Machine Setup (Trading Machine Only)

```bash
# 1-3. Same as above (install uv, clone, uv sync)

# 4. Skip PostgreSQL setup (using remote database)

# 5. Create environment variables pointing to database machine
cat > .env.production << EOF
export PG_HOST=192.168.1.100  # Database machine IP
export PG_PORT=5999
export PG_USER=ominari_user
export PG_PASSWORD=ominari_2025_secure
export PG_DB=ominari_production
export OMINARI_ENV=production
EOF

# 6. Test database connectivity
uv run python -c "from database_v2 import db_manager; print(db_manager.test_connection())"

# 7. Start trading services only (no market data heartbeat if running on DB machine)
uv run python carver_heartbeat_with_backtest.py > logs/carver_heartbeat.log 2>&1 &
uv run python execute_trading_solution.py > logs/execute_trading.log 2>&1 &
```

## Critical Files NOT in Version Control

### Configuration Files (.gitignored)
- `.env.production` - Production environment variables
- `.env.development` - Development environment variables
- `.env.local` - Local testnet configuration
- `.env.testnet` - Testnet configuration

### Data Files (.gitignored)
- `paper_trading_sessions_*.json` - Portfolio state and positions
- `logs/*.log` - Service logs
- `.flox/postgres/data/` - PostgreSQL database files
- `*.db` - SQLite database files (if used)

### Session State Files
- `paper_trading_sessions_production.json` - Production portfolio
- `paper_trading_sessions_dev.json` - Development portfolio
- `paper_trading_sessions_staging.json` - Staging portfolio

## Secrets Management

**Never commit these to version control:**

1. **Database passwords** - In `.env.*` files
2. **API keys** - Alchemy, Infura, Overtime API
3. **Private keys** - Trading wallet keys (for real blockchain trading)
4. **Session data** - Contains trading positions and P&L

**To share secrets securely between machines:**
- Use encrypted password manager
- Use SSH to transfer files directly
- Use environment variables only (never hardcode)

## Optional: API Keys

These are only needed for specific features:

**Overtime API Key** (for advanced market data):
- Contact Overtime team
- Add to `.env.production`: `OVERTIME_API_KEY=your_key`

**RPC Provider Keys** (for real blockchain trading):
- Alchemy: https://www.alchemy.com/
- Infura: https://infura.io/
- Add to `.env.production`:
  ```
  OPTIMISM_RPC_URL=https://opt-mainnet.g.alchemy.com/v2/YOUR_KEY
  ARBITRUM_RPC_URL=https://arb-mainnet.g.alchemy.com/v2/YOUR_KEY
  ```

## Verification After Setup

Run these commands to verify everything is working:

```bash
# 1. Check Python dependencies
uv run python -c "import pandas, sqlalchemy, psycopg2; print('✅ Dependencies OK')"

# 2. Check database connection
uv run python -c "from database_v2 import db_manager; print('DB:', db_manager.test_connection())"

# 3. Check services are running
ps aux | grep -E "(market_data_heartbeat|carver_heartbeat|execute_trading)" | grep -v grep

# 4. Check recent activity
tail -20 logs/execute_trading.log
```

## Quick Reference

**What's in git:** Code, docs, dependency definitions
**What's NOT in git:** Secrets, database files, session state, logs
**What you need to create:** `.env.production`, PostgreSQL database
**What auto-creates:** Session files, logs directory

## Troubleshooting

**"ModuleNotFoundError: psycopg2"**
- Always use `uv run` to execute scripts
- Never use `python` or `.venv/bin/python` directly

**"No session file found"**
- Set `OMINARI_ENV` environment variable
- Session file will be created automatically on first run

**"Database connection failed"**
- Check PostgreSQL is running: `ps aux | grep postgres`
- Verify credentials in `.env.production`
- Test network access: `psql -U ominari_user -h <DB_IP> -p 5999 -d ominari_production`

**"Permission denied" on logs**
- Create logs directory: `mkdir -p logs`
- Check write permissions: `chmod 755 logs`
