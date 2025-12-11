# Multi-Machine Deployment Summary

## Quick Answers

### How many machines should this run on?

**Minimum:** **1 Machine** ✅ (Currently operational)
**Recommended:** **2 Machines** (Database + Trading)
**Ideal:** **3 Machines** (Database + Data Ingestion + Trading)

### Current Status

Running successfully on **1 machine** with:
- PostgreSQL (32,095 markets)
- Market Data Heartbeat (live odds fetching)
- Carver Heartbeat (signal generation)
- Execute Trading (finding 10+ opportunities per cycle)
- 20 open positions, $9,816.37 portfolio value

## Architecture Breakdown

### 1-Machine Setup (Current - ✅ Working)

**All services on one machine:**
```
┌─────────────────────────────────────┐
│  Single Machine (localhost)         │
├─────────────────────────────────────┤
│  PostgreSQL :5999                   │
│  Market Data Heartbeat              │
│  Carver Heartbeat                   │
│  Execute Trading                    │
│  Position Tracker (manual)          │
└─────────────────────────────────────┘
```

**Requirements:**
- 4+ CPU cores
- 8GB+ RAM (16GB recommended)
- 100GB+ SSD
- Stable internet

**Pros:** Simple, lower cost, working now
**Cons:** Single point of failure

---

### 2-Machine Setup (Recommended)

```
┌──────────────────────┐       ┌──────────────────────┐
│  Machine 1           │       │  Machine 2           │
│  (Database + Data)   │◄─────►│  (Trading Logic)     │
├──────────────────────┤       ├──────────────────────┤
│  PostgreSQL :5999    │       │  Carver Heartbeat    │
│  Market Data         │       │  Execute Trading     │
│  Heartbeat           │       │  Position Tracker    │
└──────────────────────┘       └──────────────────────┘
      2-4 cores                      2-4 cores
      8GB RAM                        8GB RAM
      100GB SSD                      50GB SSD
```

**Benefits:**
- Database isolated from trading
- Better resource allocation
- Can restart trading without affecting data
- Improved fault tolerance

---

### 3-Machine Setup (Ideal)

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Machine 1       │     │  Machine 2       │     │  Machine 3       │
│  (Database)      │◄───►│  (Data)          │◄───►│  (Trading)       │
├──────────────────┤     ├──────────────────┤     ├──────────────────┤
│  PostgreSQL      │     │  Market Data     │     │  Carver          │
│  Primary+Replica │     │  Heartbeat       │     │  Heartbeat       │
│  Backups         │     │  WebSockets      │     │  Execute Trading │
│  Monitoring      │     │  Real-time       │     │  Risk Manager    │
└──────────────────┘     │  Scanner         │     │  Position        │
    4+ cores             └──────────────────┘     │  Tracker         │
    16GB RAM                  2-4 cores            └──────────────────┘
    250GB SSD                 8GB RAM                   4+ cores
                              50GB SSD                  8GB RAM
                                                        50GB SSD
```

**Benefits:**
- Maximum uptime and redundancy
- Database failover capability
- Best performance isolation
- Independent scaling

---

## Version Control Setup ✅

**Repository:** github.com/ScheierVentures/ominari
**Branch:** feature/ominari-updates
**Status:** All changes pushed

### Latest Commit:
```
feat: Add position tracker and fix live trading PostgreSQL connection

- Real-time position tracker with notifications
- Fixed psycopg2 module conflict (use uv run)
- Deployment architecture guide (1-3 machines)
- Updated .gitignore
- Complete documentation
```

### Files Added:
- `position_tracker.py` - Real-time position monitoring
- `POSITION_TRACKER_GUIDE.md` - Usage documentation
- `DEPLOYMENT_ARCHITECTURE.md` - Full deployment guide
- `LIVE_TRADING_DIAGNOSIS.md` - Troubleshooting guide
- `MULTI_MACHINE_DEPLOYMENT_SUMMARY.md` - This file

---

## Deployment on Another Machine

### Step 1: Clone Repository

```bash
git clone git@github.com:ScheierVentures/ominari.git
cd ominari
git checkout feature/ominari-updates
cd projects/ominari
```

### Step 2: Install Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### Step 3: Configure Environment

```bash
# Set database connection (update IP to point to DB machine)
export PG_HOST=192.168.1.100  # Use 'localhost' for single machine
export PG_PORT=5999
export PG_USER=ominari_user
export PG_PASSWORD=ominari_2025_secure
export PG_DB=ominari_production

# Set environment
export OMINARI_ENV=production  # or dev/staging
```

### Step 4: Start Services

**For Single Machine (all services):**
```bash
# Start PostgreSQL (if needed)
# Already running via flox on port 5999

# Start all services
uv run python market_data_heartbeat.py > logs/market_data_heartbeat.log 2>&1 &
uv run python carver_heartbeat_with_backtest.py > logs/carver_heartbeat.log 2>&1 &
uv run python execute_trading_solution.py > logs/execute_trading.log 2>&1 &
```

**For Multi-Machine (trading machine only):**
```bash
# Only start trading services (DB is on another machine)
export PG_HOST=192.168.1.100  # DB machine IP
uv run python carver_heartbeat_with_backtest.py > logs/carver_heartbeat.log 2>&1 &
uv run python execute_trading_solution.py > logs/execute_trading.log 2>&1 &
```

### Step 5: Monitor

```bash
# Watch position tracker
uv run python position_tracker.py

# Or check logs
tail -f logs/execute_trading.log
```

---

## Migration Path

### From 1 → 2 Machines:

1. **Prepare Machine 2:**
   ```bash
   # On Machine 2
   git clone git@github.com:ScheierVentures/ominari.git
   cd ominari/projects/ominari
   uv sync
   ```

2. **Configure Database Access:**
   ```bash
   # On Machine 2
   export PG_HOST=<Machine-1-IP>  # e.g., 192.168.1.100
   export PG_PORT=5999
   # ... other env vars
   ```

3. **Test Connectivity:**
   ```bash
   # On Machine 2
   uv run python -c "from database_v2 import db_manager; print(db_manager.test_connection())"
   ```

4. **Move Services:**
   ```bash
   # On Machine 1: Stop trading services
   pkill -f carver_heartbeat
   pkill -f execute_trading

   # On Machine 2: Start trading services
   uv run python carver_heartbeat_with_backtest.py > logs/carver_heartbeat.log 2>&1 &
   uv run python execute_trading_solution.py > logs/execute_trading.log 2>&1 &
   ```

5. **Monitor for 24 hours** to ensure stability

---

## Network Configuration (Multi-Machine)

### PostgreSQL Access

**On Database Machine**, edit PostgreSQL config:

```bash
# postgresql.conf
listen_addresses = '*'  # Allow network connections

# pg_hba.conf
# Add line to allow trading machine:
host    all    ominari_user    192.168.1.0/24    md5
```

**Restart PostgreSQL:**
```bash
# Restart to apply changes
```

### Firewall Rules

```bash
# On Database Machine
sudo ufw allow from 192.168.1.0/24 to any port 5999

# On Trading Machine (if running web dashboard)
sudo ufw allow 5000
```

---

## Monitoring & Health Checks

### Service Status
```bash
ps aux | grep -E "(carver_heartbeat|market_data|execute_trading)" | grep -v grep
```

### Database Connectivity
```bash
psql -U ominari_user -h <DB-HOST> -p 5999 -d ominari_production -c "SELECT COUNT(*) FROM ominari.markets_normalized;"
```

### Position Tracker
```bash
uv run python position_tracker.py --once
```

### Logs
```bash
tail -f logs/*.log
```

---

## Cost Comparison

### Cloud Deployment Costs (AWS/DigitalOcean)

**1 Machine:**
- Instance: $40-80/month (4 CPU, 8GB RAM)
- Storage: $10/month (100GB SSD)
- **Total: ~$50-90/month**

**2 Machines:**
- Machine 1: $40-60/month (2-4 CPU, 8GB RAM)
- Machine 2: $30-50/month (2-4 CPU, 8GB RAM)
- Storage: $15/month (150GB total)
- **Total: ~$85-125/month**

**3 Machines:**
- Machine 1: $80-120/month (4 CPU, 16GB RAM, primary DB)
- Machine 2: $30-50/month (2-4 CPU, 8GB RAM, data)
- Machine 3: $40-60/month (4 CPU, 8GB RAM, trading)
- Storage: $25/month (300GB total)
- **Total: ~$175-255/month**

---

## Recommendations

**Current Setup (1 Machine):** ✅ Keep as-is
- Working perfectly
- Cost-effective
- Easy to maintain

**When to Upgrade to 2 Machines:**
- Portfolio value >$50,000
- Need improved uptime
- Want to separate concerns
- Planning for scaling

**When to Upgrade to 3 Machines:**
- Portfolio value >$200,000
- Need database redundancy
- High-frequency trading
- Multiple trading strategies
- Require 99.9% uptime

---

## GitHub Workflow

### Pull Latest Changes
```bash
git fetch origin
git pull origin feature/ominari-updates
uv sync  # Update dependencies
```

### Deploy Changes
```bash
# Stop services
pkill -f carver_heartbeat
pkill -f execute_trading
pkill -f market_data_heartbeat

# Pull latest
git pull origin feature/ominari-updates
uv sync

# Restart services
uv run python market_data_heartbeat.py > logs/market_data_heartbeat.log 2>&1 &
uv run python carver_heartbeat_with_backtest.py > logs/carver_heartbeat.log 2>&1 &
uv run python execute_trading_solution.py > logs/execute_trading.log 2>&1 &
```

### Check for PRs
Visit: https://github.com/ScheierVentures/ominari/pulls

---

## Support & Documentation

- **Deployment Guide:** DEPLOYMENT_ARCHITECTURE.md
- **Position Tracker:** POSITION_TRACKER_GUIDE.md
- **Troubleshooting:** LIVE_TRADING_DIAGNOSIS.md
- **Repository:** github.com/ScheierVentures/ominari
- **Branch:** feature/ominari-updates

---

**Last Updated:** 2025-12-10
**Current Status:** 1 machine, all services operational
**Portfolio Value:** $9,816.37 (20 open positions)
**Markets Available:** 32,095 total, 1,016 upcoming
**Recommendation:** Continue with 1 machine, plan 2-machine migration when portfolio >$50k
