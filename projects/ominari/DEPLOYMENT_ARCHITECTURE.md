# Deployment Architecture Guide

## System Components

The Ominari trading system consists of these core components:

1. **PostgreSQL Database** - Market data storage and position tracking
2. **Market Data Heartbeat** - Fetches live odds from Overtime API
3. **Carver Heartbeat** - Generates trading signals using Carver framework
4. **Execute Trading** - Evaluates opportunities and executes trades
5. **Position Tracker** - Monitors open positions (manual/optional)
6. **Web Dashboard** - Visual interface for portfolio monitoring (optional)

## Machine Deployment Options

### Minimum Configuration: 1 Machine

**Hardware Requirements:**
- 4+ CPU cores
- 8GB+ RAM (16GB recommended)
- 100GB+ SSD storage
- Stable internet connection

**Services on Single Machine:**
- PostgreSQL database (port 5999)
- All Python services (market data, carver, execute trading)
- Optional: Web dashboard, position tracker

**Pros:**
- Simple deployment
- Lower operational cost
- Easier debugging
- Currently running successfully on 1 machine

**Cons:**
- Single point of failure
- Resource contention between services
- Limited horizontal scaling

**Current Status:** ✅ **Running successfully on 1 machine**

---

### Recommended Configuration: 2 Machines

**Machine 1: Database & Data Services**
- PostgreSQL database (primary)
- Market Data Heartbeat
- Resources: 2-4 cores, 8GB RAM, 100GB SSD

**Machine 2: Trading Services**
- Carver Heartbeat (signal generation)
- Execute Trading (trade execution)
- Position Tracker
- Resources: 2-4 cores, 8GB RAM, 50GB SSD

**Pros:**
- Separation of concerns
- Database isolation
- Better fault tolerance
- Easier to scale trading logic separately

**Cons:**
- Network latency between machines
- More complex deployment
- Higher operational cost

---

### Ideal Configuration: 3 Machines

**Machine 1: Database Cluster**
- PostgreSQL primary + replica
- Backup and disaster recovery
- Resources: 4+ cores, 16GB RAM, 250GB SSD

**Machine 2: Data Ingestion**
- Market Data Heartbeat
- Real-time market scanner
- WebSocket connections to exchanges
- Resources: 2-4 cores, 8GB RAM, 50GB SSD

**Machine 3: Trading Logic**
- Carver Heartbeat (signal generation)
- Execute Trading (trade execution)
- Risk management
- Position Tracker
- Resources: 4+ cores, 8GB RAM, 50GB SSD

**Pros:**
- Maximum fault tolerance
- Database can fail over to replica
- Can restart trading services without affecting data ingestion
- Better performance isolation
- Easier to scale each component independently

**Cons:**
- Highest operational complexity
- Highest cost
- Requires robust networking
- More deployment overhead

---

## Deployment Comparison

| Aspect | 1 Machine | 2 Machines | 3 Machines |
|--------|-----------|------------|------------|
| **Cost** | $ | $$ | $$$ |
| **Complexity** | Low | Medium | High |
| **Fault Tolerance** | None | Moderate | High |
| **Performance** | Good | Better | Best |
| **Scalability** | Limited | Good | Excellent |
| **Current Status** | ✅ Running | Recommended | Ideal |

---

## Recommendation

**Minimum Viable:** **1 Machine** (currently operational)
- Perfect for development, testing, and initial production
- Currently handling all services successfully
- Can migrate to multi-machine later without code changes

**Production Recommended:** **2 Machines**
- Separate database from trading logic
- Better resource allocation
- Improved fault tolerance
- Manageable complexity

**Enterprise/High-Value:** **3 Machines**
- Full redundancy and failover
- Maximum performance
- Best for high-frequency trading or large capital deployment

---

## Multi-Machine Deployment Setup

### Prerequisites

1. **Network Configuration**
   - Machines must be able to communicate on private network
   - PostgreSQL port 5999 accessible from trading machine
   - SSH access between machines for deployment

2. **Environment Variables**
   Set on each machine:
   ```bash
   # Database machine IP (or localhost if single machine)
   export PG_HOST=192.168.1.100  # Change to actual DB machine IP
   export PG_PORT=5999
   export PG_USER=ominari_user
   export PG_PASSWORD=ominari_2025_secure
   export PG_DB=ominari_production

   # Environment
   export OMINARI_ENV=production  # or dev/staging
   ```

3. **Code Deployment**
   ```bash
   # On each machine
   git clone <repo_url>
   cd ominari
   git checkout feature/ominari-updates
   uv sync
   ```

### Single Machine Deployment (Current)

```bash
# Start PostgreSQL (if using flox)
# Already running on port 5999

# Start services
uv run python market_data_heartbeat.py > logs/market_data_heartbeat.log 2>&1 &
uv run python carver_heartbeat_with_backtest.py > logs/carver_heartbeat.log 2>&1 &
uv run python execute_trading_solution.py > logs/execute_trading.log 2>&1 &

# Optional: Monitor with position tracker
uv run python position_tracker.py
```

### Two Machine Deployment

**Machine 1 (Database + Data):**
```bash
# Ensure PostgreSQL running and accessible
# Configure postgresql.conf to listen on network:
# listen_addresses = '*'
#
# Configure pg_hba.conf to allow trading machine:
# host    all    ominari_user    192.168.1.0/24    md5

# Start market data service
export PG_HOST=localhost
uv run python market_data_heartbeat.py > logs/market_data_heartbeat.log 2>&1 &
```

**Machine 2 (Trading):**
```bash
# Set database host to Machine 1
export PG_HOST=192.168.1.100  # Machine 1 IP

# Start trading services
uv run python carver_heartbeat_with_backtest.py > logs/carver_heartbeat.log 2>&1 &
uv run python execute_trading_solution.py > logs/execute_trading.log 2>&1 &
uv run python position_tracker.py
```

### Three Machine Deployment

**Machine 1 (Database):**
- PostgreSQL with streaming replication to backup
- Monitoring and backup scripts

**Machine 2 (Data Ingestion):**
```bash
export PG_HOST=192.168.1.100  # Machine 1 IP
uv run python market_data_heartbeat.py > logs/market_data_heartbeat.log 2>&1 &
```

**Machine 3 (Trading Logic):**
```bash
export PG_HOST=192.168.1.100  # Machine 1 IP
uv run python carver_heartbeat_with_backtest.py > logs/carver_heartbeat.log 2>&1 &
uv run python execute_trading_solution.py > logs/execute_trading.log 2>&1 &
uv run python position_tracker.py
```

---

## Service Dependencies

```
PostgreSQL Database
    ↓
Market Data Heartbeat (writes market data)
    ↓
Carver Heartbeat (reads markets, generates signals)
    ↓
Execute Trading (reads signals, executes trades)
    ↓
Position Tracker (monitors positions)
```

**Critical Path:**
1. Database must be up first
2. Market data must be fetching
3. Carver can start generating signals
4. Execute trading can begin finding opportunities

**Recovery Order:**
If all services fail, restart in this order:
1. PostgreSQL
2. Market Data Heartbeat
3. Carver Heartbeat
4. Execute Trading

---

## Current Production Setup

**Status:** Running on **1 machine** successfully

**Active Services:**
- PostgreSQL: ✅ 32,095 markets
- Market Data Heartbeat: ✅ Fetching live odds
- Carver Heartbeat: ✅ Generating signals
- Execute Trading: ✅ Finding opportunities (10 per cycle)
- Position Tracker: ✅ Tool available

**Performance Metrics:**
- Portfolio: $9,816.37 (20 open positions)
- Trade execution: ~60 second cycles
- Database queries: <100ms
- Signal generation: ~30 seconds per cycle

**Recommendation:** Continue with 1 machine for now, migrate to 2 machines when:
- Portfolio value exceeds $50,000
- Trade frequency requires lower latency
- Need improved uptime/redundancy

---

## Migration Path

**1 Machine → 2 Machines:**
1. Set up Machine 2 with trading services
2. Update environment variables to point to Machine 1 database
3. Test connectivity
4. Stop trading services on Machine 1
5. Start trading services on Machine 2
6. Monitor for 24 hours
7. Shut down trading services on Machine 1

**2 Machines → 3 Machines:**
1. Set up Machine 3 for database
2. Set up PostgreSQL replication from Machine 1 → Machine 3
3. Test failover
4. Update all services to use Machine 3 as primary
5. Decommission old database on Machine 1
6. Repurpose Machine 1 for data ingestion only

---

## Monitoring

Regardless of configuration, monitor:
- Service health (process running)
- Database connectivity
- Market data freshness (< 5 minutes old)
- Trading execution (opportunities found)
- Position P&L
- System resources (CPU, RAM, disk)

Use the position tracker for real-time monitoring:
```bash
uv run python position_tracker.py --interval 60
```
