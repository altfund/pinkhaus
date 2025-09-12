# Future Developer Guide - Getting Ominari Up and Running

## 🚨 **TL;DR - Quick Dashboard Access**

**The web dashboard everyone is looking for:**

```bash
# Quick start (works 90% of the time)
./start_system.sh

# Then open: http://localhost:8888
```

**If that doesn't work, try the hybrid system:**
```bash
uv run python hybrid_sync_deployment.py
```

## 🌐 **Port Reference - Where Everything Lives**

| Service | URL | Purpose | Credentials |
|---------|-----|---------|-------------|
| **Main Dashboard** | `http://localhost:8888` | Trading interface, markets, portfolio | None |
| **Grafana Monitoring** | `http://localhost:3000` | Advanced metrics, system health | admin/ominari123 |
| **PostgreSQL** | `localhost:5435` | Hybrid database (new architecture) | ominari_user/ominari_2025_secure |
| **Redis** | `localhost:6380` | Caching (hybrid setup) | None |
| **Graph Node** | `localhost:18000` | Blockchain indexing | None |
| **IPFS** | `localhost:15001` | Decentralized storage | None |

## 🚀 **Startup Options (Choose Your Adventure)**

### Option A: Simple Dashboard (Recommended for Dev)
```bash
# Clone and basic setup
git clone git@github.com:ScheierVentures/ominari.git
cd ominari

# Install dependencies  
uv sync

# Start the basic system
./start_system.sh

# Access dashboard
open http://localhost:8888
```

**What you get:**
- ✅ Web dashboard on port 8888
- ✅ Live market data and paper trading
- ✅ SQLite database (216GB historical data)
- ✅ Basic monitoring

### Option B: Hybrid Architecture (Production Ready)
```bash
# One-command deployment
uv run python hybrid_sync_deployment.py

# Start hybrid system
./start_hybrid_system.sh

# Access dashboard  
open http://localhost:8888
```

**What you get:**
- ✅ PostgreSQL + SQLite hybrid database
- ✅ Fresh optimized system with synthetic context
- ✅ Cross-chain arbitrage capabilities
- ✅ Full blockchain integration ready
- ✅ 90-day backfill for immediate context

### Option C: Full Monitoring Stack
```bash
# Start monitoring services
cd monitoring
docker-compose up -d

# Start main system
cd ..
./start_system.sh

# Access Grafana
open http://localhost:3000
```

**What you get:**
- ✅ Professional monitoring (Prometheus + Grafana)
- ✅ System metrics and alerting
- ✅ Performance dashboards
- ✅ Log aggregation

### Option D: Development Mode (Custom)
```bash
# Environment setup
cp .env.template .env
# Edit .env with your settings

# Manual component startup
python web_monitor.py &          # Dashboard
python free_data_pull.py &       # Data collection  
python paper_trading_engine.py & # Trading engine

# Access dashboard
open http://localhost:8888
```

**What you get:**
- ✅ Full control over each component
- ✅ Easy debugging and development
- ✅ Custom configuration

## 🔍 **Troubleshooting Common Issues**

### ❌ "I can't find the web dashboard!"

**Symptom:** Can't access `http://localhost:8888`

**Solutions:**
```bash
# Check if web server is running
curl http://localhost:8888/api/status

# Check processes
ps aux | grep web_monitor

# Check logs
tail -f logs/web_monitor.log

# Restart web monitor
pkill -f web_monitor.py
python web_monitor.py &
```

### ❌ "Dashboard shows no data!"

**Symptom:** Web dashboard loads but shows empty markets

**Solutions:**
```bash
# Check data collection service
ps aux | grep free_data_pull

# Check database
python safe_query.py recent --hours 24

# Restart data collection
pkill -f free_data_pull
python free_data_pull_normalized.py &

# Check database size
ls -lh sport_odds.db
```

### ❌ "Database locked errors!"

**Symptom:** SQLite database lock errors in logs

**Solutions:**
```bash
# Kill all processes using database
lsof sport_odds.db
pkill -f "python.*sport_odds"

# Wait and restart
sleep 5
./start_system.sh

# Alternative: Use hybrid architecture
uv run python hybrid_sync_deployment.py
```

### ❌ "Docker container conflicts!"

**Symptom:** Port already in use errors

**Solutions:**
```bash
# Check what's using ports
netstat -tulpn | grep -E "(5435|6380|8888|3000)"

# Stop conflicting containers
docker stop $(docker ps -q)

# Clean up
docker-compose down
docker system prune -f

# Restart with different ports
# Edit docker-compose files if needed
```

### ❌ "Permission denied errors!"

**Symptom:** Cannot execute scripts or access files

**Solutions:**
```bash
# Fix script permissions
chmod +x *.sh
chmod +x *.py

# Fix ownership
sudo chown -R $USER:$USER .

# Fix Python environment
uv sync
```

## 📊 **System Health Checks**

### Quick Health Check
```bash
# System status
./health_check.sh

# Or manual checks:
docker ps                              # Containers
ps aux | grep -E "(ominari|web_monitor)" # Processes
curl http://localhost:8888/api/status  # Dashboard API
tail -f logs/*.log                     # Recent logs
```

### Database Health
```bash
# SQLite stats (safe queries only!)
python safe_query.py summary

# PostgreSQL (hybrid mode)
uv run python verify_postgresql_setup.py

# Hybrid system test
uv run python hybrid_database_access.py
```

### Service Dependencies
```bash
# Check required services
docker ps | grep postgres     # PostgreSQL (hybrid)
docker ps | grep redis        # Redis (caching)
ps aux | grep ominari_daemon  # Background daemon
```

## ⚙️ **Environment Configuration**

### Basic .env Setup
```bash
# Copy template
cp .env.template .env

# Essential settings for development
PAPER_TRADING_MODE=true
RISK_PRESET=conservative
API_PORT=8888
LOG_LEVEL=INFO

# Database (choose one)
DATABASE_URL=sqlite:///sport_odds.db              # Legacy
# OR
PG_HOST=localhost                                 # Hybrid
PG_PORT=5435
PG_USER=ominari_user
PG_PASSWORD=ominari_2025_secure
```

### API Keys (Optional)
```bash
# For live data (not required for development)
ODDS_API_KEY=your-key-here
OVERTIME_API_KEY=your-key-here
ALCHEMY_API_KEY=your-key-here
```

## 🔧 **Development Commands**

### Daily Development Workflow
```bash
# Morning startup
git pull origin feature/ominari-updates
uv sync
./start_system.sh

# Check dashboard
open http://localhost:8888

# Monitor logs while developing
./ominari_logs.sh --all

# Evening shutdown
./stop_system.sh
```

### Useful Commands
```bash
# View trading activity
./ominari_logs.sh --trades

# Check for errors
./ominari_logs.sh --errors

# Manual paper trading test
python paper_trading_engine.py --test

# Database queries (safe)
python safe_query.py count Market
python safe_query.py sample Odd --limit 10

# System health
python system_health_check.py
```

## 📁 **Key Files Reference**

### Startup Scripts
- `start_system.sh` - Main system startup
- `stop_system.sh` - Clean shutdown
- `start_hybrid_system.sh` - Hybrid architecture startup
- `health_check.sh` - System health verification

### Web Interfaces
- `web_monitor.py` - Main dashboard (port 8888)
- `web_monitor_enhanced.py` - Advanced features
- `web_monitor_simple.py` - Minimal version

### Configuration
- `.env.template` - Environment variables template
- `.env.production` - Production configuration
- `pyproject.toml` - Dependencies and project config

### Core Components
- `paper_trading_engine.py` - Trading logic
- `free_data_pull_normalized.py` - Data collection
- `hybrid_database_access.py` - Unified data access
- `arbitrage_engine.py` - Cross-chain opportunities

## 🎯 **Architecture Decision Tree**

```
New to Ominari?
├── Just want to see the dashboard? → Option A (Simple)
├── Setting up for production? → Option B (Hybrid)  
├── Need advanced monitoring? → Option C (Full Stack)
└── Developing new features? → Option D (Development)

Having issues?
├── Dashboard not loading? → Check port 8888, restart web_monitor.py
├── No market data? → Check data collection service
├── Database errors? → Use hybrid architecture or restart cleanly
└── Docker conflicts? → Stop all containers, clean up, restart
```

## 🚨 **Emergency Recovery**

If everything is broken:
```bash
# Nuclear option - clean restart
pkill -f python
docker stop $(docker ps -q)
docker system prune -f

# Wait and restart fresh
sleep 10

# Try hybrid deployment (most reliable)
uv run python hybrid_sync_deployment.py

# Access dashboard
open http://localhost:8888
```

## 📞 **Getting Help**

### Log Files to Check
- `logs/web_monitor.log` - Dashboard issues
- `logs/data_collection.log` - Market data problems
- `ominari_unified.log` - Main system log
- `hybrid_deployment.log` - Hybrid system issues

### Common Log Patterns
```bash
# Good signs
grep -E "Dashboard.*ready|Server.*running|✅" logs/*.log

# Warning signs  
grep -E "ERROR|CRITICAL|Failed|Timeout" logs/*.log

# Trading activity
grep -E "trade|order|fill|position" logs/*.log
```

---

## 🎉 **Success Indicators**

You know everything is working when:

✅ **Dashboard loads** at `http://localhost:8888`  
✅ **Markets tab** shows upcoming games with odds  
✅ **Trading tab** shows portfolio and positions  
✅ **Logs show** regular data updates  
✅ **Paper trading** executes trades automatically  

**Now you're ready to trade! 🚀**

*Remember: System runs in paper trading mode by default for safety.*