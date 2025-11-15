# Ominari Trading System Deployment Guide

## Current Status

✅ **Heartbeat is integrated into main system** - When you run `python main.py`, it automatically includes:
- Paper trading system
- Discord notifications
- Hourly portfolio heartbeat
- Backtesting analysis (every 4 hours)

## Quick Start

```bash
# 1. Start everything
python main.py

# This starts:
# - PostgreSQL database
# - Web dashboard (port 8888)
# - Performance monitor (port 8889)
# - Integrated trading system WITH heartbeat
# - Real-time edge detection
# - Paper trading
```

## What Happens Automatically

When you run `main.py`:

1. **Database**: PostgreSQL starts on port 5999
2. **Web Dashboard**: Available at http://localhost:8888
3. **Trading System**: Runs `integrated_trading_with_heartbeat.py` which includes:
   - Paper trading engine
   - Edge detection
   - Discord notifications for trades
   - **Portfolio heartbeat every hour**
   - Backtesting insights every 4 hours

## Heartbeat Features

The heartbeat sends Discord updates every hour with:
- Portfolio P&L and ROI
- Active bets and coverage
- Market opportunities
- Blockchain data (gas prices, ETH price)
- System status

Every 4 hours, it also includes:
- Historical edge performance
- Kelly fraction analysis
- Strategy recommendations
- Expected ROI projections

## Deployment on New Machine

```bash
# 1. Clone repository
git clone https://github.com/ScheierVentures/ominari.git
cd ominari/projects/ominari

# 2. Install uv (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install dependencies
uv sync

# 4. Create .env file
cat > .env << EOF
DISCORD_WEBHOOK_URL=your_discord_webhook_here
DATABASE_URL=postgresql://ominari_user:ominari_2025_secure@localhost:5999/ominari_production
PG_PORT=5999
PG_USER=ominari_user
PG_PASSWORD=ominari_2025_secure
PG_DB=ominari_production
