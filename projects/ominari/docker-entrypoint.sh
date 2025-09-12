#!/bin/bash
set -e

echo "🚀 Starting Ominari Blockchain Trading System"
echo "=============================================="

# Check if database exists, if not initialize it
if [ ! -f "/app/sport_odds.db" ]; then
    echo "📊 No database found - initializing..."
    python -c "from database_v2 import init_database; init_database()"
fi

# Run database migrations if needed
echo "🔄 Running database migrations..."
python -c "
try:
    from alembic.config import Config
    from alembic import command
    alembic_cfg = Config('alembic.ini')
    command.upgrade(alembic_cfg, 'head')
    print('✅ Migrations completed')
except Exception as e:
    print(f'⚠️ Migration skipped: {e}')
"

# Start the requested service based on environment variable
SERVICE=${SERVICE:-trading}

case $SERVICE in
    "trading")
        echo "🏪 Starting trading system..."
        exec python ominari_unified.py
        ;;
    "blockchain-sync")
        echo "⛓️ Starting blockchain sync daemon..."
        exec python blockchain_reader.py --daemon
        ;;
    "paper-trading")
        echo "📄 Starting paper trading system..."
        exec python paper_trading_engine.py
        ;;
    "web-monitor")
        echo "🌐 Starting web monitor..."
        exec python web_monitor.py
        ;;
    "backtest")
        echo "📊 Running backtest..."
        exec python run_backtest.py
        ;;
    "api")
        echo "🔌 Starting API server..."
        exec python -m uvicorn main:app --host 0.0.0.0 --port 8000
        ;;
    *)
        echo "❌ Unknown service: $SERVICE"
        echo "Available services: trading, blockchain-sync, paper-trading, web-monitor, backtest, api"
        exit 1
        ;;
esac