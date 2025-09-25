#!/bin/bash
# Run the Ominari dashboard with all updates

cd /home/ess/Documents/apps/ominari/projects/ominari

# Kill any existing processes
pkill -f web_monitor_unified
pkill -f "python.*8889"
sleep 2

# Set environment variables
export PG_HOST='localhost'
export PG_PORT='5999'
export PG_USER='ominari_user'
export PG_PASSWORD='ominari_2025_secure'
export PG_DB='ominari_production'
export USE_POSTGRESQL='1'
export DASHBOARD_PORT='8889'

echo "🚀 Starting Ominari Dashboard..."
echo ""
echo "✅ Features enabled:"
echo "   - ⚽ Soccer-only filter (locked)"
echo "   - 📝 Paper trading mode"
echo "   - 🎯 Underdog betting strategy"
echo "   - 📈 Real-time market data"
echo ""
echo "🌐 Dashboard will be available at: http://localhost:8889"
echo ""

# Activate flox environment and run
source .flox/run/x86_64-linux.ominari.dev/activate
exec python web_monitor_unified.py