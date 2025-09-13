#!/bin/bash
# Startup script for Ominari Trading System
# This script starts all necessary components for paper trading

echo "Starting Ominari Trading System..."
echo "================================"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Load environment variables if available
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install/update dependencies
echo "Checking dependencies..."
uv sync

# Create necessary directories
mkdir -p logs
mkdir -p paper_trades
mkdir -p signals
mkdir -p alpha_reports
mkdir -p performance_reports
mkdir -p betting_reports

# Check if Graph Node is needed (optional)
if [ -f "docker-compose.graph-node-alt.yml" ]; then
    echo "Checking Graph Node status..."
    if ! docker ps | grep -q graph-node; then
        echo "Starting Graph Node infrastructure..."
        docker-compose -f docker-compose.graph-node-alt.yml up -d
        sleep 10  # Wait for services to start
    else
        echo "Graph Node already running"
    fi
fi

# Kill any existing Ominari daemon
if [ -f "ominari_daemon.pid" ]; then
    OLD_PID=$(cat ominari_daemon.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Stopping existing Ominari daemon (PID: $OLD_PID)..."
        kill $OLD_PID
        sleep 2
    fi
fi

# Start the unified Ominari system
echo "Starting Ominari unified system..."
nohup python ominari_unified.py > logs/ominari_$(date +%Y%m%d_%H%M%S).log 2>&1 &
NEW_PID=$!
echo $NEW_PID > ominari_daemon.pid

echo "Ominari system started with PID: $NEW_PID"
echo "Logs available at: logs/"
echo ""
echo "To check status: ps -p $NEW_PID"
echo "To view logs: tail -f logs/ominari_*.log"
echo "To stop: kill $NEW_PID"
echo ""
echo "Paper trading is now active!"