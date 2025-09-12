#!/bin/bash

# Ominari Trading System Startup Script
# This script starts all components in the correct order

set -e  # Exit on any error

echo "🚀 Starting Ominari Trading System..."
echo "=================================="

# Check if database exists
if [ ! -f "sport_odds.db" ]; then
    echo "❌ Error: sport_odds.db not found!"
    echo "Please ensure the database is in the current directory"
    exit 1
fi

# Kill any existing processes
echo "📋 Cleaning up existing processes..."
pkill -f "python web_monitor.py" 2>/dev/null || true
pkill -f "python free_data_pull.py" 2>/dev/null || true
pkill -f "python free_data_pull_normalized.py" 2>/dev/null || true
sleep 2

# Start data collection in background (using normalized version)
echo "📊 Starting normalized data collection service..."
nohup python free_data_pull_normalized.py > logs/data_collection.log 2>&1 &
DATA_PID=$!
echo "   Data collection PID: $DATA_PID"

# Wait for data collection to initialize
sleep 3

# Start web monitor
echo "🌐 Starting web monitor dashboard..."
nohup python web_monitor.py > logs/web_monitor.log 2>&1 &
WEB_PID=$!
echo "   Web monitor PID: $WEB_PID"

# Wait for web server to start
echo "⏳ Waiting for web server to start..."
for i in {1..10}; do
    if curl -s http://localhost:8888/api/status > /dev/null 2>&1; then
        echo "✅ Web server is ready!"
        break
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "=================================="
echo "✨ System started successfully!"
echo ""
echo "📊 Dashboard: http://localhost:8888"
echo ""
echo "🔍 Monitoring:"
echo "   - Data collection log: tail -f logs/data_collection.log"
echo "   - Web monitor log: tail -f logs/web_monitor.log"
echo "   - Trading activity: Check dashboard trading tab"
echo ""
echo "⚙️  Key Functions:"
echo "   - Markets update every 15 minutes automatically"
echo "   - Paper trading evaluates positions every 15 minutes"
echo "   - Click 'Execute Trades Now' for manual execution"
echo ""
echo "🛑 To stop the system:"
echo "   ./stop_system.sh"
echo ""
echo "Process IDs saved to: .ominari_pids"

# Save PIDs for stop script
echo "DATA_PID=$DATA_PID" > .ominari_pids
echo "WEB_PID=$WEB_PID" >> .ominari_pids

# Optional: Start background indexer if needed
# echo "🔍 Starting background indexer..."
# nohup python background_indexer.py > logs/indexer.log 2>&1 &
# INDEXER_PID=$!
# echo "INDEXER_PID=$INDEXER_PID" >> .ominari_pids