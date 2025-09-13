#!/bin/bash

# Ominari Trading System Stop Script
# This script cleanly stops all components

echo "🛑 Stopping Ominari Trading System..."
echo "=================================="

# Try to load PIDs from file
if [ -f ".ominari_pids" ]; then
    source .ominari_pids
    
    if [ ! -z "$WEB_PID" ]; then
        echo "📋 Stopping web monitor (PID: $WEB_PID)..."
        kill -TERM $WEB_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$DATA_PID" ]; then
        echo "📋 Stopping data collection (PID: $DATA_PID)..."
        kill -TERM $DATA_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$INDEXER_PID" ]; then
        echo "📋 Stopping background indexer (PID: $INDEXER_PID)..."
        kill -TERM $INDEXER_PID 2>/dev/null || true
    fi
else
    echo "⚠️  PID file not found, using process names..."
fi

# Fallback: Kill by process name
echo "📋 Ensuring all processes are stopped..."
pkill -f "python web_monitor.py" 2>/dev/null || true
pkill -f "python free_data_pull.py" 2>/dev/null || true
pkill -f "python free_data_pull_normalized.py" 2>/dev/null || true
pkill -f "python background_indexer.py" 2>/dev/null || true

# Clean up PID file
rm -f .ominari_pids

echo ""
echo "✅ All processes stopped"
echo "=================================="