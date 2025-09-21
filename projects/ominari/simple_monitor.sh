#!/bin/bash
# Simple real-time monitor with grep highlighting

echo "🎯 OMINARI SIMPLE MONITOR - Watching for key events"
echo "=================================================="
echo ""

# Monitor multiple patterns
tail -f web_monitor_fixed.log | grep --line-buffered -E "(Created.*chunks|Selected first chunk|Chunk.*markets|Paper trading cycle|Odds distribution|Sample odds|ERROR|exposure)" | while read line; do
    # Add timestamp
    timestamp=$(date '+%H:%M:%S')
    
    # Color based on content
    if [[ $line == *"Created"*"chunks"* ]]; then
        echo "[$timestamp] 📅 CHUNKS: $line"
    elif [[ $line == *"Selected first chunk"* ]]; then
        echo "[$timestamp] 🎯 ACTIVE: $line"
    elif [[ $line == *"Chunk"*"markets"* ]]; then
        echo "[$timestamp]    └─ $line"
    elif [[ $line == *"Paper trading cycle"* ]]; then
        echo "[$timestamp] 💰 TRADES: $line"
    elif [[ $line == *"Odds distribution"* ]]; then
        echo "[$timestamp] 📊 ODDS: $line"
    elif [[ $line == *"Sample odds"* ]]; then
        echo "[$timestamp]    └─ $line"
    elif [[ $line == *"ERROR"* ]] || [[ $line == *"exposure"* ]]; then
        echo "[$timestamp] ❌ ALERT: $line"
    else
        echo "[$timestamp] $line"
    fi
done