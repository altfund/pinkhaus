#!/bin/bash
# Quick start with all fixes

echo "🚀 Starting Ominari with all fixes..."

# Kill any existing instance
pkill -f web_monitor.py 2>/dev/null

# Start fresh
echo "Starting web_monitor.py..."
nohup python3 web_monitor.py > web_monitor_activated.log 2>&1 &
echo "✅ Started! PID: $!"

echo ""
echo "📊 Dashboard: http://localhost:8888"
echo "📝 Logs: tail -f web_monitor_activated.log"
echo "🔍 Monitor: ./simple_monitor.sh"
echo "📈 Status: python3 check_activation_status.py"