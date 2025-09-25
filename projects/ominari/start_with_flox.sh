#!/bin/bash
# Start with proper flox environment

echo "🚀 Starting Ominari with flox environment..."

# Kill any existing instance
pkill -f web_monitor.py 2>/dev/null

# Check if in flox environment
if [ -z "$FLOX_ENV" ]; then
    echo "❌ Not in flox environment!"
    echo "Please run: flox activate"
    echo "Then run this script again"
    exit 1
fi

# Start with python from flox
echo "Starting web_monitor.py with flox python..."
python web_monitor.py > web_monitor_activated.log 2>&1 &
PID=$!
echo "✅ Started! PID: $PID"

# Wait a bit
sleep 3

# Check if still running
if ps -p $PID > /dev/null; then
    echo "✅ Web monitor is running!"
    echo ""
    echo "📊 Dashboard: http://localhost:8888"
    echo "📝 Logs: tail -f web_monitor_activated.log"
    echo "🔍 Monitor: ./simple_monitor.sh"
    echo "📈 Status: python3 check_activation_status.py"
else
    echo "❌ Web monitor crashed! Check logs:"
    tail -20 web_monitor_activated.log
fi