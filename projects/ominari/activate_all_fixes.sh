#!/bin/bash
# Activate all fixes for Ominari trading system

echo "🚀 ACTIVATING ALL FIXES FOR OMINARI"
echo "===================================="
echo ""

# 1. Check if web_monitor.py is running
echo "1️⃣ Checking current web_monitor status..."
if pgrep -f "web_monitor.py" > /dev/null; then
    echo "   ⚠️  web_monitor.py is running. Stopping it..."
    pkill -f "web_monitor.py"
    sleep 2
else
    echo "   ✅ web_monitor.py is not running"
fi

# 2. Test PostgreSQL paper trading
echo ""
echo "2️⃣ Testing PostgreSQL paper trading integration..."
python3 test_postgres_paper_trading.py
if [ $? -ne 0 ]; then
    echo "   ❌ PostgreSQL test failed! Please check the error above."
    exit 1
fi

# 3. Migrate any existing JSON sessions
echo ""
echo "3️⃣ Checking for JSON sessions to migrate..."
if [ -f "paper_trading_sessions.json" ]; then
    echo "   📋 Found JSON file. Migrating to PostgreSQL..."
    python3 migrate_json_to_postgres.py
else
    echo "   ✅ No JSON file to migrate"
fi

# 4. Fix over-leveraging
echo ""
echo "4️⃣ Running emergency over-leveraging fix..."
echo "   This will reset the current session and create a fresh one."
echo "   Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5
python3 fix_overleveraging_postgres.py

# 5. Verify odds fix
echo ""
echo "5️⃣ Verifying odds query fix..."
python3 verify_odds_fix.py

# 6. Start web_monitor.py
echo ""
echo "6️⃣ Starting web_monitor.py with all fixes..."
echo "   Starting in background with logging..."
nohup python3 web_monitor.py > web_monitor_activated.log 2>&1 &
WEB_PID=$!
echo "   ✅ Started with PID: $WEB_PID"

# 7. Wait for startup
echo ""
echo "7️⃣ Waiting for web monitor to initialize..."
sleep 5

# 8. Check if it's running
if ps -p $WEB_PID > /dev/null; then
    echo "   ✅ Web monitor is running!"
else
    echo "   ❌ Web monitor failed to start. Check web_monitor_activated.log"
    exit 1
fi

# 9. Start monitoring
echo ""
echo "8️⃣ Starting live monitor in new terminal..."
echo ""
echo "===================================="
echo "✨ ALL FIXES ACTIVATED!"
echo ""
echo "📊 Dashboard: http://localhost:8888"
echo "📝 Logs: tail -f web_monitor_activated.log"
echo "🔍 Monitor: ./simple_monitor.sh"
echo ""
echo "💡 Quick commands:"
echo "   - Stop: pkill -f web_monitor.py"
echo "   - Status: ps aux | grep web_monitor"
echo "   - Logs: tail -f web_monitor_activated.log"
echo "   - Monitor: ./simple_monitor.sh"
echo ""
echo "Starting monitor in 3 seconds..."
sleep 3

# Start the monitor
./simple_monitor.sh