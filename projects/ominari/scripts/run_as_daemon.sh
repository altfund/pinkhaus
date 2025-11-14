#!/bin/bash

# Run Ominari Auto-Deploy as a daemon

echo "🚀 Starting Ominari Auto-Deploy as Daemon"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMINARI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="/tmp/ominari_auto_deploy_daemon.pid"
LOG_FILE="/tmp/ominari_auto_deploy.log"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "❌ Auto-deploy daemon is already running with PID $OLD_PID"
        echo "   To stop it: kill $(cat $PID_FILE)"
        exit 1
    fi
fi

# Start the auto-deploy script as a daemon
echo "Starting daemon..."
nohup "$SCRIPT_DIR/local_auto_deploy.sh" > "$LOG_FILE" 2>&1 &
DAEMON_PID=$!

# Save PID
echo $DAEMON_PID > "$PID_FILE"

# Wait a moment to ensure it started
sleep 2

# Check if daemon is running
if ps -p $DAEMON_PID > /dev/null 2>&1; then
    echo "✅ Daemon started successfully!"
    echo
    echo "📊 Dashboard: http://localhost:8888"
    echo "📈 Performance: http://localhost:8889"
    echo
    echo "🔍 Monitoring:"
    echo "   PID: $DAEMON_PID"
    echo "   Logs: tail -f $LOG_FILE"
    echo "   Status: ps -p $DAEMON_PID"
    echo
    echo "🛑 To stop daemon:"
    echo "   kill $DAEMON_PID"
    echo "   # or"
    echo "   kill \$(cat $PID_FILE)"
    echo
    echo "The daemon will:"
    echo "✓ Check for git updates every 30 seconds"
    echo "✓ Auto-deploy when you push to main"
    echo "✓ Keep running even if you log out"
    echo "✓ Restart Ominari if it crashes"
    
    # Create stop script for convenience
    cat > "$SCRIPT_DIR/stop_daemon.sh" << EOF
#!/bin/bash
if [ -f "$PID_FILE" ]; then
    PID=\$(cat "$PID_FILE")
    if ps -p \$PID > /dev/null 2>&1; then
        kill \$PID
        echo "✅ Daemon stopped (PID: \$PID)"
        rm -f "$PID_FILE"
    else
        echo "❌ Daemon not running"
        rm -f "$PID_FILE"
    fi
else
    echo "❌ No daemon PID file found"
fi

# Also kill any orphaned processes
pkill -f "local_auto_deploy.sh" || true
pkill -f "python.*main.py" || true
EOF
    chmod +x "$SCRIPT_DIR/stop_daemon.sh"
    
    echo
    echo "💡 Created helper script: ./scripts/stop_daemon.sh"
else
    echo "❌ Failed to start daemon!"
    rm -f "$PID_FILE"
    exit 1
fi