#!/bin/bash
if [ -f "/tmp/ominari_auto_deploy_daemon.pid" ]; then
    PID=$(cat "/tmp/ominari_auto_deploy_daemon.pid")
    if ps -p $PID > /dev/null 2>&1; then
        kill $PID
        echo "✅ Daemon stopped (PID: $PID)"
        rm -f "/tmp/ominari_auto_deploy_daemon.pid"
    else
        echo "❌ Daemon not running"
        rm -f "/tmp/ominari_auto_deploy_daemon.pid"
    fi
else
    echo "❌ No daemon PID file found"
fi

# Also kill any orphaned processes
pkill -f "local_auto_deploy.sh" || true
pkill -f "python.*main.py" || true
