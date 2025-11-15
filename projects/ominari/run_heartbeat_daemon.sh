#!/bin/bash
# Heartbeat daemon that auto-restarts on failure

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

LOG_FILE="/tmp/heartbeat_daemon.log"
PID_FILE="/tmp/heartbeat_daemon.pid"

# Function to cleanup on exit
cleanup() {
    echo "Stopping heartbeat daemon..." >> "$LOG_FILE"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        kill $PID 2>/dev/null
        rm -f "$PID_FILE"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

echo "Starting heartbeat daemon at $(date)" >> "$LOG_FILE"

# Activate virtual environment
source .venv/bin/activate

# Keep the heartbeat running
while true; do
    echo "Starting heartbeat process at $(date)" >> "$LOG_FILE"
    
    # Run the robust heartbeat
    python portfolio_heartbeat_robust.py >> "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    
    # Wait for the process
    wait $PID
    EXIT_CODE=$?
    
    echo "Heartbeat process exited with code $EXIT_CODE at $(date)" >> "$LOG_FILE"
    
    # If stopped intentionally, exit
    if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 130 ]; then
        echo "Heartbeat stopped intentionally" >> "$LOG_FILE"
        break
    fi
    
    # Otherwise, restart after a delay
    echo "Restarting heartbeat in 30 seconds..." >> "$LOG_FILE"
    sleep 30
done

cleanup