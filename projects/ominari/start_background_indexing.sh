#!/bin/bash
# Start background database indexing

echo "Starting background database indexing..."
echo "This will run with low priority to avoid impacting other operations."
echo ""

# Check if already running
if pgrep -f "background_indexer.py" > /dev/null; then
    echo "Background indexer is already running!"
    echo "Check status with: uv run python background_indexer.py --status"
    exit 1
fi

# Start in background with nohup
nohup uv run python background_indexer.py > /dev/null 2>&1 &
PID=$!

echo "Background indexer started with PID: $PID"
echo ""
echo "Commands:"
echo "  Check status:  uv run python background_indexer.py --status"
echo "  View logs:     tail -f background_indexing.log"
echo "  Stop indexing: kill $PID"
echo ""
echo "The indexer will create indexes one by one in the background."
echo "This may take several hours for the 201GB database."
echo "You can continue using the database while indexes are being created."