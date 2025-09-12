#!/bin/bash
# Start background migration with resource limits

echo "Starting background migration..."

# Set resource limits
ulimit -m 2097152  # 2GB memory limit
ulimit -v 3145728  # 3GB virtual memory limit

# Run with ionice for low disk priority
ionice -c 3 nice -n 19 python3 background_migration.py \
    --batch-size 500 \
    --sleep 2 \
    --nice 19 \
    > migration_background.log 2>&1 &

PID=$!
echo "Migration started with PID: $PID"
echo $PID > migration.pid

echo "To monitor progress:"
echo "  tail -f migration_background.log"
echo "  tail -f background_migration.log"
echo ""
echo "To stop migration:"
echo "  kill \$(cat migration.pid)"
echo ""
echo "The migration will automatically pause when system resources are low."