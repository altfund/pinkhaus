#!/bin/bash

# Super simple auto-deploy - just run this and forget about it!

echo "🚀 Starting Ominari Auto-Deploy (Simple Version)"
echo "==============================================="
echo
echo "This will:"
echo "✓ Check for git updates every 30 seconds"
echo "✓ Auto-deploy when you push to main"
echo "✓ Keep Ominari running at http://localhost:8888"
echo

# Get to the right directory
cd "$(dirname "$0")/.."

# Start the auto-deploy in background
nohup ./scripts/local_auto_deploy.sh > /tmp/ominari_auto_deploy.log 2>&1 &
echo $! > /tmp/ominari_auto_deploy.pid

echo "✅ Auto-deploy started!"
echo
echo "📊 Dashboard will be at: http://localhost:8888"
echo "📝 Logs: tail -f /tmp/ominari_auto_deploy.log"
echo
echo "To stop: pkill -f local_auto_deploy.sh"
echo
echo "Now just push to main and watch the magic happen! 🎉"