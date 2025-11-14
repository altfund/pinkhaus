#!/bin/bash

# Local Auto-Deploy Script for Ominari
# This runs locally and monitors your git repository for changes

set -e

# Configuration
OMINARI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH_TO_WATCH="feature/ominari-updates"  # Your current branch
CHECK_INTERVAL=30  # Check every 30 seconds
LOG_FILE="/tmp/ominari_auto_deploy.log"
PID_FILE="/tmp/ominari_auto_deploy.pid"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    echo -e "${1}" | tee -a "$LOG_FILE"
}

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        log "${RED}Auto-deploy is already running with PID $OLD_PID${NC}"
        exit 1
    fi
fi

# Save PID
echo $$ > "$PID_FILE"

# Cleanup on exit
cleanup() {
    rm -f "$PID_FILE"
    log "${YELLOW}Auto-deploy stopped${NC}"
}
trap cleanup EXIT

log "${GREEN}🚀 Starting Ominari Local Auto-Deploy${NC}"
log "${BLUE}Watching branch: $BRANCH_TO_WATCH${NC}"
log "${BLUE}Check interval: ${CHECK_INTERVAL}s${NC}"
log "${BLUE}Project directory: $OMINARI_DIR${NC}"

cd "$OMINARI_DIR"

# Function to get current commit
get_current_commit() {
    git rev-parse HEAD 2>/dev/null || echo ""
}

# Function to deploy
deploy() {
    local commit_msg=$(git log -1 --pretty=%B)
    log "${GREEN}🔄 Deploying new version...${NC}"
    log "${BLUE}Commit: $(git rev-parse --short HEAD) - $commit_msg${NC}"
    
    # Stop existing Ominari
    log "Stopping existing Ominari services..."
    pkill -f "python.*main.py" || true
    pkill -f "web_dashboard" || true
    pkill -f "automated_trading" || true
    sleep 3
    
    # Update dependencies
    log "Updating dependencies..."
    uv sync
    
    # Set environment
    export PG_PORT=5999
    export PG_DB=ominari_production
    export DATABASE_URL="postgresql://ominari_user:ominari_2025_secure@localhost:5999/ominari_production"
    
    # Start Ominari
    log "${GREEN}Starting Ominari Trading System...${NC}"
    nohup "$OMINARI_DIR/.venv/bin/python" "$OMINARI_DIR/main.py" > /tmp/ominari_main.log 2>&1 &
    local ominari_pid=$!
    
    # Wait for startup
    sleep 15
    
    # Verify deployment
    if curl -sf http://localhost:8888/health > /dev/null; then
        log "${GREEN}✅ Deployment successful!${NC}"
        log "📊 Dashboard: http://localhost:8888"
        log "📈 Performance: http://localhost:8889"
        log "🔍 Logs: tail -f /tmp/ominari_main.log"
        
        # Send notification (if available)
        if command -v notify-send &> /dev/null; then
            notify-send "Ominari Deployed" "New version deployed successfully to localhost:8888" -i dialog-information
        fi
    else
        log "${RED}❌ Deployment failed!${NC}"
        log "Check logs: tail -f /tmp/ominari_main.log"
        return 1
    fi
}

# Initial deployment
LAST_COMMIT=$(get_current_commit)
deploy

# Monitor for changes
log "${BLUE}🔍 Monitoring for changes...${NC}"

while true; do
    # Fetch latest changes
    git fetch origin "$BRANCH_TO_WATCH" --quiet 2>/dev/null || true
    
    # Check if local branch is behind
    LOCAL=$(git rev-parse HEAD 2>/dev/null || echo "")
    REMOTE=$(git rev-parse "origin/$BRANCH_TO_WATCH" 2>/dev/null || echo "")
    
    if [ -n "$LOCAL" ] && [ -n "$REMOTE" ] && [ "$LOCAL" != "$REMOTE" ]; then
        log "${YELLOW}📥 New changes detected on origin/$BRANCH_TO_WATCH${NC}"
        
        # Pull changes
        log "Pulling changes..."
        git pull origin "$BRANCH_TO_WATCH"
        
        # Deploy
        deploy
        
        # Update last commit
        LAST_COMMIT=$(get_current_commit)
    fi
    
    # Also check if current branch changed (manual local commits)
    CURRENT_COMMIT=$(get_current_commit)
    if [ "$CURRENT_COMMIT" != "$LAST_COMMIT" ]; then
        log "${YELLOW}🔄 Local changes detected${NC}"
        deploy
        LAST_COMMIT="$CURRENT_COMMIT"
    fi
    
    # Wait before next check
    sleep "$CHECK_INTERVAL"
done