#!/bin/bash
#
# Stop all Ominari services
#

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🛑 Stopping Ominari Trading System${NC}"
echo "======================================"

# Function to stop service
stop_service() {
    local service_name=$1
    local pid_file="logs/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if kill -0 $pid 2>/dev/null; then
            echo -e "${YELLOW}Stopping $service_name (PID: $pid)...${NC}"
            kill $pid
            rm "$pid_file"
            echo -e "${GREEN}✅ $service_name stopped${NC}"
        else
            echo -e "${YELLOW}⚠️  $service_name not running (stale PID)${NC}"
            rm "$pid_file"
        fi
    else
        echo -e "${YELLOW}⚠️  $service_name PID file not found${NC}"
    fi
}

# Stop all services
stop_service "web_dashboard"
stop_service "blockchain_sync"
stop_service "paper_trading"
stop_service "api_server"

# Also kill any remaining Python processes for these services
echo -e "\n${YELLOW}Cleaning up any remaining processes...${NC}"
pkill -f "web_monitor_unified.py" 2>/dev/null
pkill -f "blockchain_reader.py" 2>/dev/null
pkill -f "paper_trading_engine.py" 2>/dev/null
pkill -f "uvicorn main:app" 2>/dev/null

echo -e "\n${GREEN}✅ All services stopped${NC}"