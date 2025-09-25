#!/bin/bash
#
# Ominari Local Development Startup Script
# Starts all services locally without Docker
#

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting Ominari Trading System (Local Development)${NC}"
echo "======================================================="

# Check Python environment
if [ ! -f ".venv/bin/activate" ]; then
    echo -e "${RED}❌ Virtual environment not found. Please run: python -m venv .venv${NC}"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check PostgreSQL
echo -e "\n${YELLOW}Checking PostgreSQL...${NC}"
if pg_isready -h localhost -U ominari_user -d ominari_live > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PostgreSQL is running${NC}"
else
    echo -e "${YELLOW}⚠️  PostgreSQL not accessible. Starting setup...${NC}"
    python run_with_postgres.py &
    sleep 5
fi

# Function to start service in background
start_service() {
    local service_name=$1
    local command=$2
    local log_file="logs/${service_name}.log"
    
    # Check if service is already running
    if pgrep -f "$command" > /dev/null; then
        echo -e "${YELLOW}⚠️  $service_name already running${NC}"
    else
        echo -e "${GREEN}Starting $service_name...${NC}"
        mkdir -p logs
        nohup python $command > "$log_file" 2>&1 &
        echo $! > "logs/${service_name}.pid"
        echo -e "${GREEN}✅ $service_name started (PID: $!)${NC}"
    fi
}

# Start services
echo -e "\n${YELLOW}Starting services...${NC}"

# 1. Web Dashboard (highest priority - user interface)
start_service "web_dashboard" "web_monitor_unified.py"
sleep 2

# 2. Blockchain Sync Daemon
start_service "blockchain_sync" "blockchain_reader.py --daemon"
sleep 2

# 3. Paper Trading Engine
start_service "paper_trading" "paper_trading_engine.py"
sleep 2

# 4. API Server (optional)
# start_service "api_server" "-m uvicorn main:app --host 0.0.0.0 --port 8000"

echo -e "\n${GREEN}✅ All services started!${NC}"
echo "========================================"
echo -e "${GREEN}Access points:${NC}"
echo "  - Web Dashboard: http://localhost:8888"
echo "  - API (if enabled): http://localhost:8000"
echo "  - PostgreSQL: localhost:5432"
echo ""
echo -e "${YELLOW}Logs:${NC}"
echo "  - Web Dashboard: logs/web_dashboard.log"
echo "  - Blockchain Sync: logs/blockchain_sync.log"
echo "  - Paper Trading: logs/paper_trading.log"
echo ""
echo -e "${YELLOW}To stop all services:${NC}"
echo "  ./stop_local.sh"
echo ""
echo -e "${GREEN}Happy Trading! 📈${NC}"