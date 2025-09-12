#!/bin/bash
# Debug Dashboard - Troubleshoot web dashboard issues

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "🔍 Ominari Dashboard Debugger"
echo "============================="

# Check if dashboard is accessible
echo -e "${BLUE}[CHECK 1]${NC} Testing dashboard accessibility..."
if curl -s http://localhost:8888/api/status > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC} Dashboard is accessible at http://localhost:8888"
else
    echo -e "${RED}❌${NC} Dashboard is not accessible"
    
    # Check if web_monitor process is running
    echo -e "${BLUE}[CHECK 1a]${NC} Checking web_monitor process..."
    WEB_PROC=$(ps aux | grep -E "python.*web_monitor" | grep -v grep)
    if [ -n "$WEB_PROC" ]; then
        echo -e "${YELLOW}⚠️${NC} web_monitor process found but not responding:"
        echo "$WEB_PROC"
        echo -e "${BLUE}[ACTION]${NC} Restarting web_monitor..."
        pkill -f web_monitor.py
        sleep 2
        python web_monitor.py > logs/web_monitor.log 2>&1 &
        WEB_PID=$!
        echo "New web_monitor PID: $WEB_PID"
    else
        echo -e "${RED}❌${NC} No web_monitor process running"
        echo -e "${BLUE}[ACTION]${NC} Starting web_monitor..."
        python web_monitor.py > logs/web_monitor.log 2>&1 &
        WEB_PID=$!
        echo "Started web_monitor PID: $WEB_PID"
    fi
    
    # Wait and test again
    echo -e "${BLUE}[WAIT]${NC} Waiting 10 seconds for startup..."
    sleep 10
    
    if curl -s http://localhost:8888/api/status > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} Dashboard is now accessible!"
    else
        echo -e "${RED}❌${NC} Dashboard still not accessible - check logs"
        echo "Recent web_monitor.log:"
        tail -20 logs/web_monitor.log 2>/dev/null || echo "No log file found"
    fi
fi

# Check port conflicts
echo ""
echo -e "${BLUE}[CHECK 2]${NC} Checking port usage..."
PORT_8888=$(netstat -tuln 2>/dev/null | grep :8888 || echo "")
if [ -n "$PORT_8888" ]; then
    echo -e "${GREEN}✅${NC} Port 8888 is in use:"
    echo "$PORT_8888"
else
    echo -e "${RED}❌${NC} Port 8888 is not in use - web server not listening"
fi

# Check database access
echo ""
echo -e "${BLUE}[CHECK 3]${NC} Checking database access..."
if [ -f "sport_odds.db" ]; then
    DB_SIZE=$(du -h sport_odds.db | cut -f1)
    echo -e "${GREEN}✅${NC} SQLite database found ($DB_SIZE)"
    
    # Test database query
    if uv run python -c "
import sqlite3
conn = sqlite3.connect('sport_odds.db', timeout=5)
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM market LIMIT 1')
count = cursor.fetchone()[0]
conn.close()
print(f'Database query successful: {count:,} markets')
    " 2>/dev/null; then
        echo -e "${GREEN}✅${NC} Database is accessible"
    else
        echo -e "${YELLOW}⚠️${NC} Database query failed - might be locked"
        echo "Checking for processes using database:"
        lsof sport_odds.db 2>/dev/null || echo "No processes found using database"
    fi
else
    echo -e "${YELLOW}⚠️${NC} No SQLite database found"
    
    # Check for hybrid setup
    if docker ps | grep -q "postgres-hybrid"; then
        echo -e "${GREEN}✅${NC} Hybrid PostgreSQL setup detected"
        uv run python verify_postgresql_setup.py 2>/dev/null || echo "PostgreSQL verification failed"
    else
        echo -e "${RED}❌${NC} No database found (neither SQLite nor PostgreSQL)"
    fi
fi

# Check data collection
echo ""
echo -e "${BLUE}[CHECK 4]${NC} Checking data collection service..."
DATA_PROC=$(ps aux | grep -E "python.*free_data_pull" | grep -v grep)
if [ -n "$DATA_PROC" ]; then
    echo -e "${GREEN}✅${NC} Data collection process running:"
    echo "$DATA_PROC"
else
    echo -e "${YELLOW}⚠️${NC} No data collection process found"
    echo -e "${BLUE}[ACTION]${NC} Starting data collection..."
    python free_data_pull_normalized.py > logs/data_collection.log 2>&1 &
    DATA_PID=$!
    echo "Started data collection PID: $DATA_PID"
fi

# Check logs for errors
echo ""
echo -e "${BLUE}[CHECK 5]${NC} Checking recent logs for errors..."
if [ -f "logs/web_monitor.log" ]; then
    RECENT_ERRORS=$(tail -50 logs/web_monitor.log | grep -E "(ERROR|CRITICAL|Exception|Traceback)" | wc -l)
    if [ "$RECENT_ERRORS" -gt 0 ]; then
        echo -e "${RED}❌${NC} Found $RECENT_ERRORS recent errors in web_monitor.log:"
        tail -50 logs/web_monitor.log | grep -E "(ERROR|CRITICAL|Exception)" | tail -5
    else
        echo -e "${GREEN}✅${NC} No recent errors in web_monitor.log"
    fi
else
    echo -e "${YELLOW}⚠️${NC} No web_monitor.log found"
fi

# System recommendations
echo ""
echo -e "${BLUE}[RECOMMENDATIONS]${NC}"

if curl -s http://localhost:8888/api/status > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Dashboard is working!${NC}"
    echo "• Open http://localhost:8888 in your browser"
    echo "• Check Markets tab for live data"
    echo "• Use Trading tab for paper trading"
else
    echo -e "${RED}❌ Dashboard needs attention${NC}"
    echo "• Check logs: tail -f logs/web_monitor.log"
    echo "• Try restarting: ./stop_system.sh && ./start_system.sh"
    echo "• Consider hybrid setup: uv run python hybrid_sync_deployment.py"
fi

echo ""
echo -e "${BLUE}Monitoring Commands:${NC}"
echo "• View all logs: ./ominari_logs.sh --all"
echo "• System health: ./health_check.sh"
echo "• Debug again: ./debug_dashboard.sh"
echo ""