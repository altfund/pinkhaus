#!/bin/bash
# Real-time system monitor for Ominari trading dashboard
# Shows chunks, trades, and system activity

echo "🎯 OMINARI LIVE SYSTEM MONITOR"
echo "================================"
echo "Watching for: Chunks | Trades | Exposure | Errors"
echo "Press Ctrl+C to exit"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Monitor the log file with color coding
tail -f web_monitor_fixed.log | while read line; do
    # Chunk creation
    if [[ $line == *"Created"*"time-based chunks"* ]]; then
        echo -e "${CYAN}${BOLD}📅 CHUNKS: ${NC}${CYAN}$line${NC}"
    # Chunk details
    elif [[ $line == *"Chunk"*"markets"* ]]; then
        echo -e "${CYAN}   └─ $line${NC}"
    # Selected chunk for trading
    elif [[ $line == *"Selected first chunk"* ]]; then
        echo -e "${GREEN}${BOLD}🎯 ACTIVE: ${NC}${GREEN}$line${NC}"
    # Trade execution
    elif [[ $line == *"Paper trading cycle complete"* ]]; then
        echo -e "${YELLOW}${BOLD}💰 TRADES: ${NC}${YELLOW}$line${NC}"
    # Odds distribution
    elif [[ $line == *"Odds distribution"* ]]; then
        echo -e "${MAGENTA}📊 ODDS: ${NC}${MAGENTA}$line${NC}"
    # Sample odds
    elif [[ $line == *"Sample odds"* ]]; then
        echo -e "${MAGENTA}   └─ $line${NC}"
    # Error or warning
    elif [[ $line == *"ERROR"* ]] || [[ $line == *"error"* ]]; then
        echo -e "${RED}${BOLD}❌ ERROR: ${NC}${RED}$line${NC}"
    elif [[ $line == *"WARNING"* ]] || [[ $line == *"warning"* ]]; then
        echo -e "${YELLOW}⚠️  WARNING: ${NC}${YELLOW}$line${NC}"
    # Client connections
    elif [[ $line == *"Client connected"* ]]; then
        echo -e "${BLUE}🔌 CONNECT: ${NC}${BLUE}$line${NC}"
    # Found markets
    elif [[ $line == *"Found"*"markets for"* ]]; then
        echo -e "${GREEN}✅ MARKETS: ${NC}${GREEN}$line${NC}"
    # Edge calculation
    elif [[ $line == *"Calculating edges"* ]]; then
        echo -e "   📐 $line"
    # Default - show dimmed
    else
        echo -e "\033[2m$line\033[0m"
    fi
done