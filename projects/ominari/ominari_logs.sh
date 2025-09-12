#!/bin/bash
# Ominari Log Viewer - CLI interface for monitoring logs

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Default log file
LOG_FILE="ominari_unified.log"
FOLLOW=true
FILTER=""
LINES=50

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            LOG_FILE="$2"
            shift 2
            ;;
        -n|--lines)
            LINES="$2"
            FOLLOW=false
            shift 2
            ;;
        --no-follow)
            FOLLOW=false
            shift
            ;;
        -g|--grep)
            FILTER="$2"
            shift 2
            ;;
        --trades)
            FILTER="trade|order|fill"
            shift
            ;;
        --errors)
            FILTER="ERROR|CRITICAL|error|Error"
            shift
            ;;
        --all)
            # Tail all relevant log files
            LOG_FILE="ominari_unified.log web_monitor.log paper_trades.log"
            shift
            ;;
        -h|--help)
            echo "Ominari Log Viewer"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -f, --file FILE     Log file to view (default: ominari_unified.log)"
            echo "  -n, --lines N       Show last N lines (disables follow mode)"
            echo "  --no-follow         Don't follow log updates"
            echo "  -g, --grep PATTERN  Filter logs by pattern"
            echo "  --trades           Show only trade-related logs"
            echo "  --errors           Show only error logs"
            echo "  --all              Show all log files"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                      # Follow main log"
            echo "  $0 --trades             # Follow trade logs only"
            echo "  $0 -n 100 --errors      # Show last 100 error lines"
            echo "  $0 --all --grep 'paper' # Search all logs for 'paper'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to colorize log lines
colorize_log() {
    while IFS= read -r line; do
        if echo "$line" | grep -qE "ERROR|CRITICAL|error|Error"; then
            echo -e "${RED}$line${NC}"
        elif echo "$line" | grep -qE "WARNING|WARN|warning|Warning"; then
            echo -e "${YELLOW}$line${NC}"
        elif echo "$line" | grep -qE "✅|SUCCESS|success|Success"; then
            echo -e "${GREEN}$line${NC}"
        elif echo "$line" | grep -qE "trade|order|fill|Trade|Order|Fill"; then
            echo -e "${PURPLE}$line${NC}"
        elif echo "$line" | grep -qE "INFO|info|Info"; then
            echo -e "${BLUE}$line${NC}"
        else
            echo "$line"
        fi
    done
}

# Header
echo -e "${GREEN}🚀 Ominari Log Viewer${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check if log files exist
for file in $LOG_FILE; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}Error: Log file '$file' not found${NC}"
        echo "Available log files:"
        ls -la *.log 2>/dev/null | grep -v "^total"
        exit 1
    fi
done

# Build tail command
if [ "$FOLLOW" = true ]; then
    TAIL_CMD="tail -f"
else
    TAIL_CMD="tail -n $LINES"
fi

# Execute based on filter
if [ -n "$FILTER" ]; then
    echo -e "Filter: ${YELLOW}$FILTER${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ "$FOLLOW" = true ]; then
        # For multiple files with grep and follow
        $TAIL_CMD $LOG_FILE | grep -E "$FILTER" --color=never | colorize_log
    else
        # For static display with grep
        for file in $LOG_FILE; do
            echo -e "\n${BLUE}=== $file ===${NC}"
            tail -n $LINES "$file" | grep -E "$FILTER" --color=never | colorize_log
        done
    fi
else
    # No filter
    if [ "$FOLLOW" = true ]; then
        $TAIL_CMD $LOG_FILE | colorize_log
    else
        for file in $LOG_FILE; do
            echo -e "\n${BLUE}=== $file ===${NC}"
            tail -n $LINES "$file" | colorize_log
        done
    fi
fi