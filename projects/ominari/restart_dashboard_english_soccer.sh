#!/bin/bash
# Restart dashboard with filters for English and International soccer

echo "=== Restarting Dashboard with English & International Soccer Focus ==="

# Kill existing dashboard
echo "Stopping current dashboard..."
pkill -f "web_dashboard_real_odds.py"
sleep 2

# Set environment variables for filtering
export ALLOWED_SPORTS="Soccer"
export ALLOWED_NATIONS="England,International,Europe"
export ALLOWED_LEAGUES=""  # Leave empty to show all leagues for selected nations

# Also set database environment
export PG_HOST='localhost'
export PG_PORT='5999'
export PG_USER='ominari_user'
export PG_PASSWORD='ominari_2025_secure'
export PG_DB='ominari_production'
export USE_POSTGRESQL='1'

echo ""
echo "Starting dashboard with filters:"
echo "  Sports: Soccer only"
echo "  Nations: England, International, Europe"
echo "  Leagues: All leagues from selected nations"
echo ""

# Start dashboard in background
# Try to use venv if available, otherwise use uv
if [ -f ".venv/bin/python3" ]; then
    echo "Using virtual environment..."
    nohup .venv/bin/python3 web_dashboard_real_odds.py > real_odds_dashboard.out 2>&1 &
else
    echo "Using uv to run..."
    nohup uv run --no-project python3 web_dashboard_real_odds.py > real_odds_dashboard.out 2>&1 &
fi

echo "Dashboard starting on http://localhost:8888"
echo "Log file: real_odds_dashboard.out"
echo ""
echo "To view different nations, restart with:"
echo "  export ALLOWED_NATIONS='Spain,Italy,Germany'"
echo "  ./restart_dashboard_english_soccer.sh"