#!/bin/bash
# Start Ominari V1 System

cd "$(dirname "$0")"

echo "=== Starting Ominari V1 Trading System ==="
echo "Configuration: API for data/quotes, blockchain for trades"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Run the v1 system
python run_ominari_v1.py