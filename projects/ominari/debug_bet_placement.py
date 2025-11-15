#!/usr/bin/env python3
"""Debug bet placement issue"""

import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable DEBUG logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
from load_env import load_dotenv
load_dotenv()

# Set up environment
os.environ['PG_PORT'] = '5999'

from config.bankroll_config import BankrollConfig
from paper_trading_live import LivePaperTrader

# Get current bankroll
bc = BankrollConfig()
print(f'Current bankroll: ${bc.get_current_bankroll()}')
print(f'Initial bankroll: ${bc.initial_bankroll}')

# Test Kelly calculation
trader = LivePaperTrader()

# Liverpool UY market with 27.29% edge
fair_prob = 0.303
odds = 3.3
bankroll = bc.get_current_bankroll()

print(f"\nCalculating bet for Liverpool UY:")
print(f"Fair probability: {fair_prob:.3f}")
print(f"Decimal odds: {odds}")
print(f"Current bankroll: ${bankroll:.2f}")

bet_size = trader.calculate_kelly_bet(fair_prob, odds, bankroll)
print(f'\nKelly bet size: ${bet_size:.2f}')
print(f'Min bet: ${trader.min_bet}')
print(f'Max bet %: {trader.max_bet_pct * 100}%')
print(f'Max bet $: ${bankroll * trader.max_bet_pct:.2f}')

# Check risk limits
risk_limits = bc.get_risk_limits()
print(f"\nRisk limits:")
for key, value in risk_limits.items():
    print(f"  {key}: {value}")