#!/usr/bin/env python3
"""Test Discord notifications directly"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env
from load_env import load_dotenv
load_dotenv()

# Now import and test
from notifications.discord_notifier import discord_notifier

print(f"Discord enabled: {discord_notifier.enabled}")
print(f"Discord webhook URL: {discord_notifier.webhook_url[:50] if discord_notifier.webhook_url else 'None'}...")

if discord_notifier.enabled:
    print("\nTesting startup message...")
    result = discord_notifier.send_startup_message()
    print(f"Startup message result: {result}")
    
    print("\nTesting trade alert...")
    trade_data = {
        'type': 'NEW',
        'market': 'Test Market vs Demo Team',
        'outcome': 'home',
        'amount': 100.0,
        'odds': 2.5,
        'edge': 5.2,
        'bankroll': 10000.0
    }
    result = discord_notifier.send_trade_alert(trade_data)
    print(f"Trade alert result: {result}")
else:
    print("\n❌ Discord notifications not enabled!")