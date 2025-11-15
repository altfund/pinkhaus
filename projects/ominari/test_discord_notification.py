#!/usr/bin/env python3
"""Test Discord notification directly"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
from load_env import load_dotenv
load_dotenv()

from notifications.discord_notifier import discord_notifier

# Test notification
print(f"Discord webhook configured: {discord_notifier.webhook_url is not None}")
print(f"Discord enabled: {discord_notifier.enabled}")

if discord_notifier.enabled:
    # Send test notification
    result = discord_notifier.send_trade_alert({
        'type': 'BUY',
        'market': 'TEST: Portland Thorns FC vs Houston Dash (ARBITRAGE)',
        'outcome': 'home',
        'amount': 100.0,
        'odds': 2.3,
        'edge': 4.82,
        'kelly_pct': 1.0,
        'bankroll': 10000.0
    })
    
    if result:
        print("✅ Discord notification sent successfully!")
    else:
        print("❌ Failed to send Discord notification")
else:
    print("❌ Discord notifications not enabled - check DISCORD_WEBHOOK_URL in .env")