#!/usr/bin/env python3
"""
Test Discord notifications for paper trading
"""

import os
import sys
import json
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from notifications.discord_notifier import DiscordNotifier

def test_discord_notifications():
    """Test various Discord notification types"""
    
    print("🔔 Testing Discord Notifications")
    print("=" * 50)
    
    # Check if webhook is configured
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    config_path = 'config/discord_config.json'
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            webhook_url = config.get('webhook_url') or webhook_url
    
    if not webhook_url:
        print("❌ No Discord webhook configured!")
        print("\nTo configure Discord:")
        print("  ./scripts/setup_discord.sh")
        return False
        
    print(f"✅ Discord webhook configured")
    print(f"   URL: {webhook_url[:50]}...")
    
    # Initialize notifier
    notifier = DiscordNotifier(webhook_url)
    
    # Test startup message
    print("\n📤 Sending startup notification...")
    if notifier.send_startup_message():
        print("✅ Startup notification sent!")
    else:
        print("❌ Failed to send startup notification")
        return False
    
    # Test trade alert
    print("\n📤 Sending sample paper trade notification...")
    sample_trade = {
        'type': 'NEW',
        'market': 'Arsenal vs Chelsea',
        'outcome': 'home',
        'amount': 150.00,
        'odds': 2.45,
        'edge': 5.2,
        'bankroll': 9850.00,
        'liquidity_info': 'Sufficient liquidity ($5,000 available)'
    }
    
    if notifier.send_trade_alert(sample_trade):
        print("✅ Trade notification sent!")
    else:
        print("❌ Failed to send trade notification")
        return False
    
    # Test portfolio update
    print("\n📤 Sending portfolio update...")
    portfolio_data = {
        'bankroll': 10250.00,
        'initial_bankroll': 10000.00,
        'total_bets': 15,
        'won': 8,
        'lost': 7,
        'pending': 0,
        'roi': 2.5,
        'win_rate': 53.33
    }
    
    if notifier.send_daily_summary(
        total_trades=15,
        winning_trades=8,
        total_pnl=250.00,
        current_bankroll=10250.00,
        roi=2.5,
        best_trade={'market': 'Liverpool vs Man City', 'pnl': 180.00},
        worst_trade={'market': 'Tottenham vs Newcastle', 'pnl': -75.00}
    ):
        print("✅ Portfolio update sent!")
    else:
        print("❌ Failed to send portfolio update")
        return False
    
    # Test market alert
    print("\n📤 Sending market opportunity alert...")
    if notifier.send_market_alert('high_opportunity', {
        'count': 3,
        'best_edge': 8.5,
        'markets': ['Real Madrid vs Barcelona', 'Bayern vs Dortmund', 'PSG vs Marseille']
    }):
        print("✅ Market alert sent!")
    else:
        print("❌ Failed to send market alert")
        return False
    
    print("\n🎉 All Discord notifications working!")
    print("\nYou should see 4 messages in your Discord channel:")
    print("  1. 🚀 Bot startup")
    print("  2. 📊 New paper trade")
    print("  3. 📈 Daily portfolio summary")
    print("  4. 🔥 High opportunity markets")
    
    return True

if __name__ == "__main__":
    success = test_discord_notifications()
    sys.exit(0 if success else 1)