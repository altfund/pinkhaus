#!/usr/bin/env python3
"""Check what values the dashboard should be showing"""

import os
import json

# Set PostgreSQL environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from paper_trading_postgres_integrated import PaperTradingSessionManager

# Get session data
session_manager = PaperTradingSessionManager()
session_id = session_manager.get_current_session()

if not session_id:
    print("❌ No active session found!")
    exit(1)

print(f"📊 Active Session: {session_id}")

# Get session details
session = session_manager.get_session(session_id)
print(f"   Current Bankroll: ${session['current_bankroll']:,.2f}")
print(f"   Portfolio Value: ${session.get('portfolio_value', session['current_bankroll']):,.2f}")

# Get positions
positions = session_manager.get_positions(session_id)
open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
total_stake = sum(float(p.get('stake', 0)) for p in open_positions)

print(f"\n📈 Positions:")
print(f"   Open Positions: {len(open_positions)}")
print(f"   Total Stake: ${total_stake:.2f}")
print(f"   Exposure %: {(total_stake / session['current_bankroll'] * 100):.1f}%")

print(f"\n🌐 Dashboard Values (Top Left Corner):")
print(f"   Portfolio Value: ${session.get('portfolio_value', session['current_bankroll']):,.2f}")
print(f"   Cash Available: ${session['current_bankroll']:,.2f}")
print(f"   Positions Value: ${total_stake:.2f} ← This should show in dashboard")
print(f"   Exposure: {(total_stake / session['current_bankroll'] * 100):.1f}%")

# Show first few positions
if open_positions:
    print(f"\n📋 Sample Positions:")
    for i, pos in enumerate(open_positions[:3]):
        print(f"   {i+1}. {pos.get('bet_on', '?').upper()} {pos.get('home_team', '?')} vs {pos.get('away_team', '?')}")
        print(f"      Odds: {pos.get('odds', 0):.2f} | Stake: ${pos.get('stake', 0):.2f}")

print(f"\n✅ If dashboard shows $0 for Positions Value, refresh the page (F5)")
print(f"   The WebSocket should update it within 3 seconds")