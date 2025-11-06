#!/usr/bin/env python3
"""Manually place some test trades to verify system works"""

import os
from datetime import datetime, timezone

# Set PostgreSQL environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from paper_trading_postgres_integrated import PaperTradingSessionManager

# Initialize
session_manager = PaperTradingSessionManager()
session_id = session_manager.get_current_session()

if not session_id:
    print("No active session!")
    exit(1)

print(f"Using session: {session_id}")

# Get current bankroll
session = session_manager.get_session(session_id)
current_bankroll = float(session['current_bankroll'])
print(f"Current bankroll: ${current_bankroll:,.2f}")

# Create some test trades
test_trades = [
    {
        'match_id': 'test_match_001',
        'sport': 'Soccer',
        'home_team': 'Manchester United',
        'away_team': 'Liverpool FC',
        'bet_on': 'home',
        'odds': 2.20,
        'stake': 50.00,
        'edge': 0.05,
        'signal_name': 'test_signal',
        'signal_value': 0.48
    },
    {
        'match_id': 'test_match_002',
        'sport': 'Soccer', 
        'home_team': 'Real Madrid',
        'away_team': 'Barcelona FC',
        'bet_on': 'away',
        'odds': 2.80,
        'stake': 40.00,
        'edge': 0.08,
        'signal_name': 'test_signal',
        'signal_value': 0.38
    },
    {
        'match_id': 'test_match_003',
        'sport': 'Soccer',
        'home_team': 'Bayern Munich',
        'away_team': 'Borussia Dortmund',
        'bet_on': 'home',
        'odds': 1.85,
        'stake': 60.00,
        'edge': 0.03,
        'signal_name': 'test_signal',
        'signal_value': 0.56
    }
]

print(f"\nPlacing {len(test_trades)} test trades...")

# Record trades
for trade in test_trades:
    result = session_manager.record_bet(session_id, trade)
    if result:
        print(f"✅ Placed: {trade['home_team']} vs {trade['away_team']} - {trade['bet_on'].upper()} @ {trade['odds']} - ${trade['stake']}")
    else:
        print(f"❌ Failed to place trade")

# Check positions
positions = session_manager.get_positions(session_id)
open_positions = [p for p in positions if p['status'] == 'pending']

print(f"\n📊 Position Summary:")
print(f"   Total positions: {len(open_positions)}")
total_stake = sum(float(p['stake']) for p in open_positions)
print(f"   Total stake: ${total_stake:.2f}")
print(f"   Exposure: {total_stake/current_bankroll*100:.1f}%")

# Show positions
if open_positions:
    print(f"\n📈 Current Positions:")
    for i, pos in enumerate(open_positions[:5]):
        print(f"\n   {i+1}. {pos['home_team']} vs {pos['away_team']}")
        print(f"      Bet: {pos['bet_on'].upper()} @ {pos['odds']}")
        print(f"      Stake: ${pos['stake']}")
        print(f"      Placed: {pos['placed_at']}")

print("\n✅ Test trades placed successfully!")
print("The dashboard should now show these positions.")
print("Visit http://localhost:8888/ to view them.")