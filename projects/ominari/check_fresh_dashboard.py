#!/usr/bin/env python3
"""Check dashboard data for fresh session"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor

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
print(f"   Initial Bankroll: ${session.get('initial_bankroll', 0):,.2f}")
print(f"   Current Bankroll: ${session.get('current_bankroll', 0):,.2f}")
print(f"   Portfolio Value: ${session.get('portfolio_value', session.get('current_bankroll', 0)):,.2f}")

# Get positions
positions = session_manager.get_positions(session_id)
open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
total_stake = sum(float(p.get('stake', 0)) for p in open_positions)

print(f"\n📈 Positions:")
print(f"   Open Positions: {len(open_positions)}")
print(f"   Total Stake: ${total_stake:.2f}")
print(f"   Cash Available: ${session.get('current_bankroll', 10000) - total_stake:.2f}")

if open_positions:
    print(f"\n📋 Recent Positions:")
    for i, pos in enumerate(open_positions[:3]):
        print(f"   {i+1}. {pos.get('home_team', '?')} vs {pos.get('away_team', '?')}")
        print(f"      Bet: {pos.get('bet_on', '?').upper()} @ {pos.get('odds', 0):.2f} - ${pos.get('stake', 0):.2f}")

# Direct query to check snapshot
conn = psycopg2.connect(
    host=os.environ['PG_HOST'],
    port=os.environ['PG_PORT'],
    user=os.environ['PG_USER'],
    password=os.environ['PG_PASSWORD'],
    database=os.environ['PG_DB'],
    cursor_factory=RealDictCursor
)

try:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT * FROM paper_trading_snapshots 
            WHERE session_id = %s 
            ORDER BY snapshot_time DESC 
            LIMIT 1
        """, (session_id,))
        
        snapshot = cur.fetchone()
        if snapshot:
            print(f"\n📸 Latest Snapshot:")
            print(f"   Cash Balance: ${snapshot['cash_balance']:.2f}")
            print(f"   Positions Value: ${snapshot['positions_value']:.2f}")
            print(f"   Portfolio Value: ${snapshot['portfolio_value']:.2f}")
finally:
    conn.close()