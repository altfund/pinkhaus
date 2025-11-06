#!/usr/bin/env python3
"""Test dashboard position values"""

import requests
import json

# Test if web monitor is running
try:
    response = requests.get('http://localhost:8888/', timeout=2)
    print(f"✅ Dashboard is running: {response.status_code}")
except:
    print("❌ Dashboard is not accessible")
    exit(1)

# Get session data directly from database
import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from paper_trading_postgres_integrated import PaperTradingSessionManager

session_manager = PaperTradingSessionManager()
session_id = session_manager.get_current_session()

if session_id:
    positions = session_manager.get_positions(session_id)
    open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
    total_stake = sum(float(p.get('stake', 0)) for p in open_positions)
    
    print(f"\n📊 Database Values:")
    print(f"   Session ID: {session_id}")
    print(f"   Open Positions: {len(open_positions)}")
    print(f"   Total Stake (Position Value): ${total_stake:.2f}")
    
    if open_positions:
        print(f"\n📈 Sample Positions:")
        for i, pos in enumerate(open_positions[:3]):
            print(f"   {i+1}. {pos.get('bet_on', '?').upper()} @ {pos.get('odds', 0):.2f} - ${pos.get('stake', 0):.2f}")

print("\n🌐 Dashboard should show:")
print(f"   Position Value: ${total_stake:.2f} (in top left corner)")
print(f"   This is the sum of all open position stakes")
print("\nIf the dashboard shows $0, there may be a display issue.")