#!/usr/bin/env python3
"""Simple check for settlement status"""

import os
import json
from datetime import datetime, timezone

# First check the session file
session_file = 'current_session.json'
if os.path.exists(session_file):
    with open(session_file, 'r') as f:
        session_data = json.load(f)
        print(f"📊 Session ID: {session_data.get('session_id', 'Unknown')}")
        print(f"   Current Bankroll: ${session_data.get('current_bankroll', 10000):,.2f}")
        print(f"   Portfolio Value: ${session_data.get('portfolio_value', 10000):,.2f}")

# Check positions file
positions_file = 'positions.json'
if os.path.exists(positions_file):
    with open(positions_file, 'r') as f:
        positions = json.load(f)
        
    now = datetime.now(timezone.utc)
    
    # Analyze positions
    pending = [p for p in positions if p.get('status') == 'pending']
    won = [p for p in positions if p.get('status') == 'won']
    lost = [p for p in positions if p.get('status') == 'lost']
    
    print(f"\n📈 Position Summary:")
    print(f"   Total: {len(positions)}")
    print(f"   Pending: {len(pending)}")
    print(f"   Won: {len(won)}")
    print(f"   Lost: {len(lost)}")
    
    if pending:
        # Check which should have finished
        expired = []
        for p in pending:
            if 'maturity_date' in p:
                mat_str = p['maturity_date']
                if mat_str:
                    try:
                        mat_date = datetime.fromisoformat(mat_str.replace('Z', '+00:00'))
                        if mat_date < now:
                            expired.append(p)
                    except:
                        pass
        
        print(f"\n⏰ Expired Positions (should be settled): {len(expired)}")
        
        if expired:
            print("\n🚨 Sample expired positions:")
            for i, p in enumerate(expired[:3]):
                print(f"\n   {i+1}. {p.get('home_team', '?')} vs {p.get('away_team', '?')}")
                print(f"      Bet: {p.get('bet_on', '?').upper()} @ {p.get('odds', 0):.2f}")
                print(f"      Stake: ${p.get('stake', 0):.2f}")
                print(f"      Match date: {p.get('maturity_date', '?')}")
                print(f"      Placed: {p.get('placed_at', '?')}")

# Check markets file
markets_file = 'markets.json'
if os.path.exists(markets_file):
    with open(markets_file, 'r') as f:
        markets = json.load(f)
    
    # Check if any are marked as finished
    finished = [m for m in markets if m.get('is_finished', False)]
    print(f"\n🏁 Finished Markets: {len(finished)} out of {len(markets)}")
    
    if not finished:
        print("   ⚠️  No markets are marked as finished!")
        print("   This suggests match results aren't being updated")

print("\n💡 Diagnosis:")
if 'pending' in locals() and len(pending) > 0 and 'won' in locals() and len(won) == 0 and 'lost' in locals() and len(lost) == 0:
    print("   ❌ Settlement process is NOT running")
    print("   All positions remain pending, none have been settled")
    print("\n🔧 Solution:")
    print("   1. Need to fetch match results from Overtime API")
    print("   2. Update markets with is_finished = true and scores")
    print("   3. Run settlement logic to update position statuses")
    print("   4. Update session P&L based on wins/losses")