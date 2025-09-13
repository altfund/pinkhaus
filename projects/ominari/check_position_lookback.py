#!/usr/bin/env python3
"""Check how far back we need to look for paper trading positions."""

import json
from datetime import datetime, timezone
from pathlib import Path

# Load the paper trading sessions
sessions_file = Path("paper_trading_sessions.json")
if sessions_file.exists():
    with open(sessions_file) as f:
        data = json.load(f)
    
    print("PAPER TRADING SESSIONS ANALYSIS")
    print("=" * 80)
    
    # Check all sessions
    oldest_position = None
    newest_position = None
    positions_without_results = []
    
    for session_id, session_data in data.get("sessions", {}).items():
        print(f"\nSession: {session_id}")
        print(f"Status: {session_data.get('status')}")
        
        positions = session_data.get("positions", [])
        print(f"Total positions: {len(positions)}")
        
        # Check each position
        for pos in positions:
            if "entry_time" in pos:
                entry_time = datetime.fromisoformat(pos['entry_time'].replace('Z', '+00:00'))
                
                if oldest_position is None or entry_time < oldest_position[0]:
                    oldest_position = (entry_time, pos)
                if newest_position is None or entry_time > newest_position[0]:
                    newest_position = (entry_time, pos)
                
                # Check if closed without result
                if pos.get('status') == 'closed' and pos.get('final_pnl', 0) == 0:
                    positions_without_results.append((entry_time, pos))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if oldest_position:
        age = datetime.now(timezone.utc) - oldest_position[0]
        print(f"\nOldest position:")
        print(f"  Time: {oldest_position[0]}")
        print(f"  Age: {age.days} days, {age.seconds//3600} hours")
        print(f"  Market: {oldest_position[1].get('market_id', 'Unknown')}")
        print(f"  Bet: {oldest_position[1].get('bet_name', 'Unknown')}")
    
    if newest_position:
        age = datetime.now(timezone.utc) - newest_position[0]
        print(f"\nNewest position:")
        print(f"  Time: {newest_position[0]}")
        print(f"  Age: {age.days} days, {age.seconds//3600} hours")
        print(f"  Market: {newest_position[1].get('market_id', 'Unknown')}")
    
    print(f"\nPositions without results: {len(positions_without_results)}")
    
    # Show a few examples
    if positions_without_results:
        print("\nExamples of positions without results:")
        for entry_time, pos in positions_without_results[:5]:
            age = datetime.now(timezone.utc) - entry_time
            print(f"\n  Market: {pos.get('market_id', 'Unknown')}")
            print(f"  Bet: {pos.get('bet_name', 'Unknown')}")
            print(f"  Entry: {entry_time}")
            print(f"  Age: {age.days} days, {age.seconds//3600} hours")
    
    # Calculate required lookback
    if oldest_position:
        oldest_age = datetime.now(timezone.utc) - oldest_position[0]
        required_lookback_hours = int(oldest_age.total_seconds() / 3600) + 1
        
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        
        print(f"\n⚠️  Current lookback: 6 hours")
        print(f"⚠️  Required lookback: {required_lookback_hours} hours")
        
        if required_lookback_hours > 6:
            print(f"\n🔧 To catch all positions, run:")
            print(f"   python results_catchup_service.py --mode once --lookback {required_lookback_hours}")
            
            print(f"\n🔧 To update the automated scheduler for future runs:")
            print(f"   1. Edit run_everything.py and change the lookback from 6 to {min(required_lookback_hours, 48)}")
            print(f"   2. Edit ominari_unified.py and change the lookback from 6 to {min(required_lookback_hours, 48)}")
            print(f"   3. Restart the scheduler: systemctl --user restart altfund_scheduler.service")
else:
    print("No paper trading sessions file found!")