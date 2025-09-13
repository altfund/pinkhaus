#!/usr/bin/env python3
"""Check age of specific paper trading positions."""

import json
from pathlib import Path
from datetime import datetime, timezone

# Load the latest paper trading session
session_dir = Path('paper_trading_sessions')
if session_dir.exists():
    latest_session = sorted(session_dir.glob('*/session_data.json'))[-1]
    print(f'Loading session: {latest_session}')
    
    with open(latest_session) as f:
        data = json.load(f)
    
    # Find the specific positions mentioned by user
    print('\nCHECKING SPECIFIC CLOSED POSITIONS:')
    print('=' * 80)
    
    search_patterns = ['Kosovo', 'Sweden', 'Greece', 'Denmark', 'Ghana', 'Mali', 
                      'Libya', 'Eswatini', 'Vila Nova', 'Athletic Club']
    
    found_positions = []
    for pos in data['positions']:
        if pos['status'] == 'closed':
            # Check if any search pattern is in the market_id or bet_name
            for pattern in search_patterns:
                if pattern in pos.get('market_id', '') or pattern in pos.get('bet_name', ''):
                    found_positions.append(pos)
                    break
    
    for pos in found_positions[:10]:  # Show first 10
        entry_time = datetime.fromisoformat(pos['entry_time'].replace('Z', '+00:00'))
        age = datetime.now(timezone.utc) - entry_time
        
        print(f"\nMarket ID: {pos['market_id']}")
        print(f"Bet: {pos['bet_name']}")
        print(f"Status: {pos['status']}") 
        print(f"Entry Time: {entry_time}")
        print(f"Age: {age.days} days, {age.seconds//3600} hours ago")
        print(f"Result: {pos.get('result', 'NOT SET')}")
        print(f"Final P&L: ${pos.get('final_pnl', 0):.2f}")
        
    # Check session age
    session_start = datetime.fromisoformat(data['session_start_time'].replace('Z', '+00:00'))
    session_age = datetime.now(timezone.utc) - session_start
    print(f"\n\nSESSION INFO:")
    print(f"Session started: {session_start}")
    print(f"Session age: {session_age.days} days, {session_age.seconds//3600} hours")
    total_positions = len(data['positions'])
    closed_count = sum(1 for p in data['positions'] if p['status'] == 'closed')
    print(f"Total positions: {total_positions}")
    print(f"Closed positions: {closed_count}")
    
    # Check oldest position
    oldest_time = None
    for pos in data['positions']:
        entry_time = datetime.fromisoformat(pos['entry_time'].replace('Z', '+00:00'))
        if oldest_time is None or entry_time < oldest_time:
            oldest_time = entry_time
    
    if oldest_time:
        oldest_age = datetime.now(timezone.utc) - oldest_time
        print(f"\nOldest position: {oldest_time}")
        print(f"Oldest position age: {oldest_age.days} days, {oldest_age.seconds//3600} hours")
        
        # Recommendation based on age
        if oldest_age.total_seconds() > 6 * 3600:  # Older than 6 hours
            print(f"\n⚠️  IMPORTANT: Some positions are older than the 6-hour lookback window!")
            print(f"Recommended action: Run a manual catch-up with extended lookback:")
            print(f"  python results_catchup_service.py --mode once --lookback {int(oldest_age.total_seconds() / 3600) + 1}")