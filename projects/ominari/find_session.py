#!/usr/bin/env python3

import json
from paper_trading_sessions import PaperTradingSessionManager

session_manager = PaperTradingSessionManager()
sessions = session_manager.sessions.get('sessions', {})

print('Looking for session with 9 trades and $1,537.46 stake...')
for sid, session in sessions.items():
    trades = len(session.get('trades', []))
    total_stake = session.get('performance', {}).get('total_stake', 0)
    portfolio = session.get('portfolio_value', session.get('current_bankroll', 0))
    positions = len(session.get('positions', {}))
    
    # Check if this matches the Discord heartbeat data (9 bets, $1,537.46 stake)
    if 8 <= trades <= 10 or 1500 <= total_stake <= 1600:
        print(f'\n*** POTENTIAL MATCH: {sid} ***')
        print(f'  Trades: {trades}')
        print(f'  Total Stake: ${total_stake:.2f}')
        print(f'  Portfolio: ${portfolio:.2f}')
        print(f'  Positions: {positions}')
        print(f'  Status: {session.get("status", "unknown")}')
        print(f'  Created: {session.get("created_at", "unknown")}')

print(f'\nNote: Looking for session matching Discord heartbeat:')
print(f'  - 9 total bets')
print(f'  - $1,537.46 total staked')
print(f'  - $10,000 bankroll')