#!/usr/bin/env python3

import json
from paper_trading_sessions import PaperTradingSessionManager

session_manager = PaperTradingSessionManager()
sessions = session_manager.sessions.get('sessions', {})

print('All sessions and their status:')
for sid, session in sessions.items():
    status = session.get('status', 'unknown')
    created = session.get('created_at', 'unknown')
    portfolio = session.get('portfolio_value', session.get('current_bankroll', 0))
    trades = len(session.get('trades', []))
    positions = len(session.get('positions', {}))
    print(f'{sid}: status={status}, created={created[:10]}, portfolio=${portfolio:.0f}, trades={trades}, positions={positions}')

print(f'\nCurrent session ID: {session_manager.current_session_id}')
current = session_manager.get_current_session()
if current:
    print(f'Current session portfolio: ${current.get("portfolio_value", current.get("current_bankroll", 0)):.2f}')