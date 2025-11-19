#!/usr/bin/env python3
import os
os.environ['PG_PORT'] = '5999'

from paper_trading_sessions import PaperTradingSessionManager
from datetime import datetime, timezone

# Check current paper trading status
manager = PaperTradingSessionManager()

print('🔍 Paper Trading Portfolio Analysis:')
print()

# Check if there are any active sessions
try:
    # Try to load sessions
    sessions_data = manager._load_sessions()
    print(f'Sessions file exists: {bool(sessions_data)}')
    
    if sessions_data and 'sessions' in sessions_data:
        print(f'Active sessions: {len(sessions_data["sessions"])}')
        
        for session_id, session in sessions_data['sessions'].items():
            print(f'\nSession: {session_id}')
            print(f'  Status: {session.get("status", "unknown")}')
            print(f'  Initial bankroll: ${session.get("initial_bankroll", 0):,.2f}')
            print(f'  Current bankroll: ${session.get("current_bankroll", 0):,.2f}')
            print(f'  Portfolio value: ${session.get("portfolio_value", 0):,.2f}')
            print(f'  Open positions: {len(session.get("positions", {}))}')
            print(f'  Total P&L: ${session.get("performance", {}).get("total_pnl", 0):,.2f}')
            
            # Show positions
            positions = session.get('positions', {})
            if positions:
                print(f'  Positions:')
                for pos_key, pos in positions.items():
                    print(f'    {pos_key}: ${pos.get("total_stake", 0):.2f} @ {pos.get("avg_odds", 0):.2f}')
            else:
                print(f'  No open positions')
    else:
        print('No active trading sessions found')
        
except Exception as e:
    print(f'Error accessing paper trading: {e}')
    
    # Check if we can create a session
    try:
        test_session = manager.create_session('test_portfolio', initial_bankroll=10000)
        print(f'\nTest session created: {test_session}')
    except Exception as e2:
        print(f'Error creating test session: {e2}')