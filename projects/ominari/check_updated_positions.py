#!/usr/bin/env python3
"""Check if positions were updated with results."""

from paper_trading_sessions import PaperTradingSessionManager

session_manager = PaperTradingSessionManager()
current_session = session_manager.get_current_session()

if current_session:
    closed = current_session.get('closed_positions', [])
    print(f'Total closed positions: {len(closed)}')
    
    # Check how many have results now
    with_results = sum(1 for pos in closed if pos.get('result') in ['won', 'lost'])
    with_pnl = sum(1 for pos in closed if pos.get('pnl', 0) != 0)
    
    print(f'Positions with results: {with_results}')
    print(f'Positions with P&L: {with_pnl}')
    
    # Show a few examples
    print('\nFirst 5 positions:')
    for i, pos in enumerate(closed[:5]):
        print(f'\n{i+1}. {pos.get("market_name", "Unknown")}')
        print(f'   Outcome: {pos.get("outcome")}')
        print(f'   Result: {pos.get("result", "MISSING")}')
        print(f'   P&L: ${pos.get("pnl", 0):.2f}')
        
    # Check positions that should match the user's examples
    print('\n' + '=' * 60)
    print('CHECKING SPECIFIC MATCHES MENTIONED BY USER:')
    print('=' * 60)
    
    search_patterns = [
        'Kosovo', 'Sweden', 
        'Greece', 'Denmark',
        'Ghana', 'Mali',
        'Libya', 'Eswatini',
        'Vila Nova', 'Athletic Club',
        'Grêmio Novorizontino', 'Atlético Goianiense'
    ]
    
    for pattern in search_patterns:
        found = False
        for pos in closed:
            market_name = pos.get('market_name', '')
            if pattern in market_name:
                print(f'\nFound: {market_name}')
                print(f'  Result: {pos.get("result", "MISSING")}')
                print(f'  P&L: ${pos.get("pnl", 0):.2f}')
                found = True
                break
        if not found:
            print(f'\nNot found: {pattern}')
else:
    print("No active session found!")