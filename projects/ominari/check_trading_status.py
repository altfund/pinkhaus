#!/usr/bin/env python3
"""Check trading status"""
import os
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999', 
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})

from paper_trading_postgres_integrated import PaperTradingSessionManager

# Create manager
manager = PaperTradingSessionManager()

# Get current session
current_session = manager.get_current_session()
print(f"Current session: {current_session}")

# Get session data  
if current_session:
    session = manager.get_session(current_session)
    print(f"Session data: {session}")
    
    # Get positions
    positions = manager.get_positions(current_session)
    open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
    
    print(f"\nTotal positions: {len(positions)}")
    print(f"Open positions: {len(open_positions)}")
    
    if open_positions:
        total_stake = sum(float(p['stake']) for p in open_positions) 
        print(f"Total stake: ${total_stake:,.2f}")
        print(f"Exposure: {total_stake / 10000 * 100:.1f}%")
        
        # Show last few trades
        print("\nLast 5 open positions:")
        for pos in open_positions[-5:]:
            print(f"  {pos['match_id']} - {pos['bet_on']} - ${pos['stake']}")