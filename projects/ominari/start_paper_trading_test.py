#!/usr/bin/env python3
"""
Start a paper trading session for testing
"""
import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from paper_trading_postgres_integrated import PaperTradingSessionManager
from datetime import datetime, timezone
import json

def main():
    """Start a new paper trading session"""
    
    # Initialize session manager
    session_manager = PaperTradingSessionManager()
    
    # Check if there's an active session
    if session_manager.current_session_id:
        print(f"Active session found: {session_manager.current_session_id}")
        session = session_manager.get_session(session_manager.current_session_id)
        print(f"  Created: {session['created_at']}")
        print(f"  Bankroll: ${session['current_bankroll']:,.2f}")
        print(f"  Total Bets: {session['total_bets']}")
    else:
        print("No active session found")
    
    # Create a new paper trading session
    print("\nCreating new paper trading session...")
    
    session_id = session_manager.create_session(
        initial_bankroll=10000.0,
        session_name="Edge-Based Paper Trading Test"
    )
    
    print(f"\nNew session created: {session_id}")
    
    # Get the full session details
    new_session = session_manager.get_session(session_id)
    if new_session:
        print(f"  Name: {new_session.get('session_name')}")
        print(f"  Initial Bankroll: ${new_session['initial_bankroll']:,.2f}")
        print(f"  Current Bankroll: ${new_session['current_bankroll']:,.2f}")
        print(f"  Created: {new_session['created_at']}")
    
    # Set as current session
    session_manager.set_current_session(session_id)
    print(f"\nSession {session_id} set as current")
    
    # Verify it's saved
    sessions = session_manager.get_all_sessions()
    print(f"\nTotal sessions in database: {len(sessions)}")
    
    # Show recent sessions
    print("\nRecent sessions:")
    for s in sessions[:5]:
        status = 'CURRENT' if s['session_id'] == session_manager.current_session_id else ''
        name = s.get('session_name', 'Unnamed')
        print(f"  {s['session_id']}: {name} - ${s['current_bankroll']:,.2f} - {s['created_at']} {status}")

if __name__ == "__main__":
    main()