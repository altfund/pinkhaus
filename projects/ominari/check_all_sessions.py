#!/usr/bin/env python3
"""Check all sessions to find the one with settled trades."""

from paper_trading_sessions import PaperTradingSessionManager
import json

def check_all_sessions():
    """Check all sessions for the one with settled trades."""
    session_manager = PaperTradingSessionManager()
    
    print("=== All Sessions ===")
    for session_id, session in session_manager.sessions.get("sessions", {}).items():
        performance = session_manager.get_session_performance(session_id)
        
        print(f"\nSession: {session_id}")
        print(f"  Status: {session.get('status', 'unknown')}")
        print(f"  Portfolio Value: ${session.get('portfolio_value', 0):.2f}")
        print(f"  Total P&L: ${performance.get('total_pnl', 0):.2f}")
        print(f"  Win Rate: {performance.get('win_rate', 0):.1%}")
        print(f"  Wins/Losses: {performance.get('winning_trades', 0)}/{performance.get('losing_trades', 0)}")
        print(f"  Open Positions: {len(session.get('positions', {}))}")
        print(f"  Closed Positions: {len(session.get('closed_positions', []))}")
        
        # Check if this is the session we fixed
        if performance.get('winning_trades', 0) == 7 and performance.get('losing_trades', 0) == 39:
            print("  *** This is the session with settled trades! ***")
            
            # Make it the current session
            session_manager.sessions["current_session"] = session_id
            session_manager._save_sessions()
            print("  *** Set as current session ***")

if __name__ == "__main__":
    check_all_sessions()