#!/usr/bin/env python3
"""View paper trading sessions and quotes."""

from paper_trading_sessions import PaperTradingSessionManager
from paper_trading_quotes import QuoteRecorder
import json

def main():
    print("=" * 60)
    print("PAPER TRADING SESSIONS")
    print("=" * 60)
    
    # Load session manager
    session_mgr = PaperTradingSessionManager()
    
    # Show all sessions
    for session_id, session in session_mgr.sessions.get("sessions", {}).items():
        perf = session_mgr.get_session_performance(session_id)
        
        print(f"\nSession: {session_id}")
        print(f"  Name: {session.get('session_name', 'Unnamed')}")
        print(f"  Created: {session.get('created_at', 'Unknown')}")
        print(f"  Status: {session.get('status', 'Unknown')}")
        print(f"  Portfolio Value: ${session.get('portfolio_value', 0):.2f}")
        print(f"  Cash Available: ${session.get('current_bankroll', 0):.2f}")
        print(f"  ROI: {perf.get('roi', 0):.2f}%")
        print(f"  Total Trades: {perf.get('total_trades', 0)}")
        print(f"  Open Positions: {len(session.get('positions', {}))}")
        print(f"  Win Rate: {perf.get('win_rate', 0)*100:.1f}%")
        print(f"  Total P&L: ${perf.get('total_pnl', 0):.2f}")
    
    # Show quotes
    print("\n" + "=" * 60)
    print("RECENT QUOTE RECORDINGS")
    print("=" * 60)
    
    quote_rec = QuoteRecorder()
    recent = quote_rec.get_recent_sessions(limit=3)
    
    for session in recent:
        print(f"\nQuotes for session: {session.get('session_id')}")
        print(f"  Timestamp: {session.get('timestamp')}")
        print(f"  Trades recorded: {len(session.get('trades', []))}")
        
        # Show first 3 trades
        for i, trade in enumerate(session.get('trades', [])[:3]):
            print(f"    {i+1}. {trade.get('market_name')} - {trade.get('outcome')} @ {trade.get('execution_odds', 0):.2f}")

if __name__ == "__main__":
    main()