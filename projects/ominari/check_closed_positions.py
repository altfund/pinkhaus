#!/usr/bin/env python3
"""Check closed positions and their P&L."""

from paper_trading_sessions import PaperTradingSessionManager
import json

def check_closed_positions():
    """Check closed positions to understand settlement."""
    session_manager = PaperTradingSessionManager()
    current_session = session_manager.get_current_session()
    
    if not current_session:
        print("No active session found")
        return
    
    session_id = current_session['session_id']
    closed_positions = current_session.get('closed_positions', [])
    
    print(f"Session: {session_id}")
    print(f"Total Closed Positions: {len(closed_positions)}")
    print()
    
    # Calculate totals
    total_wins = 0
    total_losses = 0
    total_pnl = 0
    
    # Show last 10 closed positions
    print("Recent Closed Positions:")
    for pos in closed_positions[-10:]:
        result = pos.get('result', 'unknown')
        pnl = pos.get('pnl', 0)
        
        if result == 'won':
            total_wins += 1
        elif result == 'lost':
            total_losses += 1
        
        if pnl != 0:
            total_pnl += pnl
            
        print(f"{pos.get('market_name', 'Unknown')[:40]} - {pos.get('outcome')}")
        print(f"  Result: {result}")
        print(f"  Stake: ${pos.get('total_stake', 0):.2f}")
        print(f"  Odds: {pos.get('avg_odds', 0):.2f}")
        print(f"  P&L: ${pnl:.2f}")
        print(f"  Closed at: {pos.get('closed_at', 'Unknown')}")
        print()
    
    # Performance summary
    performance = session_manager.get_session_performance(session_id)
    
    print("\nSession Performance:")
    print(f"  Current Portfolio Value: ${performance['current_value']:.2f}")
    print(f"  Initial Bankroll: ${current_session['initial_bankroll']:.2f}")
    print(f"  Current Cash: ${current_session['current_bankroll']:.2f}")
    print(f"  Open Positions Value: ${performance['current_value'] - current_session['current_bankroll']:.2f}")
    print(f"  Total P&L: ${performance['total_pnl']:.2f}")
    print(f"  ROI: {performance['roi']:.1f}%")
    print(f"  Win Rate: {performance['win_rate']:.1%}")
    print(f"  Winning Trades: {performance['winning_trades']}")
    print(f"  Losing Trades: {performance['losing_trades']}")
    print(f"  Pending Trades: {performance['pending_trades']}")
    
    # Check if metrics match
    calculated_win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0
    print(f"\nVerification:")
    print(f"  Calculated P&L from closed: ${total_pnl:.2f}")
    print(f"  Reported Total P&L: ${performance['total_pnl']:.2f}")

if __name__ == "__main__":
    check_closed_positions()