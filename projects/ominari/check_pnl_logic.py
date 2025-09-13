#!/usr/bin/env python3
"""Check P&L calculation logic for winning positions."""

from paper_trading_sessions import PaperTradingSessionManager

sm = PaperTradingSessionManager()
session = sm.get_current_session()

if session:
    closed = session.get('closed_positions', [])
    
    print("CHECKING P&L CALCULATION LOGIC")
    print("=" * 80)
    
    # Find some winning positions
    wins = [p for p in closed if p.get('result') == 'won'][:5]
    
    for p in wins:
        print(f"\nMarket: {p.get('market_name')}")
        print(f"  Outcome: {p.get('outcome')}")
        print(f"  Stake: ${p.get('total_stake', 0):.2f}")
        print(f"  Execution stake: ${p.get('execution_stake', 0):.2f}")
        print(f"  Avg odds: {p.get('avg_odds', 0):.2f}")
        print(f"  Final value: ${p.get('final_value', 0):.2f}")
        print(f"  P&L: ${p.get('pnl', 0):.2f}")
        print(f"  Result: {p.get('result')}")
        
        # Check the math
        stake = p.get('total_stake', 0)
        exec_stake = p.get('execution_stake', stake)
        odds = p.get('avg_odds', 0)
        final_value = p.get('final_value', 0)
        pnl = p.get('pnl', 0)
        
        print("\n  CALCULATIONS:")
        print(f"  Gross payout (stake × odds): ${stake:.2f} × {odds:.2f} = ${stake * odds:.2f}")
        print(f"  Final value stored: ${final_value:.2f}")
        print(f"  P&L stored: ${pnl:.2f}")
        print(f"  P&L = final_value - exec_stake: ${final_value:.2f} - ${exec_stake:.2f} = ${final_value - exec_stake:.2f}")
        
        # Check if final_value is gross payout
        if abs(final_value - (stake * odds)) < 0.01:
            print("  ✓ Final value is gross payout (stake × odds)")
        else:
            print("  ✗ Final value doesn't match gross payout")
            
        # Check if P&L is final_value - execution_stake  
        if abs(pnl - (final_value - exec_stake)) < 0.01:
            print("  ✓ P&L is final_value - execution_stake")
        else:
            print("  ✗ P&L doesn't match final_value - execution_stake")