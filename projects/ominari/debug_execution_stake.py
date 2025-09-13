#!/usr/bin/env python3
"""Debug execution stake values."""

from paper_trading_sessions import PaperTradingSessionManager

sm = PaperTradingSessionManager()
session = sm.get_current_session()

if session:
    print("DEBUGGING EXECUTION STAKE")
    print("=" * 80)
    
    # Check a few positions
    closed = session.get('closed_positions', [])[:10]
    
    for pos in closed:
        fee_info = pos.get('fee_info', {})
        stake = pos.get('total_stake', 0)
        exec_stake = pos.get('execution_stake', 0)
        fee_amount = fee_info.get('fee_amount', 0)
        
        print(f"\n{pos.get('market_name')[:40]} - {pos.get('outcome')}")
        print(f"  Stake: ${stake:.2f}")
        print(f"  Fee amount: ${fee_amount:.2f}")
        print(f"  Fee info exec stake: ${fee_info.get('execution_stake', 0):.2f}")
        print(f"  Position exec stake: ${exec_stake:.2f}")
        print(f"  Expected (stake + fee): ${stake + fee_amount:.2f}")
        
        # Check if execution_stake includes fees
        if abs(exec_stake - (stake + fee_amount)) < 0.01:
            print("  ✓ Execution stake = stake + fees")
        elif abs(exec_stake - stake) < 0.01:
            print("  ✗ Execution stake = stake only (missing fees)")
        else:
            print("  ⚠️  Execution stake doesn't match expected values")
    
    # Check bankroll deduction
    print("\n" + "=" * 80)
    print("BANKROLL CHECK:")
    initial = session.get('initial_bankroll', 10000)
    current = session.get('current_bankroll', 0)
    
    # Calculate expected current bankroll
    total_exec_stake = sum(p.get('execution_stake', p.get('total_stake', 0)) 
                          for p in session.get('positions', {}).values())
    total_returned = sum(p.get('final_value', 0) for p in closed if p.get('result') == 'won')
    
    print(f"Initial bankroll: ${initial:.2f}")
    print(f"Current bankroll: ${current:.2f}")
    print(f"Open position stakes: ${total_exec_stake:.2f}")
    print(f"Returned from wins: ${total_returned:.2f}")
    
    # Check trades for actual deductions
    trades = session.get('trades', [])[:10]
    print(f"\nSample trades (first 10):")
    for i, trade in enumerate(trades):
        stake = trade.get('stake', 0)
        fee_info = trade.get('fee_info', {})
        fee_amount = fee_info.get('fee_amount', 0)
        exec_stake = fee_info.get('execution_stake', 0)
        
        print(f"\n{i+1}. {trade.get('market_name', 'Unknown')[:30]}")
        print(f"   Stake: ${stake:.2f}")
        print(f"   Fee: ${fee_amount:.2f}")
        print(f"   Execution stake in fee_info: ${exec_stake:.2f}")