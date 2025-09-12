#!/usr/bin/env python3
"""Analyze fees and P&L calculations."""

from paper_trading_sessions import PaperTradingSessionManager
import json

sm = PaperTradingSessionManager()
session = sm.get_current_session()

if session:
    print("ANALYZING FEES AND P&L")
    print("=" * 80)
    
    # Check a few closed positions
    closed = session.get('closed_positions', [])
    
    total_stake = 0
    total_execution_stake = 0
    total_fees = 0
    total_pnl = 0
    wins = 0
    losses = 0
    
    print("\nSample positions with fee details:")
    print("-" * 80)
    
    for i, pos in enumerate(closed[:5]):
        fee_info = pos.get('fee_info', {})
        stake = pos.get('total_stake', 0)
        execution_stake = pos.get('execution_stake', stake)
        fee_amount = fee_info.get('fee_amount', 0)
        pnl = pos.get('pnl', 0)
        result = pos.get('result', 'unknown')
        
        total_stake += stake
        total_execution_stake += execution_stake
        total_fees += fee_amount
        total_pnl += pnl
        
        if result == 'won':
            wins += 1
        elif result == 'lost':
            losses += 1
        
        print(f"\n{i+1}. {pos.get('market_name', 'Unknown')} - {pos.get('outcome')}")
        print(f"   Result: {result}")
        print(f"   Stake: ${stake:.2f}")
        print(f"   Fees: ${fee_amount:.2f} ({fee_info.get('total_fee_pct', 0)*100:.1f}%)")
        print(f"   Execution stake: ${execution_stake:.2f}")
        print(f"   Odds: {pos.get('avg_odds', 0):.2f}")
        
        if result == 'won':
            gross_payout = stake * pos.get('avg_odds', 0)
            print(f"   Gross payout: ${gross_payout:.2f}")
            print(f"   Net P&L: ${pnl:.2f}")
            print(f"   CHECK: Gross payout - Execution stake = ${gross_payout - execution_stake:.2f}")
        else:
            print(f"   Lost: -${execution_stake:.2f}")
    
    print("\n" + "=" * 80)
    print("TOTALS FOR ALL CLOSED POSITIONS:")
    print("=" * 80)
    
    # Calculate for all positions
    all_stake = sum(p.get('total_stake', 0) for p in closed)
    all_exec_stake = sum(p.get('execution_stake', p.get('total_stake', 0)) for p in closed)
    all_fees = sum(p.get('fee_info', {}).get('fee_amount', 0) for p in closed)
    all_pnl = sum(p.get('pnl', 0) for p in closed)
    
    print(f"Total positions: {len(closed)}")
    print(f"Total stake: ${all_stake:.2f}")
    print(f"Total fees: ${all_fees:.2f}")
    print(f"Total execution stake: ${all_exec_stake:.2f}")
    print(f"Total P&L: ${all_pnl:.2f}")
    
    # Check portfolio value
    print("\n" + "=" * 80)
    print("PORTFOLIO ROLLUP CHECK:")
    print("=" * 80)
    
    initial_bankroll = session.get('initial_bankroll', 10000)
    current_bankroll = session.get('current_bankroll', 0)
    portfolio_value = session.get('portfolio_value', 0)
    
    # Calculate what portfolio value should be
    open_positions = session.get('positions', {})
    open_stake = sum(p.get('total_stake', 0) for p in open_positions.values())
    open_exec_stake = sum(p.get('execution_stake', p.get('total_stake', 0)) for p in open_positions.values())
    
    print(f"Initial bankroll: ${initial_bankroll:.2f}")
    print(f"Current cash: ${current_bankroll:.2f}")
    print(f"Portfolio value (stored): ${portfolio_value:.2f}")
    
    print(f"\nOpen positions: {len(open_positions)}")
    print(f"Open stake: ${open_stake:.2f}")
    print(f"Open execution stake: ${open_exec_stake:.2f}")
    
    # Expected calculation
    expected_portfolio = current_bankroll + open_exec_stake + all_pnl
    print(f"\nExpected portfolio value: ${expected_portfolio:.2f}")
    print(f"Difference: ${portfolio_value - expected_portfolio:.2f}")
    
    # Alternative calculation
    alt_portfolio = initial_bankroll + all_pnl
    print(f"\nAlternative calc (initial + P&L): ${alt_portfolio:.2f}")
    
    # Check the math
    print("\nDETAILED BREAKDOWN:")
    print(f"Started with: ${initial_bankroll:.2f}")
    print(f"Closed P&L: ${all_pnl:.2f}")
    print(f"Currently invested: ${open_exec_stake:.2f}")
    print(f"Cash remaining: ${current_bankroll:.2f}")
    print(f"Should equal: ${initial_bankroll + all_pnl:.2f}")