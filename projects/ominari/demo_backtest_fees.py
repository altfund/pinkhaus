#!/usr/bin/env python3
"""Demo script to show backtesting now properly accounts for fees."""

import pandas as pd
from datetime import datetime, timezone, timedelta
from vectorized_backtest import fetch_markets_with_outcomes
from evaluate_open_markets import apply_overtime_fees

def demo_fee_calculation():
    """Show how fees are now included in market data."""
    
    print("Demonstrating fee-aware backtesting...\n")
    
    # Create a simple test dataframe representing market data
    test_data = pd.DataFrame({
        'source_id': ['market_1', 'market_2', 'market_3'],
        'market_type': ['winner', 'winner', 'winner'],
        'bet_name': ['Home', 'Away', 'Draw'],
        'home_team': ['Team A', 'Team B', 'Team C'],
        'away_team': ['Team X', 'Team Y', 'Team Z'],
        'decimal_odds': [2.50, 1.80, 3.20],
        'odds': [2.50, 1.80, 3.20],  # apply_overtime_fees expects 'odds' column
        'bookmaker': ['overtime_markets', 'overtime_markets', 'overtime_markets'],  # needed for fee logic
        'source': ['overtime_markets', 'overtime_markets', 'overtime_markets'],
        'sport': ['Soccer', 'Soccer', 'Soccer']
    })
    
    print("Original odds (before fees):")
    print(test_data[['source_id', 'decimal_odds']])
    
    # Apply fees
    with_fees = apply_overtime_fees(test_data.copy())
    
    print("\n\nAfter applying fees:")
    print(with_fees[['source_id', 'decimal_odds', 'total_fee_pct', 'adjusted_odds']])
    
    # Show how this affects Kelly calculations
    print("\n\nFee impact on betting edge:")
    for idx, row in with_fees.iterrows():
        raw_odds = row['decimal_odds']
        adj_odds = row['adjusted_odds']
        fee_pct = row['total_fee_pct']
        
        # Example: if we think true probability is 45%
        true_prob = 0.45
        
        # Edge without fees
        raw_edge = true_prob * raw_odds - 1
        
        # Edge with fees
        adj_edge = true_prob * adj_odds - 1
        
        print(f"\n{row['source_id']}:")
        print(f"  Raw odds: {raw_odds:.2f}, Adjusted odds: {adj_odds:.2f}")
        print(f"  Fee: {fee_pct*100:.1f}%")
        print(f"  Edge without fees: {raw_edge*100:.1f}%")
        print(f"  Edge with fees: {adj_edge*100:.1f}%")
        print(f"  Fee reduces edge by: {(raw_edge - adj_edge)*100:.1f} percentage points")
    
    # Show P&L calculation with fees
    print("\n\n=== P&L Calculation Example ===")
    stake = 100
    fee_pct = 0.03  # 3% total fee
    fee_amount = stake * fee_pct
    execution_stake = stake + fee_amount
    
    print(f"Stake: ${stake:.2f}")
    print(f"Fees (3%): ${fee_amount:.2f}")
    print(f"Execution stake: ${execution_stake:.2f}")
    
    # Win case
    odds = 2.00
    gross_payout = stake * odds
    net_win = gross_payout - execution_stake
    
    print(f"\nIf bet wins at {odds:.2f} odds:")
    print(f"  Gross payout: ${gross_payout:.2f}")
    print(f"  Net P&L: ${net_win:.2f}")
    print(f"  ROI: {net_win/execution_stake*100:.1f}%")
    
    # Loss case
    net_loss = -execution_stake
    print(f"\nIf bet loses:")
    print(f"  Net P&L: ${net_loss:.2f}")
    print(f"  ROI: {net_loss/execution_stake*100:.1f}%")

if __name__ == "__main__":
    demo_fee_calculation()