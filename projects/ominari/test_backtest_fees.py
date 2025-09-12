#!/usr/bin/env python3
"""Test script to verify backtest system properly accounts for fees."""

import pandas as pd
import numpy as np
from vectorized_backtest import vectorized_backtest
from signals import ImpliedRawSignal
from datetime import datetime, timezone, timedelta

def test_fee_accounting():
    """Test that fees are properly included in backtest calculations."""
    
    print("Testing backtest fee accounting...")
    
    # Create a test strategy
    test_strategy = {
        "name": "test_fees",
        "providers": [ImpliedRawSignal()],
        "weights": [1.0],
        "bankroll": 1000,
        "correlation_matrix": None,
        "risk_adjusted": True,
        "max_stake_per_bet": 50,
    }
    
    # Define test window (recent data)
    as_of = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=7))
    until = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=1))
    
    try:
        # Run backtest
        print(f"\nRunning backtest from {as_of} to {until}...")
        results = vectorized_backtest(
            strategies=[test_strategy],
            as_of=as_of,
            until=until,
            require_outcome=True
        )
        
        if results.empty:
            print("No results found in test window")
            return
            
        print(f"\nFound {len(results)} betting decisions")
        
        # Check if fee columns exist
        has_fees = "fee_amount" in results.columns and "execution_stake" in results.columns
        print(f"\nFee columns present: {has_fees}")
        
        if has_fees:
            # Verify fee calculations
            sample = results.head(10).copy()
            
            print("\n=== Sample Results (first 10 bets) ===")
            print(sample[["total_staked", "fee_pct", "fee_amount", "execution_stake", "result_multiplier", "net", "roi"]])
            
            # Verify fee calculations are correct
            sample["calc_fee_amount"] = sample["total_staked"] * sample["fee_pct"]
            sample["calc_execution_stake"] = sample["total_staked"] + sample["calc_fee_amount"]
            
            fee_diff = (sample["fee_amount"] - sample["calc_fee_amount"]).abs().max()
            exec_diff = (sample["execution_stake"] - sample["calc_execution_stake"]).abs().max()
            
            print(f"\nMaximum fee calculation difference: ${fee_diff:.6f}")
            print(f"Maximum execution stake difference: ${exec_diff:.6f}")
            
            # Check P&L calculations
            print("\n=== P&L Verification ===")
            for idx, row in sample.iterrows():
                if row["result_multiplier"] > 0:
                    # Win case
                    expected_pnl = row["total_staked"] * row["result_multiplier"] - row["execution_stake"]
                else:
                    # Loss case
                    expected_pnl = -row["execution_stake"]
                    
                actual_pnl = row["net"]
                diff = abs(expected_pnl - actual_pnl)
                
                if diff > 0.01:
                    print(f"Row {idx}: Expected P&L ${expected_pnl:.2f}, Actual ${actual_pnl:.2f}, Diff: ${diff:.2f}")
            
            # Summary statistics
            print("\n=== Summary Statistics ===")
            total_staked = results["total_staked"].sum()
            total_fees = results["fee_amount"].sum()
            total_execution = results["execution_stake"].sum()
            total_net = results["net"].sum()
            
            print(f"Total Staked: ${total_staked:.2f}")
            print(f"Total Fees: ${total_fees:.2f} ({total_fees/total_staked*100:.1f}%)")
            print(f"Total Execution Stake: ${total_execution:.2f}")
            print(f"Net P&L: ${total_net:.2f}")
            print(f"ROI on Execution Stake: {total_net/total_execution*100:.2f}%")
            
        else:
            print("\nWARNING: Fee columns not found in results!")
            print("Available columns:", list(results.columns))
            
    except Exception as e:
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fee_accounting()