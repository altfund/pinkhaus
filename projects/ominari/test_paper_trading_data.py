#!/usr/bin/env python3
"""Quick diagnostic to check paper trading data issues."""

from datetime import datetime, timezone
from evaluate_open_markets import fetch_open_markets_for_as_of
from signals import SIGNAL_PROVIDERS
import pandas as pd

def main():
    print("Testing paper trading data flow...")
    
    # Fetch current open markets
    as_of = datetime.now(timezone.utc)
    print(f"Fetching open markets as of {as_of}")
    
    try:
        df = fetch_open_markets_for_as_of(pd.Timestamp(as_of))
        print(f"Found {len(df)} markets")
        
        if len(df) > 0:
            print("\nFirst 5 markets:")
            print(df[['source_id', 'normalized_outcome', 'odds', 'implied_raw']].head())
            
            print("\nChecking for NaN values:")
            print(f"NaN in odds: {df['odds'].isna().sum()}")
            print(f"NaN in implied_raw: {df['implied_raw'].isna().sum()}")
            
            # Test signal provider
            print("\nTesting ImpliedRawSignal...")
            signal = next(s for s in SIGNAL_PROVIDERS if s.name == "implied_probability")
            probs = signal.get_probs(df)
            print(f"Got {len(probs)} probabilities")
            print(f"NaN in probabilities: {probs.isna().sum()}")
            
            if probs.isna().any():
                print("\nRows with NaN probabilities:")
                nan_mask = probs.isna()
                print(df[nan_mask][['source_id', 'normalized_outcome', 'odds', 'implied_raw']])
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()