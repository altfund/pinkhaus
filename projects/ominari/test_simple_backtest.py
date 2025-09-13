#!/usr/bin/env python3
"""
Ultra-simple backtest that avoids complex queries on the 201GB database.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('../pinkhaus-models'))

import pandas as pd
import sqlite3
import time

def get_recent_market_sample():
    """Get a tiny sample of recent markets to test with."""
    conn = sqlite3.connect('sport_odds.db')
    
    # Super simple query - just get 10 recent odds records
    sql = """
    SELECT 
        o.source_id,
        o.market_type,
        o.outcome,
        o.decimal_odds,
        o.updated_at,
        m.sport,
        m.home_team,
        m.away_team,
        m.market_type as market_market_type
    FROM odd o
    LEFT JOIN market m ON o.source_id = m.source_id
    ORDER BY o.rowid DESC
    LIMIT 10
    """
    
    print("Fetching 10 most recent odds...")
    start = time.time()
    df = pd.read_sql_query(sql, conn)
    print(f"Query took {time.time() - start:.2f} seconds")
    conn.close()
    
    return df

def main():
    print("Running ultra-simple backtest test...")
    print("Database size: 201 GB")
    
    # Get sample data
    sample_df = get_recent_market_sample()
    
    if sample_df.empty:
        print("No data found!")
        return
        
    print(f"\nFound {len(sample_df)} odds records")
    print("\nSample data:")
    print(sample_df[['source_id', 'outcome', 'decimal_odds', 'updated_at']].to_string())
    
    # Test signal generation
    try:
        from signals import ImpliedRawSignal
        signal = ImpliedRawSignal()
        
        # Add required columns for signal
        sample_df['odds_market_type'] = sample_df['market_type']
        sample_df['bet_name'] = sample_df['outcome']
        
        # Calculate implied_raw from decimal odds
        # Skip odds of 0 to avoid division by zero
        sample_df['implied_raw'] = sample_df.apply(
            lambda row: 100 / row['decimal_odds'] if row['decimal_odds'] > 0 else 0,
            axis=1
        )
        
        print("\nTesting signal generation...")
        probs = signal.get_probs(sample_df)
        print(f"Generated {len(probs)} probabilities")
        print("\nProbabilities (as percentages):")
        for idx, (source_id, outcome, odds, prob) in enumerate(zip(
            sample_df['source_id'], 
            sample_df['outcome'], 
            sample_df['decimal_odds'],
            probs
        )):
            if odds > 0:
                print(f"  {idx}: {outcome[:20]:20} odds={odds:.3f} prob={prob*100:.1f}%")
        
    except Exception as e:
        print(f"Error testing signals: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()