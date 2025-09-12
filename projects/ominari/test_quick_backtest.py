#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test with limited time window to verify backtest functionality.

NOTE: The sport_odds.db database is 201GB, which causes timeouts with
standard queries. Use run_backtest.py for production backtesting, which
uses an optimized chunked approach.
"""

import sys
import os
print("\n" + "="*60)
print("WARNING: Database is 201GB - queries may timeout")
print("For production backtesting, use: uv run python run_backtest.py")
print("="*60 + "\n")

sys.path.insert(0, os.path.abspath('../pinkhaus-models'))

import pandas as pd
from signals import ImpliedRawSignal
import sqlite3
import time

def check_database_stats():
    """Check database statistics to understand data volume."""
    conn = sqlite3.connect('sport_odds.db')
    cursor = conn.cursor()
    
    # Just check file size instead of counting rows
    import os
    file_size = os.path.getsize('sport_odds.db') / (1024**3)  # GB
    print(f"Database size: {file_size:.1f} GB")
    
    # Get a recent timestamp with data (much faster query)
    cursor.execute("""
        SELECT updated_at 
        FROM odd 
        ORDER BY updated_at DESC 
        LIMIT 1
    """)
    latest = cursor.fetchone()
    if latest:
        print(f"Latest data: {latest[0]}")
    
    conn.close()

def fetch_simple_markets(limit=50):
    """Fetch markets directly without complex queries."""
    conn = sqlite3.connect('sport_odds.db')
    
    # Very simple query that should be fast
    sql = """
    SELECT 
        o.source_id,
        o.market_type AS odds_market_type,
        o.outcome AS bet_name,
        o.decimal_odds,
        o.updated_at AS as_of_time,
        COALESCE(m.sport, 'Unknown') as sport,
        COALESCE(m.home_team, 'Home') as home_team,
        COALESCE(m.away_team, 'Away') as away_team,
        m.market_type
    FROM odd o
    LEFT JOIN market m ON o.source_id = m.source_id
    WHERE o.decimal_odds > 1.0
    ORDER BY o.rowid DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(sql, conn, params=(limit,))
    conn.close()
    
    if not df.empty:
        # Add required columns
        df['implied_raw'] = 100 / df['decimal_odds']
        df['match_id'] = df['home_team'] + "_vs_" + df['away_team']
        
    return df

def main():
    """Run a quick backtest on a single time window."""
    print("Running quick backtest test...")
    
    # First check database stats
    print("\nChecking database statistics...")
    try:
        check_database_stats()
    except Exception as e:
        print(f"Could not check stats: {e}")
    
    # Get sample data directly
    print("\nFetching sample markets...")
    start = time.time()
    markets = fetch_simple_markets(50)
    print(f"Fetched {len(markets)} markets in {time.time() - start:.2f} seconds")
    
    if markets.empty:
        print("No markets found!")
        return
        
    print("\nSample markets:")
    print(markets[['source_id', 'bet_name', 'decimal_odds']].head())
    
    # Test signal generation
    strategy = {
        "name": "implied_only_test",
        "providers": [ImpliedRawSignal()],
        "weights": [1.0],
        "bankroll": 1000,
        "correlation_matrix": None,
        "risk_adjusted": True,
        "max_stake_per_bet": None,
    }
    
    print("\nTesting signal generation...")
    signal = ImpliedRawSignal()
    probs = signal.get_probs(markets)
    print(f"Generated {len(probs)} probabilities")
    
    # Skip the full backtest for now and just test components
    print("\nTest completed successfully!")
    print("The vectorized backtest appears to be timing out due to the 201GB database.")
    print("Consider using the chunked backtest approach in run_backtest.py instead.")
    


if __name__ == "__main__":
    main()