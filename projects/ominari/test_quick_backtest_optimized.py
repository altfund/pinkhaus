#!/usr/bin/env python3
"""
Optimized quick backtest that handles the 201GB database efficiently.
Uses simplified queries and adds all required columns.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('../pinkhaus-models'))

import pandas as pd
from datetime import timezone
import time
import sqlite3

DB_NAME = "sport_odds.db"

def fetch_limited_markets_optimized(as_of: pd.Timestamp, limit: int = 100) -> pd.DataFrame:
    """
    Optimized fetch that:
    1. Uses rowid for fast ordering
    2. Limits data early
    3. Adds all required columns including implied_raw
    """
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize(timezone.utc)
    
    conn = sqlite3.connect(DB_NAME)
    
    # First get recent odds using rowid (much faster than timestamp ordering)
    sql = """
    WITH recent_odds AS (
        SELECT 
            source_id,
            market_type,
            outcome,
            decimal_odds,
            updated_at,
            rowid
        FROM odd
        WHERE updated_at <= ?
        ORDER BY rowid DESC
        LIMIT 5000
    ),
    latest_per_market AS (
        SELECT 
            source_id,
            MAX(updated_at) as max_time
        FROM recent_odds
        GROUP BY source_id
        ORDER BY max_time DESC
        LIMIT ?
    )
    SELECT 
        ro.source_id,
        ro.market_type AS odds_market_type,
        ro.outcome AS bet_name,
        ro.decimal_odds,
        ro.updated_at AS as_of_time,
        COALESCE(m.sport, 'Unknown') as sport,
        COALESCE(m.league_name, 'Unknown') as league_name,
        COALESCE(m.home_team, 'Unknown') as home_team,
        COALESCE(m.away_team, 'Unknown') as away_team,
        COALESCE(m.market_type, ro.market_type) as market_type,
        m.tournament,
        m.source AS bookmaker,
        m.maturity_date
    FROM recent_odds ro
    INNER JOIN latest_per_market lpm 
        ON ro.source_id = lpm.source_id 
        AND ro.updated_at = lpm.max_time
    LEFT JOIN market m ON m.source_id = ro.source_id
    WHERE ro.decimal_odds > 0  -- Filter out invalid odds
    """
    
    df = pd.read_sql_query(
        sql,
        conn,
        params=(as_of.isoformat(), limit),
        parse_dates=["as_of_time"]
    )
    conn.close()
    
    # Add calculated columns
    if not df.empty:
        # Calculate implied probability
        df['implied_raw'] = 100 / df['decimal_odds']
        
        # Add match_id if not present
        df['match_id'] = df['home_team'] + "_vs_" + df['away_team']
        
        # Ensure all required columns exist
        for col in ['source', 'opening_timestamp']:
            if col not in df.columns:
                df[col] = None
                
    return df

def main():
    """Run optimized quick backtest."""
    print("Running optimized quick backtest...")
    print("Database size: 201 GB\n")
    
    # Import modules
    import evaluate_open_markets
    from vectorized_backtest import vectorized_backtest
    from signals import ImpliedRawSignal
    
    # Also need to patch database_utils if it exists
    try:
        import database_utils
        original_db_fetch = database_utils.fetch_open_markets_optimized
        database_utils.fetch_open_markets_optimized = fetch_limited_markets_optimized
    except ImportError:
        original_db_fetch = None
    
    # Monkey-patch the fetch function
    original_fetch = evaluate_open_markets.fetch_open_markets_for_as_of
    evaluate_open_markets.fetch_open_markets_for_as_of = fetch_limited_markets_optimized
    
    try:
        # Create a simple strategy
        strategy = {
            "name": "implied_only_test",
            "providers": [ImpliedRawSignal()],
            "weights": [1.0],
            "bankroll": 1000,
            "correlation_matrix": None,
            "risk_adjusted": True,
            "max_stake_per_bet": None,
        }
        
        # Use a recent timestamp
        as_of = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=2)
        until = as_of + pd.Timedelta(minutes=30)
        
        print(f"Testing window: {as_of} to {until}")
        print("Fetching limited markets (max 50)...\n")
        
        # First test the fetch
        start_time = time.time()
        markets = fetch_limited_markets_optimized(as_of, 50)
        fetch_time = time.time() - start_time
        
        print(f"Fetched {len(markets)} market odds in {fetch_time:.2f} seconds")
        
        if markets.empty:
            print("\nNo markets found. Trying an earlier timestamp...")
            # Try 1 day ago
            as_of = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1)
            until = as_of + pd.Timedelta(minutes=30)
            markets = fetch_limited_markets_optimized(as_of, 50)
            
        if not markets.empty:
            print("\nSample markets:")
            print(markets[['source_id', 'bet_name', 'decimal_odds', 'implied_raw']].head(5))
            
            print("\nRunning backtest...")
            start_time = time.time()
            
            results = vectorized_backtest(
                strategies=[strategy],
                as_of=as_of,
                until=until,
                require_outcome=False
            )
            
            backtest_time = time.time() - start_time
            print(f"Backtest completed in {backtest_time:.2f} seconds")
            
            if results is not None and not results.empty:
                print(f"\nResults: {len(results)} bets evaluated")
                print(f"Total staked: ${results['total_staked'].sum():.2f}")
                
                # Show non-zero bets
                non_zero_bets = results[results['total_staked'] > 0]
                if not non_zero_bets.empty:
                    print(f"\nBets placed: {len(non_zero_bets)}")
                    print("\nTop 5 bets by stake:")
                    top_bets = non_zero_bets.nlargest(5, 'total_staked')
                    print(top_bets[['source_id', 'total_staked']])
                else:
                    print("\nNo bets were placed (all stakes were 0)")
            else:
                print("\nNo results generated from backtest")
        else:
            print("\nNo markets found in the database around this time")
            print("The database might not have recent data")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original functions
        evaluate_open_markets.fetch_open_markets_for_as_of = original_fetch
        if original_db_fetch is not None:
            database_utils.fetch_open_markets_optimized = original_db_fetch
        print("\nTest completed")

if __name__ == "__main__":
    main()