#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed quick backtest that handles the 201GB database efficiently.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('../pinkhaus-models'))

import pandas as pd
from datetime import timezone
from sqlalchemy import create_engine
import time

# Monkey-patch the fetch function to limit data
def fetch_limited_markets(as_of: pd.Timestamp, limit: int = 100) -> pd.DataFrame:
    """Fetch only a limited number of markets to avoid timeout."""
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize(timezone.utc)
    
    engine = create_engine("sqlite:///sport_odds.db")
    conn = engine.connect()
    
    # Much more efficient query that limits early
    sql = """
    WITH recent_odds AS (
        SELECT source_id, market_type, outcome, decimal_odds, updated_at
        FROM odd
        WHERE updated_at <= ?
        ORDER BY updated_at DESC
        LIMIT 10000
    ),
    latest_per_market AS (
        SELECT source_id, MAX(updated_at) as max_time
        FROM recent_odds
        GROUP BY source_id
        LIMIT ?
    )
    SELECT 
        ro.source_id,
        ro.market_type AS odds_market_type,
        ro.outcome AS bet_name,
        ro.decimal_odds,
        ro.updated_at AS as_of_time,
        m.sport,
        m.league_name,
        m.home_team,
        m.away_team,
        m.market_type,
        m.tournament,
        m.source AS bookmaker,
        m.maturity_date
    FROM recent_odds ro
    INNER JOIN latest_per_market lpm 
        ON ro.source_id = lpm.source_id 
        AND ro.updated_at = lpm.max_time
    LEFT JOIN market m ON m.source_id = ro.source_id
    """
    
    df = pd.read_sql_query(
        sql,
        conn,
        params=(as_of.isoformat(), limit),
        parse_dates=["as_of_time"]
    )
    conn.close()
    return df

def main():
    """Run a quick backtest with limited data."""
    print("Running fixed quick backtest...")
    print("Database size: 201 GB")
    
    # Import after patching
    import evaluate_open_markets
    original_fetch = evaluate_open_markets.fetch_open_markets_for_as_of
    evaluate_open_markets.fetch_open_markets_for_as_of = fetch_limited_markets
    
    try:
        from vectorized_backtest import vectorized_backtest
        from signals import ImpliedRawSignal
        
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
        
        # Use current time minus 1 hour
        as_of = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=1)
        until = as_of + pd.Timedelta(minutes=15)
        
        print(f"\nTesting window: {as_of} to {until} (15 minutes)")
        print("Fetching limited markets (max 100)...")
        
        start_time = time.time()
        
        # Test the fetch function first
        markets = fetch_limited_markets(as_of, 50)
        print(f"Fetched {len(markets)} market odds in {time.time() - start_time:.2f} seconds")
        
        if markets.empty:
            print("\nNo recent markets found. The database might not have recent data.")
            return
        
        print("\nRunning backtest...")
        start_time = time.time()
        
        results = vectorized_backtest(
            strategies=[strategy],
            as_of=as_of,
            until=until,
            require_outcome=False
        )
        
        elapsed = time.time() - start_time
        print(f"Backtest completed in {elapsed:.2f} seconds")
        
        if results is not None and not results.empty:
            print(f"\nResults: {len(results)} bets placed")
            print(f"Total staked: ${results['total_staked'].sum():.2f}")
            if len(results) > 0:
                print("\nSample results:")
                print(results[['strategy_name', 'source_id', 'total_staked']].head(3))
        else:
            print("\nNo bets placed in this window")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original function
        evaluate_open_markets.fetch_open_markets_for_as_of = original_fetch

if __name__ == "__main__":
    main()