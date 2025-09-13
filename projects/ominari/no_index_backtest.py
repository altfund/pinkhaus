#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest implementation that works efficiently without database indexes.
Uses smart query strategies to minimize database load on the 201GB database.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('../pinkhaus-models'))

import pandas as pd
import numpy as np
import sqlite3
import logging
from typing import List
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class NoIndexDataFetcher:
    """
    Fetches data efficiently without relying on indexes.
    Uses rowid and recent data strategies.
    """
    
    def __init__(self, db_path="sport_odds.db", cache_dir="cache/markets"):
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._market_cache = {}
        
    def fetch_recent_markets(self, limit=1000) -> pd.DataFrame:
        """
        Fetch most recent markets using rowid (always indexed).
        This is much faster than using timestamps without indexes.
        """
        conn = sqlite3.connect(self.db_path)
        
        # Use rowid DESC which is always fast
        query = """
        SELECT 
            o.rowid,
            o.source_id,
            o.market_type,
            o.outcome,
            o.decimal_odds,
            o.updated_at,
            o.bookmaker
        FROM odd o
        WHERE o.rowid > (SELECT MAX(rowid) - ? FROM odd)
        ORDER BY o.rowid DESC
        """
        
        logger.info(f"Fetching {limit} recent odds using rowid...")
        df = pd.read_sql_query(query, conn, params=(limit * 10,))  # Get more to filter
        conn.close()
        
        if df.empty:
            return df
            
        # Get unique recent markets
        df = df.drop_duplicates(subset=['source_id', 'outcome'], keep='first')
        df = df.head(limit)
        
        # Get market metadata from cache or fetch if needed
        df = self._enrich_with_market_data(df)
        
        return df
    
    def fetch_by_time_range_chunked(self, start: pd.Timestamp, end: pd.Timestamp,
                                   chunk_hours: int = 1) -> pd.DataFrame:
        """
        Fetch data in small time chunks to avoid large queries.
        """
        all_chunks = []
        current = start
        
        conn = sqlite3.connect(self.db_path)
        
        while current < end:
            chunk_end = min(current + pd.Timedelta(hours=chunk_hours), end)
            
            logger.info(f"Fetching chunk: {current} to {chunk_end}")
            
            # Smaller, focused query
            query = """
            SELECT 
                source_id,
                market_type,
                outcome,
                decimal_odds,
                updated_at
            FROM odd
            WHERE updated_at >= ? AND updated_at < ?
            ORDER BY updated_at DESC
            LIMIT 10000
            """
            
            chunk = pd.read_sql_query(
                query, 
                conn,
                params=(current.isoformat(), chunk_end.isoformat()),
                parse_dates=['updated_at']
            )
            
            if not chunk.empty:
                all_chunks.append(chunk)
                
            current = chunk_end
            
        conn.close()
        
        if not all_chunks:
            return pd.DataFrame()
            
        df = pd.concat(all_chunks, ignore_index=True)
        df = df.drop_duplicates(subset=['source_id', 'outcome'], keep='last')
        
        return self._enrich_with_market_data(df)
    
    def _enrich_with_market_data(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Add market metadata efficiently."""
        if odds_df.empty:
            return odds_df
            
        # Check cache first
        uncached_ids = []
        for source_id in odds_df['source_id'].unique():
            if source_id not in self._market_cache:
                uncached_ids.append(source_id)
        
        # Fetch uncached markets in batches
        if uncached_ids:
            self._fetch_markets_batch(uncached_ids[:100])  # Limit batch size
        
        # Enrich with cached data
        market_data = []
        for source_id in odds_df['source_id'].unique():
            if source_id in self._market_cache:
                market_data.append(self._market_cache[source_id])
            else:
                # Default values if not found
                market_data.append({
                    'source_id': source_id,
                    'home_team': 'Unknown',
                    'away_team': 'Unknown', 
                    'sport': 'Unknown',
                    'market_type': 'Unknown'
                })
        
        market_df = pd.DataFrame(market_data)
        
        # Merge with odds
        result = odds_df.merge(market_df, on='source_id', how='left', suffixes=('', '_market'))
        
        # Add computed columns
        result['implied_raw'] = 100 / result['decimal_odds']
        result['bet_name'] = result['outcome']
        result['match_id'] = result['home_team'] + "_vs_" + result['away_team']
        result['odds_market_type'] = result['market_type']
        result['line'] = 0  # Default line value for markets without lines
        
        return result
    
    def _fetch_markets_batch(self, source_ids: List[str]):
        """Fetch market data in batch."""
        if not source_ids:
            return
            
        conn = sqlite3.connect(self.db_path)
        
        placeholders = ','.join('?' * len(source_ids))
        query = f"""
        SELECT source_id, home_team, away_team, sport, market_type
        FROM market
        WHERE source_id IN ({placeholders})
        """
        
        df = pd.read_sql_query(query, conn, params=source_ids)
        conn.close()
        
        # Update cache
        for _, row in df.iterrows():
            self._market_cache[row['source_id']] = row.to_dict()
    
    def save_cache(self):
        """Save market cache to disk."""
        cache_file = self.cache_dir / "market_cache.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(self._market_cache, f)
            
    def load_cache(self):
        """Load market cache from disk."""
        cache_file = self.cache_dir / "market_cache.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                self._market_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._market_cache)} cached markets")


def run_no_index_backtest(strategy, n_samples=500):
    """
    Run a backtest using smart queries that don't require indexes.
    """
    from kelly_multimarket import calculate_kelly_stakes_with_exclusivity
    
    logger.info("=" * 60)
    logger.info("NO-INDEX BACKTEST")
    logger.info("=" * 60)
    logger.info("Using smart queries to avoid index dependency")
    
    # Initialize fetcher
    fetcher = NoIndexDataFetcher()
    fetcher.load_cache()
    
    # Get recent markets
    logger.info(f"\nFetching {n_samples} recent markets...")
    markets = fetcher.fetch_recent_markets(limit=n_samples)
    
    if markets.empty:
        logger.error("No markets found!")
        return None
        
    logger.info(f"Fetched {len(markets)} market odds")
    logger.info(f"Unique markets: {markets['source_id'].nunique()}")
    
    # Debug: show sample of data
    logger.info("\nSample odds:")
    sample = markets[['source_id', 'bet_name', 'decimal_odds', 'implied_raw']].head()
    for _, row in sample.iterrows():
        logger.info(f"  {row['bet_name'][:30]:30} odds={row['decimal_odds']:.3f} impl={row['implied_raw']:.1f}%")
    
    # Compute signals
    logger.info("\nComputing signals...")
    signal_provider = strategy['providers'][0]
    probabilities = signal_provider.get_probs(markets)
    
    # Prepare for Kelly
    bets_df = pd.DataFrame({
        'match_id': markets['match_id'],
        'source_id': markets['source_id'],  # Add source_id
        'unified_market_type': markets.get('market_type', 'unknown'),
        'bet_name': markets['bet_name'],
        'odds': markets['decimal_odds'],
        'probability': probabilities / 100.0,
        'line': markets.get('line', 0),  # Get line from markets or default to 0
        'normalized_line': markets.get('line', 0)  # normalized_line same as line for simple cases
    })
    
    # Filter out invalid bets (only mathematically impossible values)
    logger.info(f"\nBefore filtering: {len(bets_df)} bets")
    
    # Remove bets with invalid odds (must be > 1)
    invalid_odds = bets_df['odds'] <= 1.0
    if invalid_odds.any():
        logger.warning(f"Removing {invalid_odds.sum()} bets with odds <= 1.0")
        bets_df = bets_df[~invalid_odds]
    
    # Remove bets with invalid probabilities
    invalid_probs = (bets_df['probability'] <= 0) | (bets_df['probability'] >= 1)
    if invalid_probs.any():
        logger.warning(f"Removing {invalid_probs.sum()} bets with probability outside (0,1)")
        bets_df = bets_df[~invalid_probs]
    
    # Remove any rows with NaN values
    nan_rows = bets_df.isnull().any(axis=1)
    if nan_rows.any():
        logger.warning(f"Removing {nan_rows.sum()} bets with NaN values")
        bets_df = bets_df[~nan_rows]
    
    logger.info(f"After filtering: {len(bets_df)} valid bets")
    
    # Show EV distribution (for diagnostics)
    ev = bets_df['probability'] * bets_df['odds']
    logger.info("Expected value distribution:")
    logger.info(f"  Positive EV (>1): {(ev > 1).sum()} bets")
    logger.info(f"  Negative EV (<1): {(ev < 1).sum()} bets")
    logger.info(f"  Mean EV: {ev.mean():.3f}")
    
    if len(bets_df) == 0:
        logger.warning("No valid bets after filtering!")
        return pd.DataFrame()
        
    # Check for potential issues with exclusivity groups
    groups = bets_df.groupby(['source_id', 'unified_market_type', 'normalized_line'])
    logger.info(f"\nExclusivity groups: {groups.ngroups}")
    for name, group in groups:
        if len(group) > 1:
            logger.debug(f"  Group {name}: {len(group)} mutually exclusive bets")
    
    # Calculate stakes
    logger.info("Calculating Kelly stakes...")
    try:
        # Add missing columns that Kelly expects
        bets_df_kelly = bets_df.copy()
        bets_df_kelly['normalized_outcome'] = bets_df_kelly['bet_name']
        bets_df_kelly['market_name'] = bets_df_kelly['match_id']
        bets_df_kelly['league_name'] = 'Unknown'
        bets_df_kelly['bookmaker'] = 'Unknown'
        
        result_df = calculate_kelly_stakes_with_exclusivity(
            bets_df_kelly,
            bankroll=strategy.get('bankroll', 1000),
            risk_adjusted=strategy.get('risk_adjusted', True),
            max_stake_per_bet=strategy.get('max_stake_per_bet')
        )
        
        # Extract just the stakes
        stakes = result_df['stake']
    except ValueError as e:
        logger.error(f"Kelly optimization failed: {e}")
        logger.info("Falling back to simple Kelly formula...")
        
        # Simple Kelly: f = (p*b - q) / b, where p=prob, q=1-p, b=odds-1
        p = bets_df['probability'].values
        b = bets_df['odds'].values - 1
        q = 1 - p
        
        # Kelly fraction (will be negative for negative EV bets)
        f = (p * b - q) / b
        
        # Apply Kelly fraction (e.g., 25% Kelly)
        kelly_fraction = strategy.get('kelly_fraction', 0.25)
        f = f * kelly_fraction
        
        # Clip to reasonable bounds
        # Important: lower bound is 0 (no negative stakes, even for negative EV)
        f = np.clip(f, 0, 0.1)  # Max 10% per bet
        
        # Convert to stakes
        bankroll = strategy.get('bankroll', 1000)
        stakes = pd.Series(f * bankroll, index=bets_df.index)
        
        logger.info(f"Simple Kelly allocated stakes to {(stakes > 0).sum()} bets")
    
    # Results
    results = pd.DataFrame({
        'source_id': bets_df['source_id'],
        'bet_name': bets_df['bet_name'],
        'odds': bets_df['odds'],
        'probability': bets_df['probability'] * 100,  # Convert back to percentage
        'stake': stakes
    })
    
    # Filter to non-zero stakes
    results = results[results['stake'] > 0]
    
    logger.info("\nResults:")
    logger.info(f"Bets placed: {len(results)}")
    logger.info(f"Total staked: ${results['stake'].sum():.2f}")
    
    if len(results) > 0:
        logger.info("\nTop 5 bets:")
        top_bets = results.nlargest(5, 'stake')
        for _, bet in top_bets.iterrows():
            logger.info(f"  {bet['bet_name']}: ${bet['stake']:.2f} at {bet['odds']:.2f}")
    
    # Save cache for next run
    fetcher.save_cache()
    
    return results


def main():
    """Demo no-index backtest."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from signals import ImpliedRawSignal
    
    # Test strategy
    strategy = {
        "name": "no_index_test",
        "providers": [ImpliedRawSignal()],
        "weights": [1.0],
        "bankroll": 1000,
        "risk_adjusted": True,
        "max_stake_per_bet": 100
    }
    
    logger.info("Running backtest without database indexes...")
    logger.info("This approach uses smart queries that work on large databases")
    
    results = run_no_index_backtest(strategy, n_samples=500)
    
    if results is not None:
        # Save results
        results.to_csv("no_index_backtest_results.csv", index=False)
        logger.info("\nResults saved to no_index_backtest_results.csv")


if __name__ == "__main__":
    main()