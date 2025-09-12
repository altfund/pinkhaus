#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized vectorized backtest with caching and sliding window approach.
Designed to handle large databases efficiently by reusing data across time windows.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional
from collections import defaultdict
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class SlidingWindowCache:
    """
    Maintains a sliding window of market data to avoid repeated database queries.
    """
    
    def __init__(self):
        self.markets_cache: Dict[str, pd.Series] = {}  # source_id -> market data
        self.odds_cache: Dict[str, pd.Series] = {}     # source_id -> latest odds
        self.current_window_start: Optional[pd.Timestamp] = None
        self.current_window_end: Optional[pd.Timestamp] = None
        
    def update_window(self, new_start: pd.Timestamp, new_end: pd.Timestamp, 
                     fetch_func, remove_func) -> pd.DataFrame:
        """
        Update the cache for a new time window.
        
        Args:
            new_start: Start of new window
            new_end: End of new window
            fetch_func: Function to fetch new markets (start, end) -> DataFrame
            remove_func: Function to get closed markets (start, end) -> List[str]
            
        Returns:
            DataFrame of all markets in the current window
        """
        if self.current_window_start is None:
            # First window - fetch all data
            logger.info(f"Initial window: {new_start} to {new_end}")
            df = fetch_func(new_start, new_end)
            self._populate_cache(df)
            self.current_window_start = new_start
            self.current_window_end = new_end
            return df
            
        # Sliding window update
        logger.debug(f"Sliding window from [{self.current_window_start}, {self.current_window_end}] "
                    f"to [{new_start}, {new_end}]")
        
        # Remove markets that closed
        if new_start > self.current_window_start:
            closed_ids = remove_func(self.current_window_start, new_start)
            for source_id in closed_ids:
                self.markets_cache.pop(source_id, None)
                self.odds_cache.pop(source_id, None)
                
        # Add new markets that opened
        if new_start != self.current_window_start or new_end != self.current_window_end:
            # Fetch only new/updated markets
            new_df = fetch_func(
                max(new_start, self.current_window_end), 
                new_end
            )
            self._populate_cache(new_df)
            
        self.current_window_start = new_start
        self.current_window_end = new_end
        
        # Return current window data as DataFrame
        return self._cache_to_dataframe()
    
    def _populate_cache(self, df: pd.DataFrame):
        """Add/update cache with new data."""
        for _, row in df.iterrows():
            source_id = row['source_id']
            self.markets_cache[source_id] = row
            # Store latest odds update time for staleness check
            if 'updated_at' in row:
                self.odds_cache[source_id] = row['updated_at']
                
    def _cache_to_dataframe(self) -> pd.DataFrame:
        """Convert cache to DataFrame."""
        if not self.markets_cache:
            return pd.DataFrame()
        return pd.DataFrame(list(self.markets_cache.values()))


class SignalCache:
    """
    Caches signal computations to avoid recalculating unchanged markets.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.cache_dir = cache_dir or Path("cache/signals")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_or_compute(self, provider_name: str, market_df: pd.DataFrame, 
                      compute_func) -> pd.Series:
        """
        Get cached signals or compute if missing/stale.
        
        Args:
            provider_name: Name of signal provider
            market_df: DataFrame with market data
            compute_func: Function to compute signals
            
        Returns:
            Series of signal values
        """
        # Create cache keys based on source_id and odds update time
        cache_keys = market_df.apply(
            lambda row: f"{row['source_id']}_{row.get('updated_at', 'none')}", 
            axis=1
        )
        
        # Check cache and identify missing
        signals = []
        missing_mask = []
        
        for idx, key in enumerate(cache_keys):
            if key in self.cache[provider_name]:
                signals.append(self.cache[provider_name][key])
                missing_mask.append(False)
            else:
                signals.append(np.nan)
                missing_mask.append(True)
                
        signals = pd.Series(signals, index=market_df.index)
        missing_mask = pd.Series(missing_mask, index=market_df.index)
        
        # Compute missing signals
        if missing_mask.any():
            missing_df = market_df[missing_mask]
            computed = compute_func(missing_df)
            
            # Update cache
            for idx, (key, value) in zip(missing_df.index, 
                                        zip(cache_keys[missing_mask], computed)):
                signals.loc[idx] = value
                self.cache[provider_name][key[0]] = value
                
        return signals
    
    def save_to_disk(self):
        """Persist cache to disk."""
        cache_file = self.cache_dir / "signal_cache.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(dict(self.cache), f)
            
    def load_from_disk(self):
        """Load cache from disk if exists."""
        cache_file = self.cache_dir / "signal_cache.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                self.cache = defaultdict(dict, pickle.load(f))


class BatchDataLoader:
    """
    Loads all required data upfront for the entire backtest period.
    """
    
    def __init__(self, db_path: str = "sport_odds.db"):
        self.db_path = db_path
        self.markets_df: Optional[pd.DataFrame] = None
        self.odds_timeline: Optional[pd.DataFrame] = None
        
    def load_period(self, start: pd.Timestamp, end: pd.Timestamp, 
                   chunk_size: int = 50000) -> None:
        """
        Load all data for the backtest period in chunks.
        
        Args:
            start: Start timestamp
            end: End timestamp  
            chunk_size: Rows per database chunk
        """
        import sqlite3
        
        logger.info(f"Loading data from {start} to {end}")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load markets
        market_query = """
        SELECT DISTINCT m.*, o.updated_at as first_seen
        FROM market m
        INNER JOIN (
            SELECT source_id, MIN(updated_at) as updated_at
            FROM odd
            WHERE updated_at BETWEEN ? AND ?
            GROUP BY source_id
        ) o ON m.source_id = o.source_id
        """
        
        self.markets_df = pd.read_sql_query(
            market_query, 
            conn, 
            params=(start.isoformat(), end.isoformat()),
            parse_dates=['maturity_date', 'first_seen']
        )
        
        logger.info(f"Loaded {len(self.markets_df)} unique markets")
        
        # Load odds timeline in chunks
        odds_chunks = []
        offset = 0
        
        while True:
            odds_query = f"""
            SELECT source_id, market_type, outcome, decimal_odds, 
                   updated_at, bookmaker
            FROM odd
            WHERE updated_at BETWEEN ? AND ?
            ORDER BY updated_at
            LIMIT {chunk_size} OFFSET {offset}
            """
            
            chunk = pd.read_sql_query(
                odds_query,
                conn,
                params=(start.isoformat(), end.isoformat()),
                parse_dates=['updated_at']
            )
            
            if chunk.empty:
                break
                
            odds_chunks.append(chunk)
            offset += chunk_size
            logger.info(f"Loaded {offset} odds records...")
            
        conn.close()
        
        self.odds_timeline = pd.concat(odds_chunks, ignore_index=True)
        logger.info(f"Total odds records: {len(self.odds_timeline)}")
        
    def get_snapshot_at(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        """
        Get market snapshot at a specific timestamp using loaded data.
        """
        if self.odds_timeline is None:
            raise ValueError("Data not loaded. Call load_period first.")
            
        # Get latest odds for each market before timestamp
        mask = self.odds_timeline['updated_at'] <= timestamp
        latest_odds = (self.odds_timeline[mask]
                      .sort_values('updated_at')
                      .groupby(['source_id', 'outcome'])
                      .last()
                      .reset_index())
        
        # Merge with market data
        snapshot = latest_odds.merge(
            self.markets_df,
            on='source_id',
            how='left'
        )
        
        # Add computed columns
        snapshot['implied_raw'] = 100 / snapshot['decimal_odds']
        snapshot['bet_name'] = snapshot['outcome']
        snapshot['odds_market_type'] = snapshot['market_type_x']
        snapshot['match_id'] = (snapshot['home_team'].astype(str) + 
                               "_vs_" + 
                               snapshot['away_team'].astype(str))
        
        return snapshot


def create_optimized_fetch_functions(loader: BatchDataLoader):
    """
    Create fetch functions that use the batch loader.
    """
    def fetch_new_markets(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # In batch mode, we already have all data
        return loader.get_snapshot_at(end)
    
    def get_closed_markets(start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
        # Markets that reached maturity
        mask = ((loader.markets_df['maturity_date'] >= start) & 
                (loader.markets_df['maturity_date'] < end))
        return loader.markets_df[mask]['source_id'].tolist()
        
    return fetch_new_markets, get_closed_markets


def cached_vectorized_backtest(
    strategies: list,
    as_of: pd.Timestamp,
    until: pd.Timestamp,
    window_cache: SlidingWindowCache,
    signal_cache: SignalCache,
    batch_loader: Optional[BatchDataLoader] = None,
    require_outcome: bool = False
) -> pd.DataFrame:
    """
    Vectorized backtest with caching optimizations.
    
    This is a drop-in replacement for vectorized_backtest() but with:
    - Sliding window cache for market data
    - Signal computation caching
    - Optional batch data loading
    """
    from kelly_multimarket import calculate_kelly_stakes_with_exclusivity
    
    # Get fetch functions based on mode
    if batch_loader:
        fetch_func, remove_func = create_optimized_fetch_functions(batch_loader)
    else:
        # Fallback to original database queries
        from evaluate_open_markets import fetch_open_markets_for_as_of
        
        def fetch_func(start, end):
            return fetch_open_markets_for_as_of(start)
            
        def remove_func(start, end):
            return []  # TODO: Implement actual removal logic
    
    # Update window cache
    market_df = window_cache.update_window(as_of, until, fetch_func, remove_func)
    
    if market_df.empty:
        logger.warning(f"No markets in window {as_of} to {until}")
        return pd.DataFrame()
    
    # Process strategies
    all_perf = []
    
    for spec in strategies:
        name = spec.get("name")
        providers = spec.get("providers", [])
        weights = spec.get("weights", [])
        bankroll = spec.get("bankroll", 1000)
        
        # Compute signals with caching
        signals = []
        for provider in providers:
            if signal_cache:
                signal = signal_cache.get_or_compute(
                    provider.name,
                    market_df,
                    provider.get_probs
                )
            else:
                signal = provider.get_probs(market_df)
            signals.append(signal.fillna(0.0))
        
        # Combine signals
        if isinstance(weights, dict):
            weight_list = [weights.get(p.name, 0.0) for p in providers]
        else:
            weight_list = weights
            
        total_weight = sum(weight_list) or 1.0
        weight_list = [w / total_weight for w in weight_list]
        
        prob = sum(w * s for w, s in zip(weight_list, signals))
        prob = prob.clip(0.0, 1.0)
        
        # Prepare for Kelly calculation
        bets_df = pd.DataFrame({
            'match_id': market_df['match_id'],
            'unified_market_type': market_df.get('unified_market_type', 
                                               market_df.get('market_type')),
            'bet_name': market_df['bet_name'],
            'odds': market_df['decimal_odds'],
            'probability': prob
        })
        
        # Calculate stakes
        stakes = calculate_kelly_stakes_with_exclusivity(
            bets_df,
            bankroll=bankroll,
            correlation_matrix=spec.get('correlation_matrix'),
            risk_adjusted=spec.get('risk_adjusted', True),
            max_stake_per_bet=spec.get('max_stake_per_bet')
        )
        
        # Build performance DataFrame
        df = pd.DataFrame({
            'strategy_name': name,
            'source_id': market_df['source_id'].values,
            'total_staked': stakes,
            'result_multiplier': market_df.get('result_multiplier', 0.0).fillna(0.0).values
        })
        
        df = df[df['total_staked'] > 0].copy()
        
        if not df.empty:
            df['net'] = df['total_staked'] * df['result_multiplier'] - df['total_staked']
            df['roi'] = df['net'] / df['total_staked']
            all_perf.append(df)
    
    if not all_perf:
        return pd.DataFrame()
        
    return pd.concat(all_perf, ignore_index=True)


def run_cached_backtest_with_chunks(
    strategies: list,
    min_break_minutes: float = 60.0,
    avg_game_duration_minutes: float = 120.0,
    use_batch_loading: bool = True,
    save_results: bool = True,
    parallel: bool = False,
    n_workers: Optional[int] = None
):
    """
    Run optimized backtest with caching across multiple time windows.
    
    Args:
        strategies: List of strategy specifications
        min_break_minutes: Minimum break between sessions
        avg_game_duration_minutes: Average game duration
        use_batch_loading: Whether to load all data upfront
        save_results: Whether to save results to disk
        parallel: Whether to use parallel processing
        n_workers: Number of parallel workers (default: CPU count)
    """
    from backtest import compute_backtest_as_of_list
    import multiprocessing as mp
    
    # Get time windows
    as_of_pairs = compute_backtest_as_of_list(
        min_break_minutes=min_break_minutes,
        avg_game_duration_minutes=avg_game_duration_minutes
    )
    
    if not as_of_pairs:
        logger.error("No backtest windows found")
        return None
        
    # Initialize components
    window_cache = SlidingWindowCache()
    signal_cache = SignalCache()
    batch_loader = None
    
    # Batch loading mode
    if use_batch_loading:
        batch_loader = BatchDataLoader()
        start_ts, _ = as_of_pairs[0]
        _, end_ts = as_of_pairs[-1]
        batch_loader.load_period(start_ts, end_ts)
        logger.info("Batch loading complete")
    
    # Setup for results
    runtime = datetime.now(timezone.utc)
    backtest_id = f"cached_{runtime.strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Running cached backtest: {backtest_id}")
    logger.info(f"Processing {len(as_of_pairs)} windows")
    
    # Process windows
    if parallel and len(as_of_pairs) > 1:
        # Parallel processing
        n_workers = n_workers or mp.cpu_count()
        logger.info(f"Using {n_workers} parallel workers")
        
        # Split windows into chunks for workers
        chunk_size = max(1, len(as_of_pairs) // n_workers)
        window_chunks = [
            as_of_pairs[i:i + chunk_size] 
            for i in range(0, len(as_of_pairs), chunk_size)
        ]
        
        # Process in parallel
        with mp.Pool(n_workers) as pool:
            chunk_results = pool.starmap(
                process_window_chunk,
                [(chunk, strategies, batch_loader, i) 
                 for i, chunk in enumerate(window_chunks)]
            )
            
        all_results = [r for chunk in chunk_results for r in chunk]
        
    else:
        # Sequential processing
        all_results = []
        
        for i, (as_of, until) in enumerate(as_of_pairs):
            logger.info(f"[{i+1}/{len(as_of_pairs)}] Processing {as_of} to {until}")
            
            try:
                perf_df = cached_vectorized_backtest(
                    strategies=strategies,
                    as_of=as_of,
                    until=until,
                    window_cache=window_cache,
                    signal_cache=signal_cache,
                    batch_loader=batch_loader,
                    require_outcome=True
                )
                
                if not perf_df.empty:
                    perf_df['as_of'] = as_of
                    perf_df['until'] = until
                    perf_df['backtest_id'] = backtest_id
                    all_results.append(perf_df)
                    
            except Exception as e:
                logger.error(f"Error in window {as_of}: {e}")
                continue
    
    # Save signal cache
    signal_cache.save_to_disk()
    
    if not all_results:
        logger.warning("No results generated")
        return None
        
    # Combine results
    full_results = pd.concat(all_results, ignore_index=True)
    
    # Summary
    logger.info("\n=== BACKTEST COMPLETE ===")
    summary = full_results.groupby('strategy_name').agg({
        'total_staked': 'sum',
        'net': 'sum', 
        'roi': 'mean'
    })
    logger.info(f"\n{summary}")
    
    # Save if requested
    if save_results:
        output_dir = Path(f"backtests/{backtest_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        full_results.to_csv(output_dir / "results.csv", index=False)
        summary.to_csv(output_dir / "summary.csv")
        
        logger.info(f"Results saved to {output_dir}")
        
    return full_results


def process_window_chunk(window_pairs, strategies, batch_loader, worker_id):
    """
    Process a chunk of windows (for parallel processing).
    """
    # Each worker needs its own cache instances
    window_cache = SlidingWindowCache()
    signal_cache = SignalCache()
    
    results = []
    for as_of, until in window_pairs:
        try:
            perf_df = cached_vectorized_backtest(
                strategies=strategies,
                as_of=as_of,
                until=until,
                window_cache=window_cache,
                signal_cache=signal_cache,
                batch_loader=batch_loader,
                require_outcome=True
            )
            if not perf_df.empty:
                results.append(perf_df)
        except Exception as e:
            logger.error(f"Worker {worker_id} error at {as_of}: {e}")
            
    return results


if __name__ == "__main__":
    # Example usage
    from signals import SIGNAL_PROVIDERS, SIGNAL_WEIGHTS
    
    # Strategy configuration
    test_strategy = {
        "name": "cached_test",
        "providers": SIGNAL_PROVIDERS,
        "weights": SIGNAL_WEIGHTS,
        "bankroll": 1000,
        "correlation_matrix": None,
        "risk_adjusted": True,
        "max_stake_per_bet": None
    }
    
    # Run cached backtest
    results = run_cached_backtest_with_chunks(
        strategies=[test_strategy],
        use_batch_loading=True,
        parallel=True
    )