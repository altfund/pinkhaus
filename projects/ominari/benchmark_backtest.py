#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark script to compare original vs cached backtest performance.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('../pinkhaus-models'))

import time
import pandas as pd
from datetime import datetime
import logging
import psutil
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_original_backtest(strategies, n_windows=5):
    """
    Benchmark the original vectorized backtest.
    """
    from vectorized_backtest import vectorized_backtest
    from evaluate_open_markets import fetch_open_markets_for_as_of
    
    logger.info("=" * 60)
    logger.info("BENCHMARKING ORIGINAL VECTORIZED BACKTEST")
    logger.info("=" * 60)
    
    # Use recent timestamps to minimize data
    end_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=24)
    
    results = []
    total_time = 0
    memory_start = get_memory_usage()
    
    for i in range(n_windows):
        # Create time window
        window_end = end_time - pd.Timedelta(hours=i * 2)
        window_start = window_end - pd.Timedelta(hours=1)
        
        logger.info(f"\nWindow {i+1}/{n_windows}: {window_start} to {window_end}")
        
        # Time the fetch
        fetch_start = time.time()
        try:
            markets = fetch_open_markets_for_as_of(window_start)
            fetch_time = time.time() - fetch_start
            logger.info(f"  Fetch time: {fetch_time:.2f}s ({len(markets)} markets)")
        except Exception as e:
            logger.error(f"  Fetch failed: {e}")
            continue
            
        # Time the backtest
        backtest_start = time.time()
        try:
            result = vectorized_backtest(
                strategies=strategies,
                as_of=window_start,
                until=window_end,
                require_outcome=False
            )
            backtest_time = time.time() - backtest_start
            
            window_time = fetch_time + backtest_time
            total_time += window_time
            
            logger.info(f"  Backtest time: {backtest_time:.2f}s")
            logger.info(f"  Total window time: {window_time:.2f}s")
            logger.info(f"  Results: {len(result) if result is not None else 0} bets")
            
            results.append({
                'window': i + 1,
                'fetch_time': fetch_time,
                'backtest_time': backtest_time,
                'total_time': window_time,
                'n_markets': len(markets),
                'n_bets': len(result) if result is not None else 0
            })
            
        except Exception as e:
            logger.error(f"  Backtest failed: {e}")
            continue
    
    memory_end = get_memory_usage()
    memory_used = memory_end - memory_start
    
    logger.info("\nOriginal Backtest Summary:")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Average per window: {total_time/n_windows:.2f}s")
    logger.info(f"  Memory used: {memory_used:.2f} MB")
    
    return pd.DataFrame(results), total_time


def benchmark_cached_backtest(strategies, n_windows=5):
    """
    Benchmark the cached/optimized backtest.
    """
    from cached_vectorized_backtest import (
        cached_vectorized_backtest, 
        SlidingWindowCache,
        SignalCache
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARKING CACHED VECTORIZED BACKTEST")
    logger.info("=" * 60)
    
    # Initialize components
    window_cache = SlidingWindowCache()
    signal_cache = SignalCache()
    
    # For fair comparison, don't use batch loading in basic benchmark
    batch_loader = None
    
    # Use same time windows as original
    end_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=24)
    
    results = []
    total_time = 0
    memory_start = get_memory_usage()
    
    for i in range(n_windows):
        window_end = end_time - pd.Timedelta(hours=i * 2)
        window_start = window_end - pd.Timedelta(hours=1)
        
        logger.info(f"\nWindow {i+1}/{n_windows}: {window_start} to {window_end}")
        
        # Time the entire cached backtest
        start_time = time.time()
        try:
            result = cached_vectorized_backtest(
                strategies=strategies,
                as_of=window_start,
                until=window_end,
                window_cache=window_cache,
                signal_cache=signal_cache,
                batch_loader=batch_loader,
                require_outcome=False
            )
            
            window_time = time.time() - start_time
            total_time += window_time
            
            logger.info(f"  Total window time: {window_time:.2f}s")
            logger.info(f"  Results: {len(result)} bets")
            logger.info(f"  Cache size: {len(window_cache.markets_cache)} markets")
            
            results.append({
                'window': i + 1,
                'total_time': window_time,
                'n_bets': len(result),
                'cache_size': len(window_cache.markets_cache)
            })
            
        except Exception as e:
            logger.error(f"  Cached backtest failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    memory_end = get_memory_usage()
    memory_used = memory_end - memory_start
    
    logger.info("\nCached Backtest Summary:")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Average per window: {total_time/n_windows:.2f}s")
    logger.info(f"  Memory used: {memory_used:.2f} MB")
    logger.info(f"  Final cache size: {len(window_cache.markets_cache)} markets")
    
    return pd.DataFrame(results), total_time


def benchmark_batch_loading(strategies, n_windows=5):
    """
    Benchmark cached backtest with batch data loading.
    """
    from cached_vectorized_backtest import (
        cached_vectorized_backtest,
        SlidingWindowCache,
        SignalCache, 
        BatchDataLoader
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARKING BATCH-LOADED CACHED BACKTEST")
    logger.info("=" * 60)
    
    # Time windows
    end_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=24)
    start_time = end_time - pd.Timedelta(hours=n_windows * 2 + 2)
    
    # Batch loading
    batch_start = time.time()
    batch_loader = BatchDataLoader()
    
    try:
        batch_loader.load_period(start_time, end_time, chunk_size=10000)
        batch_time = time.time() - batch_start
        logger.info(f"Batch loading completed in {batch_time:.2f}s")
    except Exception as e:
        logger.error(f"Batch loading failed: {e}")
        return pd.DataFrame(), 0
    
    # Run cached backtest with batch data
    window_cache = SlidingWindowCache()
    signal_cache = SignalCache()
    
    results = []
    total_time = batch_time
    memory_start = get_memory_usage()
    
    for i in range(n_windows):
        window_end = end_time - pd.Timedelta(hours=i * 2)
        window_start = window_end - pd.Timedelta(hours=1)
        
        logger.info(f"\nWindow {i+1}/{n_windows}: {window_start} to {window_end}")
        
        start = time.time()
        try:
            result = cached_vectorized_backtest(
                strategies=strategies,
                as_of=window_start,
                until=window_end,
                window_cache=window_cache,
                signal_cache=signal_cache,
                batch_loader=batch_loader,
                require_outcome=False
            )
            
            window_time = time.time() - start
            total_time += window_time
            
            logger.info(f"  Window time: {window_time:.2f}s")
            logger.info(f"  Results: {len(result)} bets")
            
            results.append({
                'window': i + 1,
                'total_time': window_time,
                'n_bets': len(result)
            })
            
        except Exception as e:
            logger.error(f"  Batch backtest failed: {e}")
            continue
    
    memory_end = get_memory_usage()
    memory_used = memory_end - memory_start
    
    logger.info("\nBatch-Loaded Backtest Summary:")
    logger.info(f"  Batch load time: {batch_time:.2f}s")
    logger.info(f"  Backtest time: {total_time - batch_time:.2f}s")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Average per window: {(total_time - batch_time)/n_windows:.2f}s")
    logger.info(f"  Memory used: {memory_used:.2f} MB")
    
    return pd.DataFrame(results), total_time


def main():
    """
    Run comprehensive benchmark comparison.
    """
    from signals import ImpliedRawSignal
    
    # Simple test strategy
    test_strategy = {
        "name": "benchmark_test",
        "providers": [ImpliedRawSignal()],
        "weights": [1.0],
        "bankroll": 1000,
        "correlation_matrix": None,
        "risk_adjusted": True,
        "max_stake_per_bet": None
    }
    
    strategies = [test_strategy]
    n_windows = 3  # Small number for testing
    
    logger.info("Starting backtest benchmark comparison...")
    logger.info(f"Testing with {n_windows} time windows")
    logger.info("Database: sport_odds.db (201 GB)")
    
    # Collect results
    benchmark_results = {}
    
    # 1. Original implementation
    try:
        gc.collect()
        orig_df, orig_time = benchmark_original_backtest(strategies, n_windows)
        benchmark_results['original'] = {
            'time': orig_time,
            'details': orig_df
        }
    except Exception as e:
        logger.error(f"Original benchmark failed: {e}")
        benchmark_results['original'] = {'time': float('inf'), 'error': str(e)}
    
    # 2. Cached implementation
    try:
        gc.collect()
        cached_df, cached_time = benchmark_cached_backtest(strategies, n_windows)
        benchmark_results['cached'] = {
            'time': cached_time,
            'details': cached_df
        }
    except Exception as e:
        logger.error(f"Cached benchmark failed: {e}")
        benchmark_results['cached'] = {'time': float('inf'), 'error': str(e)}
    
    # 3. Batch-loaded implementation
    try:
        gc.collect()
        batch_df, batch_time = benchmark_batch_loading(strategies, n_windows)
        benchmark_results['batch'] = {
            'time': batch_time,
            'details': batch_df
        }
    except Exception as e:
        logger.error(f"Batch benchmark failed: {e}")
        benchmark_results['batch'] = {'time': float('inf'), 'error': str(e)}
    
    # Summary comparison
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK COMPARISON SUMMARY")
    logger.info("=" * 60)
    
    if 'original' in benchmark_results and 'time' in benchmark_results['original']:
        base_time = benchmark_results['original']['time']
        
        for method, results in benchmark_results.items():
            if 'time' in results and results['time'] != float('inf'):
                speedup = base_time / results['time'] if results['time'] > 0 else 0
                logger.info(f"{method:12s}: {results['time']:8.2f}s (speedup: {speedup:.2f}x)")
            else:
                logger.info(f"{method:12s}: FAILED")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for method, results in benchmark_results.items():
        if 'details' in results and isinstance(results['details'], pd.DataFrame):
            results['details'].to_csv(f"benchmark_{method}_{timestamp}.csv", index=False)
    
    logger.info(f"\nDetailed results saved to benchmark_*_{timestamp}.csv")


if __name__ == "__main__":
    main()