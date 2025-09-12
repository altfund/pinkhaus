#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 02:19:02 2025

@author: ess
"""

from datetime import datetime
from evaluate_open_markets import *

# backtest_utils.py

import pandas as pd


from signals import *

from database_utils import fetch_odds_window_optimized, fetch_markets_in_window

from performance import (
    summarize_backtest_performance,
)

from evaluate_open_markets import (
    summarize_match_schedule_from_open_markets,
    find_upcoming_game_breaks,
    extract_active_game_periods_from_breaks,
)
from typing import Optional, List, Tuple

import sys
import grpc

print("RUNTIME PYTHON:", sys.executable)
print("RUNTIME grpcio version:", grpc.__version__)


def _load_data_windows() -> Tuple[
    Optional[pd.Timestamp],  # earliest_odds_update
    Optional[pd.Timestamp],  # latest_odds_update
    Optional[pd.Timestamp],  # earliest_maturity_date
    Optional[pd.Timestamp],  # latest_maturity_date
]:
    """
    Fetches both:
      • the earliest & latest odd.updated_at, and
      • the earliest & latest market.maturity_date
    for Soccer / winner on overtime_markets.

    All four are normalized to UTC‐aware pandas Timestamps.
    If no data is present, any missing bound becomes None.
    """
    # Use ORM-based function that handles the query more efficiently
    earliest, latest = fetch_odds_window_optimized(
        sport="Soccer", market_type="winner", bookmaker="overtime_markets"
    )

    # The optimized function returns combined windows, so we need to get individual components
    # For now, return the same values for odds and maturity windows
    return (earliest, latest, earliest, latest)


def _load_odds_window() -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Returns a combined backtest window for Soccer 'winner' markets:
      • earliest = min(earliest odds update, earliest market maturity)
      • latest   = max(latest odds update,   latest market maturity)
    All returned as UTC‐aware pandas Timestamps, or (None, None) if no data.
    """
    # Pull both odds and maturity windows
    earliest_odds, latest_odds, earliest_mat, latest_mat = _load_data_windows()

    # Gather non‐None candidates
    start_candidates = [ts for ts in (earliest_odds, earliest_mat) if ts is not None]
    end_candidates = [ts for ts in (latest_odds, latest_mat) if ts is not None]

    if not start_candidates or not end_candidates:
        # No usable data
        return None, None

    # Determine combined window
    earliest = min(start_candidates)
    latest = max(end_candidates)
    return earliest, latest


def _to_utc(dt: pd.Timestamp) -> pd.Timestamp:
    """Ensure tz-aware UTC timestamp."""
    if dt.tzinfo is None:
        return dt.tz_localize("UTC")
    else:
        return dt.tz_convert("UTC")


def _to_naive(dt_utc: pd.Timestamp) -> datetime:
    """Strip tz so we can bind to SQL BETWEEN."""
    return dt_utc.tz_convert("UTC").tz_localize(None).to_pydatetime()


def _load_markets(start_naive: datetime, end_naive: datetime) -> pd.DataFrame:
    return fetch_markets_in_window(start_naive, end_naive)


def _compute_chunks(
    raw: pd.DataFrame,
    start_utc: pd.Timestamp,
    min_break_minutes: float,
    avg_game_duration_minutes: float,
) -> pd.DataFrame:
    match_df = summarize_match_schedule_from_open_markets(raw)
    match_df["maturity_date"] = pd.to_datetime(match_df["maturity_date"], utc=True)

    breaks_df = find_upcoming_game_breaks(
        match_df,
        min_break_minutes=min_break_minutes,
        avg_game_duration_minutes=avg_game_duration_minutes,
        now=start_utc,
    )
    chunk_df = extract_active_game_periods_from_breaks(
        match_df, breaks_df, avg_game_duration_minutes=avg_game_duration_minutes
    )

    # keep only those chunks that start before our window’s end
    chunk_df["chunk_start"] = pd.to_datetime(chunk_df["chunk_start"], utc=True)
    chunk_df["chunk_end"] = pd.to_datetime(chunk_df["chunk_end"], utc=True)
    return chunk_df


def _compute_midpoints(
    chunk_df: pd.DataFrame,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """For each adjacent pair of chunks, return (midpoint, this_chunk_end)."""
    pairs = []
    for i in range(1, len(chunk_df)):
        prev_end = chunk_df.loc[i - 1, "chunk_end"]
        this_start = chunk_df.loc[i, "chunk_start"]
        this_end = chunk_df.loc[i, "chunk_end"]
        midpoint = prev_end + 0.5 * (this_start - prev_end)
        pairs.append((midpoint, this_end))
    return pairs


def compute_backtest_as_of_list(
    min_break_minutes: float = 300.0,
    avg_game_duration_minutes: float = 120.0,
) -> List[Tuple[datetime, datetime]]:
    """
    Compute unique, sorted (as_of, chunk_end) pairs:
      • as_of at 50% of each break,
      • one per chunk whose start falls in our odds‐data window.
    """
    # 1) load earliest/latest odds
    start_ts, end_ts = _load_odds_window()
    if pd.isna(start_ts) or pd.isna(end_ts):
        print("🛑 No odds data!")
        return []

    # 2) normalize to UTC‐aware + prepare naive for SQL
    start_utc = _to_utc(start_ts)
    end_utc = _to_utc(end_ts)
    start_naive, end_naive = _to_naive(start_utc), _to_naive(end_utc)
    print(f"🔍 Backtest window (UTC‐aware): {start_utc} → {end_utc}")

    # 3) pull raw markets in that window
    raw = _load_markets(start_naive, end_naive)
    if raw.empty:
        print("🛑 No markets in window.")
        return []

    # 4) compute the game‐chunks
    chunk_df = _compute_chunks(
        raw, start_utc, min_break_minutes, avg_game_duration_minutes
    )
    # drop any chunks that start after our window end
    chunk_df = chunk_df[chunk_df["chunk_start"] < end_utc].reset_index(drop=True)
    if chunk_df.empty:
        print("🛑 All chunks start after data ends.")
        return []

    # 5) build all midpoint‐of‐break → chunk_end pairs
    raw_pairs = _compute_midpoints(chunk_df)

    # 6) dedupe by chunk_end (keep earliest midpoint)
    seen = {}
    for midpoint, chunk_end in sorted(raw_pairs, key=lambda x: (x[1], x[0])):
        if chunk_end not in seen:
            seen[chunk_end] = midpoint

    # 7) return sorted list
    result = [(seen[ce], ce) for ce in sorted(seen)]
    print(f"✅ Found {len(result)} backtest sessions.")
    return result


def main():
    """
    Legacy main function - now uses vectorized backtest.
    For the new implementation, see run_backtest.py
    """
    print("Note: This uses the legacy backtest implementation.")
    print(
        "For better performance, use run_backtest.py which uses the vectorized version."
    )
    print()

    from vectorized_backtest import run_vectorized_backtest_with_chunks

    # Create a strategy matching the old configuration
    legacy_strategy = {
        "name": "implied_kelly+random",
        "providers": SIGNAL_PROVIDERS,
        "weights": SIGNAL_WEIGHTS,
        "bankroll": 1000,
        "correlation_matrix": None,
        "risk_adjusted": True,
        "max_stake_per_bet": None,
    }

    # Run vectorized backtest
    results = run_vectorized_backtest_with_chunks(
        strategies=[legacy_strategy],
        min_break_minutes=60.0,
        avg_game_duration_minutes=120.0,
        save_results=True,
    )

    if results is not None:
        # Summarize performance
        strat_name = legacy_strategy["name"]
        summarize_backtest_performance(strat_name=strat_name, print_per_session=True)


if __name__ == "__main__":
    main()


# make market primary key consist of:
# game_id/source_id
# type_id
# line (0 if none)
# player (0 if none)
# position (odds position in return list)

# make trade/bet/wager table
# connect to market table
# if backtest just record and save source and any other data needed
# if paper trade get quote from overtime and record it
# if live, get quote and then execute trade

# make backtest subfolders
# iterate chunks based on end dates of prior backtest betting reports
# determine dates to backtest within (R&D, out of sample, etc.)

# compose multiple backtests into a weight on them based on performance, correlation etc. per Carver
#
