#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 02:19:02 2025

@author: ess
"""

from datetime import datetime, timezone
from evaluate_open_markets import *
from evaluate_open_markets import _init_external_stub

_external_stub = _init_external_stub()

# backtest_utils.py

import pandas as pd

from sqlalchemy import text

from evaluate_open_markets import (
    generate_betting_session_report_and_save,
    SIGNAL_PROVIDERS,
    SIGNAL_WEIGHTS,
)

from database import engine

from performance import (
    summarize_backtest_performance,
)

from evaluate_open_markets import (
    summarize_match_schedule_from_open_markets,
    find_upcoming_game_breaks,
    extract_active_game_periods_from_breaks,
)
from typing import Optional, List, Tuple


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
    sql = text("""
      SELECT
        MIN(o.updated_at)    AS earliest_odds,
        MAX(o.updated_at)    AS latest_odds,
        MIN(m.maturity_date) AS earliest_mat,
        MAX(m.maturity_date) AS latest_mat
      FROM odd o
      JOIN market m ON m.source_id = o.source_id
      WHERE o.bookmaker   = 'overtime_markets'
        AND o.market_type = 'winner'
        AND m.sport       = 'Soccer'
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(
            sql,
            conn,
            parse_dates=["earliest_odds", "latest_odds", "earliest_mat", "latest_mat"],
        )

    row = df.iloc[0]

    def _to_utc(ts: pd.Timestamp) -> Optional[pd.Timestamp]:
        if pd.isna(ts):
            return None
        ts = pd.to_datetime(ts)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        else:
            return ts.tz_convert("UTC")

    return (
        _to_utc(row["earliest_odds"]),
        _to_utc(row["latest_odds"]),
        _to_utc(row["earliest_mat"]),
        _to_utc(row["latest_mat"]),
    )


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
    sql = """
      SELECT maturity_date, home_team, away_team
      FROM market
      WHERE maturity_date BETWEEN :start AND :end
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(
            sql,
            conn,
            params={"start": start_naive, "end": end_naive},
            parse_dates=["maturity_date"],
        )
    if not df.empty:
        df["maturity_date"] = pd.to_datetime(df["maturity_date"], utc=True)
    return df


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


SIGNAL_PROVIDERS = [
    ImpliedRawSignal(),
    ExternalGrpcSignal(_external_stub),
]

SIGNAL_WEIGHTS = {
    "implied_raw": 1.0,  # base implied probability
    "external": 1.0,  # weight for external model
}


def main():
    min_break_minutes = 1 * 60.0
    avg_game_duration_minutes = 120.0  # used only for chunking
    abs_game_limit = None
    base_strat = "implied_kelly+random"

    # 1) Compute all (as_of, chunk_end) pairs
    as_of_pairs = compute_backtest_as_of_list(
        min_break_minutes=min_break_minutes,
        avg_game_duration_minutes=avg_game_duration_minutes,
    )
    if not as_of_pairs:
        print("No backtest chunks found; exiting.")
        return

    # record the runtime for unique backtest ID
    runtime = datetime.now(timezone.utc)

    # 2) Derive dynamic strat_name including runtime
    start_ts, _ = as_of_pairs[0]
    end_ts, _ = as_of_pairs[-1]
    strat_name = (
        f"{base_strat}_"
        f"{runtime.strftime('%Y%m%d_%H%M%S')}_"
        f"start={start_ts.strftime('%Y%m%d_%H%M')}_"
        f"end={end_ts.strftime('%Y%m%d_%H%M')}_"
        f"break={min_break_minutes}_"
        f"games={avg_game_duration_minutes}"
    )
    print(f"Using strategy name/backtest ID: {strat_name}")

    # 3) Run each session and save bets
    for as_of, chunk_end in as_of_pairs:
        print(f"\n=== Running session @ {as_of.isoformat()} ===")
        generate_betting_session_report_and_save(
            execution_bankroll=1000,
            avg_game_duration_minutes=avg_game_duration_minutes,
            min_break_minutes=min_break_minutes,
            abs_game_limit=abs_game_limit,
            base_dir="backtests/" + strat_name,
            signal_providers=SIGNAL_PROVIDERS,
            signal_weights=SIGNAL_WEIGHTS,
            as_of=as_of,
            mode="backtest",
            strat=strat_name,
            window_end=chunk_end,
        )

    print(f"=== Completed all sessions for: {strat_name} ===")
    # Now summarize by strat_name directly
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
