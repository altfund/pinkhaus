#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:51:44 2025

@author: ess
"""
import pandas as pd

from odds_formatting_helpers import *
from kelly_multimarket import *

DB_NAME = "sport_odds.db"

import pandas as pd
import grpc
import time

import sqlite3
import pandas as pd
from odds_formatting_helpers import (
    normalize_outcome,
    normalize_team_outcome,
    add_implied_probabilities,
    generate_internal_mispricing_signals
)
from datetime import datetime, timezone

# ─── 1. SIGNAL ABSTRACTION ─────────────────────────────────────────────────────

class SignalProvider:
    """Base class: must return a pd.Series of probabilities (0–1)."""
    name: str
    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class ImpliedRawSignal(SignalProvider):
    name = "implied_raw"
    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        # assumes df["implied_raw"] is in percent
        return df["implied_raw"].astype(float).div(100.0)


class ExternalGrpcSignal(SignalProvider):
    name = "external"
    def __init__(self, stub, timeout: float = 2.0):
        self.stub = stub
        self.timeout = timeout

    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series([], index=df.index)
        from external_pb2 import SignalBatchRequest
        from external_pb2_grpc import SignalServiceStub
        import grpc

        # Build one batch request
        req = SignalBatchRequest()
        for _, row in df.iterrows():
            r = req.requests.add()
            r.source_id          = row["source_id"]
            r.normalized_outcome = row["normalized_outcome"]
            r.as_of_time         = row["time"].isoformat()

        try:
            print(f"[CLIENT] Calling RPC for {len(df)} rows...")
            t0 = time.time()
            resp = self.stub.GetProbabilities(req, timeout=self.timeout)
            probs = list(resp.probabilities)
            print(f"[CLIENT] RPC returned in {time.time()-t0:.2f}s")
        except grpc.RpcError as e:
            # on any error (timeout, unavailable, etc.) return zeros
            probs = [0.0] * len(df)

        # Return a properly indexed Series
        return pd.Series(probs, index=df.index)


def aggregate_signals(
    df: pd.DataFrame,
    providers: list[SignalProvider],
    weights: dict[str, float],
) -> pd.DataFrame:
    """
    For each provider:
      - compute a prob series
      - store it in df[provider.name]
      - store its weighted contribution in df[f"contrib_{name}"]
    Then set df["probability"] = sum_i weight_i * df[name] / sum(weights).
    """
    df = df.copy()
    total_weight = sum(weights.values()) or 1.0

    for provider in providers:
        w = weights.get(provider.name, 0.0)
        probs = provider.get_probs(df).fillna(0.0)
        df[provider.name] = probs
        df[f"contrib_{provider.name}"] = probs.mul(w).div(total_weight)

    # final blended probability
    contrib_cols = [f"contrib_{p.name}" for p in providers]
    df["probability"] = df[contrib_cols].sum(axis=1)

    return df


# ─── 2. CONFIGURE SIGNALS & WIRING INTO prepare_kelly_input ─────────────────
# Initialize external gRPC stub once at module load
# in evaluate_open_markets.py, near top:
def _init_external_stub():
    # for local testing, use your dummy server:
    channel = grpc.insecure_channel("localhost:50051")
    return __import__("external_pb2_grpc").SignalServiceStub(channel)

_external_stub = _init_external_stub()


SIGNAL_PROVIDERS = [
    ImpliedRawSignal(),
    ExternalGrpcSignal(_external_stub),
]

SIGNAL_WEIGHTS = {
    "implied_raw": 1.0,    # base implied probability
    "external":    1.0,    # weight for external model
}


def get_upcoming_overtime_markets_with_signals(
        limit: int = 10,
        as_of: datetime = None,
        min_break_minutes=60,
        avg_game_duration_minutes=180        
    ) -> pd.DataFrame:
    """
    Fetches all upcoming Overtime markets that have odds available,
    calculates implied probabilities and evaluates internal mispricings.

    Args:
        limit (int): Max number of rows to return for preview; 0 = no limit.

    Returns:
        pd.DataFrame: Summary with signal flags and edge metrics.
    """
    
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    
    # Step 1: Query upcoming Overtime market odds with necessary fields
    query = """
        SELECT 
            o.updated_at AS time,
            o.line AS line,
            o.decimal_odds AS odds,
            o.outcome,
            o.source,
            o.bookmaker,
            o.market_type,
            o.source_id AS source_id,
            m.source_id AS market_source_id,
            m.maturity_date,
            m.home_team,
            m.away_team,
            m.league_name,
            m.sport
        FROM odd o
        JOIN market m ON o.source_id = m.source_id
        WHERE 
            o.bookmaker = 'overtime_markets' AND
            m.sport = 'Soccer' AND
            o.market_type = 'winner'
        ORDER BY m.maturity_date ASC
    """
    with sqlite3.connect(DB_NAME) as conn:
        overtime_odds = pd.read_sql_query(query, conn)

    # Step 2: Normalize outcomes and lines
    overtime_odds["normalized_outcome"] = overtime_odds.apply(
        lambda row: normalize_outcome(row["bookmaker"], row["market_type"], row["outcome"]),
        axis=1
    )
    overtime_odds["normalized_outcome"] = overtime_odds.apply(normalize_team_outcome, axis=1)
    overtime_odds["normalized_line"] = pd.to_numeric(overtime_odds["line"], errors="coerce").fillna(0.0)

    # Step 3: Map market_type to unified type
    market_type_map = pd.DataFrame([
        {"bookmaker": "overtime_markets", "market_type": "winner", "unified_market_type": "h2h"},
        {"bookmaker": "overtime_markets", "market_type": "spread", "unified_market_type": "spread"},
        {"bookmaker": "overtime_markets", "market_type": "total", "unified_market_type": "total"},
    ])
    merged_odds = pd.merge(
        overtime_odds,
        market_type_map,
        how="inner",
        on=["bookmaker", "market_type"]
    )

    # Step 4: Compute implied probabilities
    # ensure all times are parsed as UTC-aware
    merged_odds["time"] = pd.to_datetime(
        merged_odds["time"], errors="coerce", utc=True
    )
    # and also maturity_date
    merged_odds["maturity_date"] = pd.to_datetime(
        merged_odds["maturity_date"], errors="coerce", utc=True
    )
    
    # filter in pandas using our as_of
    merged_odds = merged_odds[merged_odds["maturity_date"] >= as_of]
    
    # ──────────────── 3) Build match schedule & find breaks ────────────────
    # summarize markets into per-match rows
    match_df = summarize_match_schedule_from_open_markets(merged_odds)
    # find all upcoming breaks between games, as of our timestamp
    breaks_df = find_upcoming_game_breaks(
        match_df,
        min_break_minutes=min_break_minutes,
        avg_game_duration_minutes=avg_game_duration_minutes,
        now=as_of
    )

    # ────────── 4) Turn breaks into active game time windows ───────────
    # e.g. [(game1_start, game1_end), (game2_start, game2_end), …]
    active_periods = extract_active_game_periods_from_breaks(match_df, breaks_df)

    # ────────── 5) Keep only markets in the *first* upcoming game window ──
    if not active_periods.empty:
        start, end = active_periods[0]
        mask = (
            (merged_odds["maturity_date"] >= start) &
            (merged_odds["maturity_date"] <  end)
        )
        merged_odds = merged_odds.loc[mask]


    merged_odds = merged_odds.dropna(subset=["odds"])
    merged_odds = apply_overtime_fees(merged_odds)
    merged_odds = add_implied_probabilities(merged_odds)

    # Step 5: Get most recent odds per outcome-line-market
    merged_odds["time"] = pd.to_datetime(merged_odds["time"], errors="coerce")
    merged_odds = merged_odds.sort_values("time", ascending=False)
    latest_odds = merged_odds.drop_duplicates(
        subset=["unified_market_type", "normalized_outcome", "normalized_line", "source_id"]
    )

    # Step 6: Evaluate internal edge and generate signals
    result = filter_valid_internal_signals(generate_internal_mispricing_signals(latest_odds))
    
    # Make a synthetic match_id if it's not already there
    if "market_name" not in result.columns:
        result["market_name"] = (
            result["home_team"].astype(str) + "_vs_" + result["away_team"].astype(str)
        )

    if limit:
        result = result.head(limit)

    return result

def filter_valid_internal_signals(open_markets: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the internal signals DataFrame to remove invalid or unusable rows.

    Conditions:
    - odds must be non-zero
    - implied probabilities must be valid
    - other filters can be added as needed

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    df = open_markets.copy()

    # 1. Drop rows where odds are 0 or missing
    df = df[df["odds"].fillna(0) != 0]

    # Optional: Remove where implied_raw or normalized are invalid
    df = df[df["implied_raw"].notna() & df["implied_normalized"].notna()]

    return df


def prepare_kelly_input(open_markets: pd.DataFrame) -> pd.DataFrame:
    """
    Converts filtered open_markets to the format required for calculate_kelly_stakes_with_exclusivity.
    """
    df = open_markets.copy()

    # Make a synthetic match_id if it's not already there
    if "market_name" not in df.columns:
        df["market_name"] = (
            df["home_team"].astype(str) + "_vs_" + df["away_team"].astype(str)
        )

    # Create human-readable bet name
    def format_bet_name(row):
        if row["unified_market_type"] in ["total", "spread"]:
            return f"{row['normalized_outcome']} {row['normalized_line']}"
        else:
            return row["normalized_outcome"]

    df["bet_name"] = df.apply(format_bet_name, axis=1)

    # Choose your probability column: implied_normalized or implied_raw
    #df["probability"] = df["implied_raw"] / 100.0 # could also use implied_raw
    
    # EARLY EXIT: no markets → no work
    if df.empty:
        # Still return the same column layout (but 0 rows)
        return df[[
            "source_id","market_name","bet_name","league_name","unified_market_type",
            "normalized_outcome","normalized_line"
        ]].assign(
            odds=pd.Series(dtype=float),
            probability=pd.Series(dtype=float),
            **{f"contrib_{p.name}": pd.Series(dtype=float) for p in SIGNAL_PROVIDERS}
        )
    
    # now it's guaranteed non-empty
    
    df = aggregate_signals(df, SIGNAL_PROVIDERS, SIGNAL_WEIGHTS)
    
    df["odds"] = df["adjusted_odds"]

    return df[["source_id", "market_name", "bet_name", "league_name", "unified_market_type", 
               "normalized_outcome", "normalized_line", "odds", "probability", "bookmaker",
               *[f"contrib_{p.name}" for p in SIGNAL_PROVIDERS]
               ]]

def summarize_match_schedule_from_open_markets(open_markets: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the open_markets odds data into a unique match-level DataFrame suitable for scheduling analysis.

    Returns:
        pd.DataFrame with columns: ['match_id', 'maturity_date']
    """
    df = open_markets.copy()

    # Parse maturity_date and drop missing ones
    # parse as UTC-aware so grouping/joining stays consistent
    df["maturity_date"] = pd.to_datetime(
        df["maturity_date"], errors="coerce", utc=True
    )

    df = df.dropna(subset=["maturity_date"])

    # Create synthetic match_id if necessary
    if "match_id" not in df.columns:
        df["match_id"] = df["home_team"].astype(str) + "_vs_" + df["away_team"].astype(str)

    # Group to get earliest maturity_date for each match
    match_df = (
        df.groupby("match_id", as_index=False)["maturity_date"]
        .min()
        .sort_values("maturity_date")
        .reset_index(drop=True)
    )

    return match_df

# Re-import necessary modules after code execution environment reset
import pandas as pd
from datetime import datetime, timedelta, timezone


def find_upcoming_game_breaks(
    matches_df: pd.DataFrame,
    min_break_minutes: int = 60,
    avg_game_duration_minutes: int = 180,
    now: datetime = None
) -> pd.DataFrame:
    """
    Identifies breaks in upcoming matches where no games are scheduled for at least `min_break_minutes`.

    Args:
        matches_df (pd.DataFrame): Must contain a 'maturity_date' datetime column.
        min_break_minutes (int): Minimum time gap (in minutes) to consider as a break.
        avg_game_duration_minutes (int): Estimated duration of a game (for overlap handling).
        now (datetime): Optional override for "current time".

    Returns:
        pd.DataFrame: List of breaks with start/end times, duration, and game counts before/after.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    df = matches_df.copy()
    # 1) Parse & enforce UTC-aware timestamps
    df["maturity_date"] = pd.to_datetime(
        df["maturity_date"], errors="coerce", utc=True
    )
    
    # 2) Use an aware 'now' in UTC
    now = datetime.now(timezone.utc)
    
    # 3) Safe comparison
    df = df[df["maturity_date"] > now]

    df = df.sort_values("maturity_date").reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(columns=["break_start", "break_end", "duration_minutes", "games_before", "games_after"])

    # Calculate time between consecutive games
    df["next_game"] = df["maturity_date"].shift(-1)
    df["gap_minutes"] = (df["next_game"] - df["maturity_date"]).dt.total_seconds() / 60

    breaks = df[df["gap_minutes"] >= min_break_minutes].copy()

    # Construct break metadata
    break_info = []
    games_before_running = 0
    for idx, row in breaks.iterrows():
        break_start = row["maturity_date"] + timedelta(minutes=avg_game_duration_minutes)
        break_end = row["next_game"]
        duration = (break_end - break_start).total_seconds() / 60

        games_before = ((df["maturity_date"] < break_start).sum()) - games_before_running
        games_after = (df["maturity_date"] >= break_end).sum()

        break_info.append({
            "break_start": break_start,
            "break_end": break_end,
            "duration_minutes": duration,
            "games_before": games_before,
            "games_after": games_after
        })
        
        games_before_running += games_before

    break_df = pd.DataFrame(break_info)
    return break_df

def extract_active_game_periods_from_breaks(match_df: pd.DataFrame, break_df: pd.DataFrame, avg_game_duration_minutes: int = 120,) -> pd.DataFrame:
    """
    Given a list of games and the breaks between them, derive the active game periods ("chunks")
    with start/end times, number of games, and time span.

    Returns:
        pd.DataFrame: Summary of contiguous game periods with their stats
    """
    match_df = match_df.sort_values("maturity_date").reset_index(drop=True).copy()
    break_df = break_df.sort_values("break_start").reset_index(drop=True).copy()

    chunks = []
    last_idx = 0

    for i, row in break_df.iterrows():
        # Get all games before the next break_start
        chunk = match_df[
            (match_df["maturity_date"] >= match_df.iloc[last_idx]["maturity_date"]) &
            (match_df["maturity_date"] < row["break_start"])
        ]
        if not chunk.empty:
            chunks.append({
                "chunk_start": chunk["maturity_date"].min(),
                "chunk_end": chunk["maturity_date"].max() + timedelta(minutes=avg_game_duration_minutes),
                "duration_minutes": (chunk["maturity_date"].max()  + timedelta(minutes=avg_game_duration_minutes) - chunk["maturity_date"].min()).total_seconds() / 60,
                "num_games": len(chunk)
            })
        # Update pointer
        last_idx = match_df[match_df["maturity_date"] >= row["break_end"]].index.min()

    # Capture the final chunk after the last break, if any
    if last_idx is not None and last_idx < len(match_df):
        final_chunk = match_df.iloc[last_idx:]
        chunks.append({
            "chunk_start": final_chunk["maturity_date"].min(),
            "chunk_end": final_chunk["maturity_date"].max(),
            "duration_minutes": (final_chunk["maturity_date"].max() - final_chunk["maturity_date"].min()).total_seconds() / 60,
            "num_games": len(final_chunk)
        })

    chunk_df = pd.DataFrame(chunks)
    return chunk_df

def build_structural_correlation_matrix(
    bets_df,
    nested_corr_base: float = 0.95,
    nested_corr_decay: float = 0.05,
    opposite_total_corr: float = -10.0,
    opposite_h2h_corr: float = -10.0,
    total_h2h_corr: float = 0.3,
    spread_h2h_corr: float = 0.3,
) -> np.ndarray:
    """
    Builds a structural correlation matrix reflecting logical and empirical relationships between bets.

    Parameters are tunable for decay or strength of correlation assumptions.
    """
    n = len(bets_df)
    corr = np.eye(n)

    for i in range(n):
        for j in range(i+1, n):
            bi, bj = bets_df.iloc[i], bets_df.iloc[j]

            # Only correlate bets from the same match
            if bi["source_id"] != bj["source_id"]:
                continue

            type_i, type_j = bi["unified_market_type"], bj["unified_market_type"]
            outcome_i, outcome_j = bi["normalized_outcome"], bj["normalized_outcome"]
            line_i, line_j = bi.get("normalized_line", 0), bj.get("normalized_line", 0)

            # Total market logic
            if type_i == type_j == "total":
                if outcome_i == outcome_j:
                    corr[i, j] = corr[j, i] = nested_corr_base - nested_corr_decay * abs(line_i - line_j)
                elif sorted([outcome_i, outcome_j]) == ["Over", "Under"]:
                    corr[i, j] = corr[j, i] = opposite_total_corr

            # Spread market logic
            elif type_i == type_j == "spread":
                if outcome_i == outcome_j:
                    corr[i, j] = corr[j, i] = nested_corr_base - nested_corr_decay * abs(line_i - line_j)
                elif outcome_i != outcome_j:
                    corr[i, j] = corr[j, i] = -nested_corr_base

            # Total ↔ H2H loose positive correlation (e.g. Over + Home win)
            elif {"h2h", "total"} <= {type_i, type_j}:
                corr[i, j] = corr[j, i] = total_h2h_corr

            # Spread ↔ H2H loose positive correlation
            elif {"h2h", "spread"} <= {type_i, type_j}:
                corr[i, j] = corr[j, i] = spread_h2h_corr

            # Opposite outcomes in h2h (e.g. Home vs Away)
            elif type_i == type_j == "h2h" and outcome_i != outcome_j:
                corr[i, j] = corr[j, i] = opposite_h2h_corr
                
            corr[i, j] = corr[j, i] = np.clip(corr[j, i], -1.0, 1.0)


    return corr

import pandas as pd

# Simulated version of apply_overtime_fees
def apply_overtime_fees(df: pd.DataFrame, default_skew: float = 0.01) -> pd.DataFrame:
    # 1) Early exit on empty DataFrame
    if df.empty:
        return df

    # 2) Now it’s safe to reset index and proceed
    # ── 1. Reset to a clean 0..n-1 index so .loc[…]=scalar always works
    df = df.reset_index(drop=True).copy()

    # ── 2. Identify Overtime rows
    is_overtime = df["bookmaker"] == "overtime_markets"

    # ── 3. Assign fees (scalar assignment now safe)
    df.loc[is_overtime, "safebox_fee"]   = 0.02
    df.loc[is_overtime, "skew_fee"]      = default_skew
    df.loc[is_overtime, "total_fee_pct"] = (
        df.loc[is_overtime, "safebox_fee"] +
        df.loc[is_overtime, "skew_fee"]
    )

    # ── 4. Fill the non-overtime defaults
    df["safebox_fee"]   = df["safebox_fee"].fillna(0.0)
    df["skew_fee"]      = df["skew_fee"].fillna(0.0)
    df["total_fee_pct"] = df["total_fee_pct"].fillna(0.0)

    # ── 5. Recompute adjusted_odds & implied_raw
    df["adjusted_odds"] = df["odds"] / (1.0 + df["total_fee_pct"])
    df["implied_raw"]   = 100.0 / df["adjusted_odds"]

    return df


# Function to enrich trimmed results with fee info
def enrich_trimmed_results_with_fee_info(trimmed_results: pd.DataFrame) -> pd.DataFrame:
    enriched = apply_overtime_fees(trimmed_results)
    enriched["fee_pct"] = enriched["total_fee_pct"]
    enriched["execution_stake"] = enriched["stake"] * (1 + enriched["fee_pct"])
    enriched["fee_amount"] = enriched["stake"] * enriched["fee_pct"]
    # Adjusted odds already calculated in apply_overtime_fees
    return enriched


def calculate_expected_kelly_return(kelly_results: pd.DataFrame, correlation_matrix: np.ndarray) -> float:
    """
    Estimate the expected log return of the full betting portfolio based on Kelly stakes and a correlation matrix.

    Parameters:
    - kelly_results (pd.DataFrame): Must contain 'odds', 'probability', and 'stake_fraction'.
    - correlation_matrix (np.ndarray): Structural correlation matrix between bets.

    Returns:
    - float: Expected log return of the portfolio
    """
    odds = kelly_results["odds"].values
    probs = kelly_results["probability"].values
    stakes = kelly_results["stake_fraction"].values

    # Expected log returns per bet
    expected_returns = probs * np.log1p(stakes * (odds - 1))

    # Adjust for correlation risk if matrix is provided
    if correlation_matrix is None:
        correlation_matrix = build_structural_correlation_matrix(kelly_results)    
        portfolio_variance = stakes.T @ correlation_matrix @ stakes
        expected_total_return = np.sum(expected_returns) - 0.5 * portfolio_variance
    #else:
    #    expected_total_return = np.sum(expected_returns)

    return expected_total_return


def trim_kelly_results(
    kelly_df: pd.DataFrame,
    kelly_fraction: float = 0.25,
    bankroll: float = 1000.0,
    cap_per_game: float = 0.1,
    cap_per_bet: float = 0.05,
    cap_per_game_market: float = 0.07,
    min_bet_abs: float = 1.0,
    min_bet_pct: float = 0.005
) -> pd.DataFrame:
    """
    Applies risk controls and sizing rules to raw Kelly recommendations with logging for debugging.

    Parameters:
    - kelly_df: DataFrame with original kelly output including `stake_fraction` and `stake`
    - kelly_fraction: Scalar multiplier on recommended stake (e.g., 0.25 for quarter Kelly)
    - bankroll: Current bankroll in dollars
    - cap_per_game: Max % of bankroll allowed on a single match_id
    - cap_per_bet: Max % of bankroll allowed on a single bet
    - cap_per_game_market: Max % of bankroll allowed on a (match_id, unified_market_type)
    - min_bet_abs: Minimum dollar amount to place a bet
    - min_bet_pct: Minimum % of bankroll to place a bet

    Returns:
    - DataFrame with adjusted stakes and trim_reason notes
    """

    df = kelly_df.copy()
    df["original_stake_fraction"] = df["stake_fraction"]
    df["original_stake"] = df["stake"]
    df["stake_fraction"] *= kelly_fraction
    #df["stake_fraction"] = df["stake"] / bankroll
    df["trim_reason"] = ""

    print("Pre-trim stake stats:")
    print(df[["stake", "stake_fraction"]].describe())

    # Cap per bet
    over_bet_cap = df["stake_fraction"] > cap_per_bet
    df.loc[over_bet_cap, "stake_fraction"] = cap_per_bet
    df.loc[over_bet_cap, "trim_reason"] += "|cap_per_bet"

    # Cap per game (using match_id instead of source_id)
    per_game = df.groupby("source_id")["stake_fraction"].transform("sum")
    scale_game = (cap_per_game / per_game).clip(upper=1.0, lower=0.0).fillna(1.0)
    df["stake_fraction"] *= scale_game
    df.loc[per_game > cap_per_game, "trim_reason"] += "|cap_per_game"

    # Cap per game + market
    per_game_market = df.groupby(["source_id", "unified_market_type"])["stake_fraction"].transform("sum")
    scale_gm = (cap_per_game_market / per_game_market).clip(upper=1.0, lower=0.0).fillna(1.0)
    df["stake_fraction"] *= scale_gm
    df.loc[per_game_market > cap_per_game_market, "trim_reason"] += "|cap_per_game_market"

    # Convert back to absolute stake
    df["stake"] = df["stake_fraction"] * bankroll

    # Enforce minimum bet size
    min_bet_thresh = max(min_bet_abs, min_bet_pct * bankroll)
    too_small = df["stake"] < min_bet_thresh
    df.loc[too_small, "stake"] = 0
    df.loc[too_small, "stake_fraction"] = 0
    df.loc[too_small, "trim_reason"] += "|min_bet"
    
    display(df[["market_name", "odds", "probability", "original_stake"]].sort_values("original_stake", ascending=False))
    display(df[["market_name", "unified_market_type","normalized_outcome"]].sort_values("market_name", ascending=False))

    print(f"\nTrimmed to zero (min bet): {too_small.sum()} / {len(df)}")
    print("Total stake before trim:", df['original_stake_fraction'].sum())
    print("Total stake after trim:", df['stake_fraction'].sum())
    print("Trim reasons summary:")
    print(df["trim_reason"].value_counts())

    # Clean trim reason
    df["trim_reason"] = df["trim_reason"].str.strip("|")
    
    print("Final stake allocations:")
    print(df.groupby("source_id")["stake"].sum().sort_values(ascending=False))


    return df[[
        "source_id", "unified_market_type", "normalized_outcome", "normalized_line",
        "market_name", "odds", "probability", "stake", "stake_fraction", "trim_reason", 
        "original_stake_fraction", "original_stake", "league_name", "bookmaker"
    ]]



def generate_betting_session_report_and_save(
    kelly_bankroll: float = 1.0,
    execution_bankroll: float = 15.0,
    kelly_fraction: float = 0.25,
    cap_per_game: float = 0.25,
    cap_per_bet: float = 0.25,
    cap_per_game_market: float = 0.10,
    min_bet_abs: float = 0.5,
    min_bet_pct: float = 0.0025,
    abs_game_limit: int = None,
    as_of: datetime = None,
    min_break_minutes: float = 360.0,
    avg_game_duration_minutes: float = 180.0,
    base_dir: str = "betting_reports",
    display_md: bool = True,
    save_as_latest: bool = True
) -> dict:
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from pathlib import Path
    import json, pickle
    import pdfkit
    from markdown2 import markdown as md2
    from IPython.display import Markdown, display
    
    if as_of is None:
        as_of = datetime.now(timezone.utc)

    open_markets = get_upcoming_overtime_markets_with_signals(
        limit=abs_game_limit,
        as_of=as_of,
        min_break_minutes=min_break_minutes,
        avg_game_duration_minutes=avg_game_duration_minutes
        )

    match_df = summarize_match_schedule_from_open_markets(open_markets)
    breaks_df = find_upcoming_game_breaks(
        match_df,
        min_break_minutes=min_break_minutes,
        avg_game_duration_minutes=avg_game_duration_minutes,
        now=as_of
    )
    game_times_df = extract_active_game_periods_from_breaks(match_df, breaks_df)
    # EARLY EXIT if there are no games to report
    if game_times_df.empty:
        print("No upcoming games—skipping report generation.")
        # return a minimal dict so callers can still index into ["report"]
        return {
            "report": "",         # empty report
            "filename": None,     # or "" if you prefer
        }


    first_chunk = game_times_df.iloc[0]
    start, end = first_chunk["chunk_start"], first_chunk["chunk_end"]

    filtered = open_markets.copy()
    filtered["maturity_date"] = pd.to_datetime(filtered["maturity_date"], errors="coerce")
    filtered = filtered[(filtered["maturity_date"] >= start) & (filtered["maturity_date"] <= end)]

    kelly_ready = prepare_kelly_input(filtered)

    kelly_results = calculate_kelly_stakes_with_exclusivity(
        kelly_ready,
        bankroll=kelly_bankroll,
        correlation_matrix=build_structural_correlation_matrix(kelly_ready),
        risk_adjusted=True,
    )

    expected_return = calculate_expected_kelly_return(kelly_results, None)
    normalized_return = expected_return / kelly_results["stake"].sum()

    trimmed_kelly_results = trim_kelly_results(
        kelly_results,
        kelly_fraction=kelly_fraction,
        bankroll=execution_bankroll,
        cap_per_game=cap_per_game,
        cap_per_bet=cap_per_bet,
        cap_per_game_market=cap_per_game_market,
        min_bet_abs=min_bet_abs,
        min_bet_pct=min_bet_pct,
    )
    
    trimmed_kelly_results = enrich_trimmed_results_with_fee_info(trimmed_kelly_results)

    expected_trimmed_return = calculate_expected_kelly_return(trimmed_kelly_results, None)
    total_staked = trimmed_kelly_results["stake"].sum()
    normalized_return_trimmed = expected_trimmed_return / total_staked if total_staked > 0 else 0
    expected_multiplier = np.exp(expected_trimmed_return)
    kelly_volatility = np.std(np.log1p(trimmed_kelly_results["stake_fraction"] * (trimmed_kelly_results["odds"] - 1)))
    kelly_sharpe_ratio = expected_trimmed_return / kelly_volatility

    active_games = sorted(filtered["market_name"].str.extract(r"(.+?)_vs_.*")[0].dropna().unique())
    bets_to_make = trimmed_kelly_results[
        ["market_name", "unified_market_type", "normalized_outcome", "normalized_line",
         "stake", "execution_stake", "fee_amount", "fee_pct", "odds", "adjusted_odds"]
    ][trimmed_kelly_results["stake"] > 0]


    report = f"""
BETTING SESSION REPORT
=======================

Session Time Window: {start} to {end}
Duration: {(end - start)}

Parameters:
  - Kelly Bankroll (for % calc): {kelly_bankroll}
  - Execution Bankroll: {execution_bankroll}
  - Kelly Fraction: {kelly_fraction}
  - Cap per Game: {cap_per_game}
  - Cap per Bet: {cap_per_bet}
  - Cap per Game Market: {cap_per_game_market}
  - Min Bet (abs): {min_bet_abs}
  - Min Bet (%): {min_bet_pct}

Games Covered:
{', '.join(active_games)}

Expected Return (Log): {expected_trimmed_return:.4f}
Expected Multiplier: {expected_multiplier:.3f}x
Volatility: {kelly_volatility:.4f}
Sharpe Ratio: {kelly_sharpe_ratio:.4f}
Total Stake: {total_staked:.2f}

Bets to Place:
{bets_to_make.to_string(index=False)}
"""

    result = {
        "report": report,
        "kelly_input": kelly_ready,
        "kelly_results": kelly_results,
        "trimmed_results": trimmed_kelly_results,
        "match_df": match_df,
        "game_times_df": game_times_df,
        "open_markets": open_markets,
        "start": start,
        "end": end,
        "metrics": {
            "expected_return": expected_trimmed_return,
            "expected_multiplier": expected_multiplier,
            "sharpe_ratio": kelly_sharpe_ratio,
            "volatility": kelly_volatility,
            "total_staked": total_staked,
            "normalized_return": normalized_return_trimmed,
        },
    }

    # Save the report
    save_and_display_betting_report(result, base_dir=base_dir, display_md=display_md)

    # Optionally save as latest session
    if save_as_latest:
        latest_path = Path(base_dir) / "latest_session.pkl"
        with open(latest_path, "wb") as f:
            pickle.dump(result, f)

    return result


import os
from datetime import datetime
from pathlib import Path
import markdown
from markdown2 import markdown as md2
import pdfkit
from IPython.display import Markdown, display
import json
import pickle

def save_and_display_betting_report(report_data: dict, base_dir: str = "betting_reports", display_md: bool = True) -> dict:
    """
    Saves and displays the betting session report in multiple formats.

    Args:
        report_data (dict): The dictionary returned by generate_betting_session_report().
        base_dir (str): Directory where reports will be saved.
        display_md (bool): Whether to display the markdown version in notebook environments.

    Returns:
        dict: Paths to the saved report files.
    """
    # Prepare timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_dir = Path(base_dir) / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save markdown
    report_md_path = report_dir / "betting_session_report.md"
    with open(report_md_path, "w") as f:
        f.write(report_data["report"])

    # Convert to HTML
    report_html_path = report_dir / "betting_session_report.html"
    with open(report_html_path, "w") as f:
        f.write(md2(report_data["report"]))

    # Convert to PDF (requires wkhtmltopdf installed)
    report_pdf_path = report_dir / "betting_session_report.pdf"
    try:
        pdfkit.from_file(str(report_html_path), str(report_pdf_path))
    except Exception as e:
        print("PDF generation failed:", e)
        report_pdf_path = None

    # Save JSON and Pickle of the full result object
    result_json_path = report_dir / "session_result.json"
    with open(result_json_path, "w") as f:
        json.dump(report_data, f, default=str)  # convert non-serializable objects to strings

    result_pickle_path = report_dir / "session_result.pkl"
    with open(result_pickle_path, "wb") as f:
        pickle.dump(report_data, f)

    # Display report in notebook
    if display_md:
        try:
            display(Markdown(report_data["report"]))
        except Exception:
            pass

    # Terminal output
    print("\n" + "=" * 80)
    print("BETTING SESSION REPORT:")
    print("=" * 80)
    print(report_data["report"])
    print("=" * 80)
    print(f"Saved to directory: {report_dir.resolve()}\n")

    return {
        "markdown": str(report_md_path),
        "html": str(report_html_path),
        "pdf": str(report_pdf_path) if report_pdf_path else None,
        "json": str(result_json_path),
        "pickle": str(result_pickle_path),
    }



# if __name__ == "__main__":
#     save_and_display_betting_report((
#         generate_betting_session_report_and_save(
#             execution_bankroll=100,
#             avg_game_duration_minutes=200.0,
#             min_break_minutes=60.0,
#             abs_game_limit=None
#             )))



