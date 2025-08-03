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
from typing import Optional, Dict

from sqlalchemy import create_engine
from sqlalchemy.orm  import sessionmaker
from models          import Base, BettingSession, Bet

engine = create_engine(f"sqlite:///{DB_NAME}")

Session = sessionmaker(bind=engine)


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
        import grpc
        from external_pb2 import SignalBatchRequest
        from external_pb2_grpc import SignalServiceStub

        # 1) Always log entry & DataFrame size
        print(f"[CLIENT] ExternalGrpcSignal.get_probs: {len(df)} rows", flush=True)

        # 2) If empty, bail early (but log it)
        if df.empty:
            print("[CLIENT]  → empty df → returning empty Series", flush=True)
            return pd.Series([], index=df.index)

        # 3) Build & log the batch request
        req = SignalBatchRequest()
        for idx, row in df.iterrows():
            r = req.requests.add()
            r.source_id          = row["source_id"]
            r.normalized_outcome = row["normalized_outcome"]
            r.as_of_time         = row["time"].isoformat()
        print(f"[CLIENT]  → sending {len(req.requests)} RPC requests", flush=True)

        # 4) Actually call the RPC, but don’t hide exceptions
        try:
            t0 = time.time()
            resp = self.stub.GetProbabilities(req, timeout=self.timeout)
            took = time.time() - t0
            print(f"[CLIENT]  → RPC returned in {took:.3f}s, {len(resp.probabilities)} probs", flush=True)
            probs = list(resp.probabilities)
        except grpc.RpcError as e:
            # log the error before fallback
            print(f"[CLIENT]  ! RPC error: {e.code()} {e.details()}", flush=True)
            probs = [0.0] * len(df)

        # 5) Return and log
        series = pd.Series(probs, index=df.index)
        print(f"[CLIENT]  → returning series head:\n{series.head()}", flush=True)
        return series

def save_bets_to_db(
    bets_df,
    as_of_datetime,
    session_type: str,
    strategy_name: str,
    # ← new parameters ↓
    kelly_bankroll: float,
    execution_bankroll: float,
    kelly_fraction: float,
    cap_per_game: float,
    cap_per_bet: float,
    cap_per_game_market: float,
    min_bet_abs: float,
    min_bet_pct: float,
    abs_game_limit: Optional[int],
    min_break_minutes: float,
    avg_game_duration_minutes: float,
    signal_weights: Dict[str, float],
    num_open_markets: int,
    num_games: int,
    normalized_return: float, 
    expected_return: float,
    total_original_stake: float,
    total_trimmed_stake: float,
    total_fee_amount: float,
    db_path: str = DB_NAME
):
    session = Session()

    bs = BettingSession(
        as_of                     = as_of_datetime,
        session_type              = session_type,
        strategy_name             = strategy_name,
        kelly_bankroll            = kelly_bankroll,
        execution_bankroll        = execution_bankroll,
        kelly_fraction            = kelly_fraction,
        cap_per_game              = cap_per_game,
        cap_per_bet               = cap_per_bet,
        cap_per_game_market       = cap_per_game_market,
        min_bet_abs               = min_bet_abs,
        min_bet_pct               = min_bet_pct,
        abs_game_limit            = abs_game_limit,
        min_break_minutes         = min_break_minutes,
        avg_game_duration_minutes = avg_game_duration_minutes,
        signal_weights            = json.dumps(signal_weights),
        num_open_markets          = num_open_markets,
        num_games                 = num_games,
        num_bets_recommended      = len(bets_df),
        expected_return           = expected_return,
        normalized_return         = (expected_return / total_trimmed_stake)
                                        if total_trimmed_stake else None,
        total_original_stake      = total_original_stake,
        total_trimmed_stake       = total_trimmed_stake,
        total_fee_amount          = total_fee_amount,
    )
    session.add(bs)
    session.commit()  # populate bs.id

    for _, r in bets_df.iterrows():
        bet = Bet(
            session_id          = bs.id,
            source_id           = r["source_id"],
            unified_market_type = r["unified_market_type"],
            normalized_outcome  = r["normalized_outcome"],
            normalized_line     = float(r["normalized_line"]),
            bet_name            = r.get("bet_name"),
            probability         = float(r.get("probability", 0.0)),
            odds                = float(r.get("odds", 0.0)),
            stake               = float(r["stake"]),
            execution_stake     = float(r.get("execution_stake", 0.0)),
            fee_amount          = float(r.get("fee_amount", 0.0)),
            fee_pct             = float(r.get("fee_pct", 0.0))
        )
        session.merge(bet)

    session.commit()
    session_id = bs.id
    session.close()
    return session_id



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


import pandas as pd
from typing import Optional

def get_upcoming_overtime_markets_with_signals(
        limit: int = 10,
        as_of: Optional[datetime] = None,
        min_break_minutes: int = 60,
        avg_game_duration_minutes: int = 180,
        window_end: Optional[datetime] = None
    ) -> pd.DataFrame:
    """
    Fetches Overtime markets as of `as_of`, spanning from the next game
    start through `window_end` (or +avg_game_duration if no window_end),
    with implied probabilities & signals.
    """
    t0 = time.time()
    # --- 1) normalize as_of ---
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    elif as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    print(f"[TIMING] as_of setup: {time.time()-t0:.3f}s")

    # --- 2) find the very next maturity_date after as_of ---
    t1 = time.time()
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT MIN(maturity_date) AS next_date
              FROM market
             WHERE source        = 'overtime_markets'
               AND sport         = 'Soccer'
               AND market_type   = 'winner'
               AND maturity_date >= :as_of
            """,
            {"as_of": as_of.isoformat()}
        ).fetchone()
    next_date = row["next_date"]
    print(f"[TIMING] found next_date ({next_date}): {time.time()-t1:.3f}s")
    if next_date is None:
        print("[DEBUG] no upcoming games → empty")
        return pd.DataFrame([])

    # convert that to a UTC-aware datetime
    # convert that to a UTC-aware datetime
    chunk_start = pd.to_datetime(next_date, utc=True)

    # --- 3) decide real window_end ---
    if window_end is None:
        chunk_end = chunk_start + timedelta(minutes=avg_game_duration_minutes)
    else:
        chunk_end = pd.to_datetime(window_end, utc=True)

    # for SQL we need naive UTC datetimes
    ws_naive = chunk_start.astimezone(timezone.utc).replace(tzinfo=None)
    we_naive = chunk_end   .astimezone(timezone.utc).replace(tzinfo=None)
    print(f"[DEBUG] pulling markets {chunk_start}→{chunk_end}")

    # --- 4) pull _all_ relevant markets in that window via SQL ---
    t2 = time.time()
    query = f"""
    WITH relevant_markets AS (
      SELECT source_id
        FROM market
       WHERE source        = 'overtime_markets'
         AND sport         = 'Soccer'
         AND market_type   = 'winner'
         AND maturity_date BETWEEN :ws AND :we
    ),
    market_first_seen AS (
      SELECT o.source_id,
             MIN(o.updated_at) AS first_seen
        FROM odd o
        JOIN relevant_markets rm USING (source_id)
       WHERE o.bookmaker   = 'overtime_markets'
         AND o.market_type = 'winner'
       GROUP BY o.source_id
    )
    SELECT o.updated_at   AS time,
           o.line, o.decimal_odds AS odds, o.outcome,
           o.source, o.bookmaker, o.market_type, o.source_id,
           m.maturity_date, m.home_team, m.away_team, m.league_name, m.sport
      FROM odd o
      JOIN market m               ON o.source_id = m.source_id
      JOIN relevant_markets rm    ON o.source_id = rm.source_id
      JOIN market_first_seen fs   ON fs.source_id = o.source_id
     WHERE o.bookmaker     = 'overtime_markets'
       AND m.sport         = 'Soccer'
       AND o.market_type   = 'winner'
       AND o.updated_at   <= :as_of
       AND fs.first_seen  <= :as_of
     ORDER BY o.updated_at DESC
     {f"LIMIT {limit}" if limit else ""}
    """
    params = {
        "as_of": as_of.isoformat(),
        "ws":    ws_naive.isoformat(),
        "we":    we_naive.isoformat(),
    }
    df = pd.read_sql_query(
        query,
        sqlite3.connect(DB_NAME),
        params=params,
        parse_dates=["time","maturity_date"]
    )
    print(f"[TIMING] SQL fetch ({len(df)} rows): {time.time()-t2:.3f}s")

    # --- 5) normalize & unify as before ---
    t3 = time.time()
    # first pull back the raw mapping
    df["normalized_outcome"] = df.apply(
        lambda r: normalize_outcome(r.bookmaker, r.market_type, r.outcome),
        axis=1
    )
    
    # then re‐normalize to Home/Away based on team names
    df["normalized_outcome"] = df.apply(normalize_team_outcome, axis=1)

    df["normalized_line"] = pd.to_numeric(df["line"], errors="coerce").fillna(0.0)
    print(f"[TIMING] normalize: {time.time()-t3:.3f}s")

    t4 = time.time()
    mt_map = pd.DataFrame([
        {"bookmaker":"overtime_markets","market_type":"winner","unified_market_type":"h2h"},
        {"bookmaker":"overtime_markets","market_type":"spread","unified_market_type":"spread"},
        {"bookmaker":"overtime_markets","market_type":"total","unified_market_type":"total"},
    ])
    df = df.merge(mt_map, on=["bookmaker","market_type"], how="inner")
    print(f"[TIMING] merge market_type: {time.time()-t4:.3f}s")

    # --- 6) final maturity_date filter (now honors chunk_end) ---
    t5 = time.time()
    df["time"]          = pd.to_datetime(df["time"],          utc=True)
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], utc=True)
    mask = (
        (df["maturity_date"] >= chunk_start) &
        (df["maturity_date"] <= chunk_end)
    )
    df = df.loc[mask]
    print(f"[DEBUG] window {chunk_start}→{chunk_end}, rows: {mask.sum()}")
    print(f"[TIMING] filter window: {time.time()-t5:.3f}s")

    # --- 7) fees, implied, dedupe, signals as before ---
    t6 = time.time()
    df = apply_overtime_fees(df)
    df = add_implied_probabilities(df)
    print(f"[TIMING] fees+implied: {time.time()-t6:.3f}s")

    t7 = time.time()
    df = (
        df.sort_values("time", ascending=False)
          .drop_duplicates(
             subset=["unified_market_type","normalized_outcome","normalized_line","source_id"]
          )
    )
    print(f"[TIMING] dedupe latest: {time.time()-t7:.3f}s")

    t8 = time.time()
    result = filter_valid_internal_signals(generate_internal_mispricing_signals(df))
    print(f"[TIMING] internal signals: {time.time()-t8:.3f}s")

    print(f"[TIMING] TOTAL get_upcoming...: {time.time()-t0:.3f}s")
    return result.head(limit) if limit else result


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


def prepare_kelly_input(open_markets: pd.DataFrame,
                        signal_providers: None,
                        signal_weights: None
                        ) -> pd.DataFrame:
    """
    Converts filtered open_markets to the format required for calculate_kelly_stakes_with_exclusivity.
    """
    df = open_markets.copy()
    
    print(f"[DEBUG] prepare_kelly_input called with {len(open_markets)} rows; "
          f"providers={[p.name for p in signal_providers]}, weights={signal_weights}")

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
            **{f"contrib_{p.name}": pd.Series(dtype=float) for p in signal_providers}
        )
    
    # now it's guaranteed non-empty
    
    df = aggregate_signals(df, signal_providers, signal_weights)
    
    df["odds"] = df["adjusted_odds"]

    return df[["source_id", "market_name", "bet_name", "league_name", "unified_market_type", 
               "normalized_outcome", "normalized_line", "odds", "probability", "bookmaker",
               *[f"contrib_{p.name}" for p in signal_providers]
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


from datetime import timedelta, datetime, timezone
import pandas as pd

def find_upcoming_game_breaks(
    matches_df: pd.DataFrame,
    min_break_minutes: float = 60,
    avg_game_duration_minutes: float = 180,
    now: datetime = None
) -> pd.DataFrame:
    if now is None:
        now = datetime.now(timezone.utc)

    df = matches_df.copy()
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], utc=True)
    df = df[df["maturity_date"] > now].sort_values("maturity_date").reset_index(drop=True)

    if len(df) < 2:
        return pd.DataFrame([], columns=[
            "break_start", "break_end", "duration_minutes", "games_before", "games_after"
        ])

    df["next_game"] = df["maturity_date"].shift(-1)
    # raw gap in minutes
    df["raw_gap"] = (df["next_game"] - df["maturity_date"]).dt.total_seconds() / 60
    # effective idle time after you allow the first game to finish
    df["effective_gap"] = df["raw_gap"] - avg_game_duration_minutes

    # only keep those breaks where effective idle ≥ threshold
    breaks = df[df["effective_gap"] >= min_break_minutes]

    rows = []
    for idx, row in breaks.iterrows():
        start = row["maturity_date"] + timedelta(minutes=avg_game_duration_minutes)
        end   = row["next_game"]
        rows.append({
            "break_start":       start,
            "break_end":         end,
            "duration_minutes":  (end - start).total_seconds() / 60,
            "games_before":      idx + 1,
            "games_after":       len(df) - idx - 1,
        })

    return pd.DataFrame(rows)


def extract_active_game_periods_from_breaks(
    match_df: pd.DataFrame,
    break_df: pd.DataFrame,
    avg_game_duration_minutes: float = 120,
) -> pd.DataFrame:
    """
    Carve out contiguous periods of play given the true break intervals.
    """
    # --- new guard ---
    if match_df.empty:
        # No games at all → no chunks
        return pd.DataFrame(columns=[
            "chunk_start", "chunk_end", "duration_minutes", "num_games"
        ])

    df = match_df.sort_values("maturity_date").reset_index(drop=True)
    # always compute an explicit end_time for chunk-termination
    df["end_time"] = df["maturity_date"] + timedelta(minutes=avg_game_duration_minutes)

    # if no breaks, single chunk from first start to last end
    if break_df.empty:
        return pd.DataFrame([{
            "chunk_start":      df["maturity_date"].iloc[0],
            "chunk_end":        df["end_time"].iloc[-1],
            "duration_minutes": (df["end_time"].iloc[-1] - df["maturity_date"].iloc[0]).total_seconds() / 60,
            "num_games":        len(df)
        }])

    # build lists of (start, end) for each chunk
    chunk_starts = [df["maturity_date"].iloc[0]] + list(break_df["break_end"])
    chunk_ends   = list(break_df["break_start"])  + [df["end_time"].iloc[-1]]

    chunks = []
    for start, end in zip(chunk_starts, chunk_ends):
        sub = df[(df["maturity_date"] >= start) & (df["end_time"] < end)]
        if not sub.empty:
            chunks.append({
                "chunk_start":      start,
                "chunk_end":        end,
                "duration_minutes": (end - start).total_seconds() / 60,
                "num_games":        len(sub)
            })

    return pd.DataFrame(chunks)



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


SIGNAL_PROVIDERS = [
    ImpliedRawSignal()#,
    #ExternalGrpcSignal(_external_stub),
]

SIGNAL_WEIGHTS = {
    "implied_raw": 1.0#,    # base implied probability
    #"external":    1.0,    # weight for external model
}

# assume all your existing imports: get_upcoming_overtime_markets_with_signals,
# summarize_match_schedule_from_open_markets, find_upcoming_game_breaks,
# extract_active_game_periods_from_breaks, prepare_kelly_input, calculate_kelly_stakes_with_exclusivity,
# build_structural_correlation_matrix, calculate_expected_kelly_return, trim_kelly_results,
# enrich_trimmed_results_with_fee_info, save_and_display_betting_report, save_bets_to_db

def generate_betting_session_report_and_save(
    kelly_bankroll: float = 1.0,
    execution_bankroll: float = 15.0,
    kelly_fraction: float = 0.25,
    cap_per_game: float = 0.25,
    cap_per_bet: float = 0.25,
    cap_per_game_market: float = 0.10,
    min_bet_abs: float = 0.5,
    min_bet_pct: float = 0.0025,
    abs_game_limit: Optional[int] = None,
    as_of: Optional[datetime] = None,
    min_break_minutes: float = 360.0,
    avg_game_duration_minutes: float = 180.0,
    base_dir: str = "betting_reports",
    display_md: bool = True,
    save_as_latest: bool = True,
    signal_providers=None,
    signal_weights=None,
    mode: Optional[str] = None,
    strat: Optional[str] = None,
    window_end: Optional[datetime] = None
) -> dict:
    # --- 1) normalize as_of ---
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    elif as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    print(f"evaluating open markets as of {as_of.isoformat()}")

    # --- 2) determine session window [start, end] ---
    start = as_of
    default_end = as_of + timedelta(minutes=avg_game_duration_minutes)
    # allow explicit override
    if window_end is not None:
        end = window_end
    else:
        end = default_end

    # --- 3) load full as_of snapshot and filter by maturity_date ---
    open_markets = get_upcoming_overtime_markets_with_signals(
        limit=abs_game_limit,
        as_of=as_of,
        window_end=end,  
        min_break_minutes=min_break_minutes,
        avg_game_duration_minutes=avg_game_duration_minutes
    )
    print(f"[DEBUG] open_markets → {len(open_markets)} rows")
    
    if open_markets.empty:
        print("[DEBUG] no upcoming games → empty")
        return {"report": "", "filename": None}

    open_markets["maturity_date"] = pd.to_datetime(
        open_markets["maturity_date"], utc=True
    )
    
    # --- 5) prepare for Kelly ---
    match_df    = summarize_match_schedule_from_open_markets(open_markets)
    breaks_df   = find_upcoming_game_breaks(match_df, min_break_minutes, avg_game_duration_minutes, now=as_of)
    game_times  = extract_active_game_periods_from_breaks(match_df, breaks_df)
    
    # after game_times_df = extract_active_game_periods_from_breaks(...)
    if not game_times.empty:
        first_chunk  = game_times.iloc[0]
        if len(game_times) > 1:
            # session ends exactly when the next chunk starts
            end = game_times.iloc[1]["chunk_start"]
        else:
            end = first_chunk["chunk_start"] + timedelta(minutes=avg_game_duration_minutes)


    markets_df = open_markets[
        (open_markets["maturity_date"] >= start) &
        (open_markets["maturity_date"] < end)
    ]
    print(f"[DEBUG] filtered markets → {len(markets_df)} rows")
    
    
    # --- 4) early exit if nothing to bet on ---
    if markets_df.empty:
        print(f"No markets maturing before {end.isoformat()} — skipping session.")
        return {"report": "", "filename": None}
    

    
    # --- ensure we have a market_name column ---
    if "market_name" not in markets_df.columns:
        markets_df = markets_df.copy()
        markets_df["market_name"] = (
            markets_df["home_team"].astype(str)
            + "_vs_"
            + markets_df["away_team"].astype(str)
        )

    kelly_ready = prepare_kelly_input(markets_df, signal_providers, signal_weights)
    kelly_raw   = calculate_kelly_stakes_with_exclusivity(
        kelly_ready,
        bankroll=kelly_bankroll,
        correlation_matrix=build_structural_correlation_matrix(kelly_ready),
        risk_adjusted=True,
    )
    trimmed     = trim_kelly_results(
        kelly_raw,
        kelly_fraction=kelly_fraction,
        bankroll=execution_bankroll,
        cap_per_game=cap_per_game,
        cap_per_bet=cap_per_bet,
        cap_per_game_market=cap_per_game_market,
        min_bet_abs=min_bet_abs,
        min_bet_pct=min_bet_pct,
    )
    trimmed     = enrich_trimmed_results_with_fee_info(trimmed)

    # --- 6) compute stats & build report ---
    total_staked            = trimmed["stake"].sum()
    exp_ret                 = calculate_expected_kelly_return(trimmed, None)
    exp_mult                = np.exp(exp_ret)
    vol                     = np.std(np.log1p(trimmed["stake_fraction"] * (trimmed["odds"] - 1)))
    sharpe                  = exp_ret / vol if vol else 0

    active_games = sorted(markets_df["market_name"].unique())
    bets_to_make = trimmed[trimmed["stake"] > 0]

    report_md = f"""
BETTING SESSION REPORT
=======================

Session Window: {start} → {end}  (duration: {end-start})

Parameters:
  • Kelly bankroll: {kelly_bankroll}
  • Execution bankroll: {execution_bankroll}
  • Kelly fraction: {kelly_fraction}
  • Caps: game={cap_per_game}, bet={cap_per_bet}, game‐market={cap_per_game_market}
  • Min bets: {min_bet_abs} abs, {min_bet_pct:.4f} pct

Games Covered: {', '.join(active_games)}

Expected Return (log): {exp_ret:.4f}
Expected Multiplier: {exp_mult:.3f}x
Volatility: {vol:.4f}
Sharpe: {sharpe:.4f}
Total Stake: {total_staked:.2f}

Bets to Place:
{bets_to_make.to_string(index=False)}
"""

    # display and/or save report
    if mode != "backtest":
        save_and_display_betting_report(
            {"report": report_md, "start": start, "end": end}, 
            base_dir=base_dir, display_md=display_md
        )

    # --- 7) persist to DB ---
    stats = dict(
        num_open_markets      = len(open_markets),
        #num_bettable_markets  = len(markets_df),
        total_trimmed_stake   = float(total_staked),
        total_fee_amount      = float(bets_to_make["fee_amount"].sum()),
        expected_return   = float(exp_ret),
        normalized_return     = float(exp_ret / total_staked if total_staked else 0),
        total_original_stake  = float(trimmed["original_stake"].sum())
    )

    session_id = save_bets_to_db(
        bets_df                    = bets_to_make,
        as_of_datetime             = as_of,
        session_type               = mode or "simulation",
        strategy_name              = strat,

        # ← your strategy params
        kelly_bankroll             = kelly_bankroll,
        execution_bankroll         = execution_bankroll,
        kelly_fraction             = kelly_fraction,
        cap_per_game               = cap_per_game,
        cap_per_bet                = cap_per_bet,
        cap_per_game_market        = cap_per_game_market,
        min_bet_abs                = min_bet_abs,
        min_bet_pct                = min_bet_pct,
        abs_game_limit             = abs_game_limit,
        min_break_minutes          = min_break_minutes,
        avg_game_duration_minutes  = avg_game_duration_minutes,
        signal_weights             = signal_weights,

        # ← session‐level stats
        num_open_markets           = len(open_markets),
        num_games                  = match_df.shape[0],
        normalized_return          = stats["normalized_return"],
        expected_return            = stats["expected_return"],
        total_original_stake       = stats["total_original_stake"],
        total_trimmed_stake        = stats["total_trimmed_stake"],
        total_fee_amount           = stats["total_fee_amount"],

        # (optional override of default DB path)
        db_path                    = DB_NAME,
    )
    print(f"Saved BettingSession id={session_id}")

    # --- 8) optionally record latest ---
    if save_as_latest and mode != "backtest":
        p = Path(base_dir) / "latest_session.pkl"
        with open(p, "wb") as f:
            pickle.dump({"report": report_md, **stats}, f)

    return {
        "report": report_md,
        "kelly_input": kelly_ready,
        "kelly_raw": kelly_raw,
        "trimmed": trimmed,
        "session_id": session_id,
    }


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
    timestamp = report_data["as_of"].strftime("%Y-%m-%d_%H-%M-%S")
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


if __name__ == "__main__":
    save_and_display_betting_report((
        generate_betting_session_report_and_save(
            execution_bankroll=100,
            avg_game_duration_minutes=200.0,
            min_break_minutes=60.0,
            abs_game_limit=None,
            signal_providers=SIGNAL_PROVIDERS,
            signal_weights=SIGNAL_WEIGHTS
            )))



