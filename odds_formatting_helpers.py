#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:46:44 2025

@author: ess
"""

import pandas as pd

def normalize_outcome(bookmaker: str, market_type: str, outcome: str) -> str:
    """
    Normalize an outcome label based on bookmaker and market_type.
    Falls back to the original if no mapping is found.
    """
    
    OUTCOME_MAPPING = {
        "overtime_markets": {
            "total": {
                "option_1": "Over",
                "option_2": "Under"
            },
            "winner": {
                "option_1": "Home",
                "option_2": "Away",
                "option_3": "Draw"
            },
            "spread": {
                "option_1": "Home",
                "option_2": "Away"
            },
        },
        "pinnacle": {
            # Often already clean, but you could map here too if needed
        }}
    
    return (
        OUTCOME_MAPPING
        .get(bookmaker, {})
        .get(market_type, {})
        .get(outcome, outcome)
    )

def normalize_team_outcome(row):
    """
    Converts team name outcomes to 'Home' or 'Away' dynamically.
    If the outcome doesn't match, return it unchanged.
    """
    outcome = row["outcome"]
    if outcome == row.get("home_team"):
        return "Home"
    elif outcome == row.get("away_team"):
        return "Away"
    elif row.get("normalized_outcome")!="":
        return row.get("normalized_outcome")
    return outcome

# Function to add implied probabilities to merged_odds
def add_implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds raw and normalized implied probability columns to the odds DataFrame.
    
    Raw implied = 100 / decimal_odds
    Normalized implied = adjusted so that total per group sums to 100%
    
    Grouping is done by:
    [bookmaker, time, unified_market_type, normalized_line]
    
    Returns:
        DataFrame with 'implied_raw' and 'implied_normalized' columns added.
    """
    df = df.copy()
    
    if "source_id" not in df.columns:
        df["source_id"] = df["match_id"]
    
    # Raw implied probability
    df["implied_raw"] = 100 / df["odds"]
    
    # Fill NaNs in normalized_line for grouping
    df["normalized_line"] = df["line"].fillna(0)

    # Grouped normalization
    group_cols = ["bookmaker", "time", "unified_market_type", "normalized_line", "source_id"]
    
    # Calculate normalized implied probabilities
    df["implied_normalized"] = (
        df.groupby(group_cols)["implied_raw"]
        .transform(lambda x: x / x.sum() * 100)
    )
    
    return df


def evaluate_odds_edges(latest_odds: pd.DataFrame, edge_threshold: float = 1.5) -> pd.DataFrame:
    """
    Evaluates edge metrics between Overtime and Pinnacle for each outcome-line-market combo.

    Args:
        latest_odds (pd.DataFrame): Must contain both overtime_markets and pinnacle data with implied probabilities.
        edge_threshold (float): Minimum edge percentage to consider for flagging.

    Returns:
        pd.DataFrame: Pivoted summary of odds with edge metrics and flags.
    """
    df = latest_odds.copy()

    # Step 1: Separate Overtime and Pinnacle rows
    overtime = df[df["bookmaker"] == "overtime_markets"].copy()
    pinnacle = df[df["bookmaker"] == "pinnacle"].copy()

    # Step 2: Define merge keys and prepare Pinnacle subset
    merge_keys = ["unified_market_type", "normalized_outcome", "normalized_line"]
    pinnacle_subset = pinnacle[merge_keys + ["implied_normalized", "implied_raw"]].rename(
        columns={"implied_normalized": "pinnacle_implied_normalized",
                 "implied_raw": "pinnacle_implied_raw"}
    )

    # Step 3: Merge Pinnacle normalized probabilities into Overtime rows
    overtime = pd.merge(
        overtime,
        pinnacle_subset,
        on=merge_keys,
        how="left"
    )

    # Step 4: Calculate edge metrics
    overtime["edge_vs_pinnacle"] = overtime["pinnacle_implied_normalized"] - overtime["implied_raw"]
    overtime["internal_edge"] = overtime["implied_normalized"] - overtime["implied_raw"]
    overtime["vig_diff"] = overtime["implied_normalized"] - overtime["pinnacle_implied_normalized"]
    overtime["raw_implied_diff"] = overtime["implied_raw"] - overtime["pinnacle_implied_raw"]

    # Step 5: Flag meaningful edges
    overtime["flag_edge_vs_pinnacle"] = overtime["edge_vs_pinnacle"].abs() >= edge_threshold
    overtime["flag_internal_edge"] = overtime["internal_edge"].abs() >= edge_threshold
    overtime["flag_vig_diff"] = overtime["vig_diff"].abs() >= edge_threshold
    overtime["flag_raw_implied_diff"] = overtime["raw_implied_diff"].abs() >= edge_threshold

    # Step 6: Pivot wider to make it easier to scan
    pivot_cols = [
        "unified_market_type", "normalized_outcome", "normalized_line",
        "implied_raw", "implied_normalized", "pinnacle_implied_normalized", "pinnacle_implied_raw",
        "edge_vs_pinnacle", "internal_edge", "vig_diff","raw_implied_diff",
        "flag_edge_vs_pinnacle", "flag_internal_edge", "flag_vig_diff","flag_raw_implied_diff"
    ]

    result = overtime[pivot_cols].sort_values(
        by=["unified_market_type", "normalized_outcome", "normalized_line"]
    )

    return result


def generate_signals(edge_summary: pd.DataFrame, edge_threshold: float = 1.5, vig_tolerance: float = 0.25) -> pd.DataFrame:
    """
    Generates betting signals and scores based on the expected convergence of Overtime odds toward Pinnacle's line.

    Args:
        edge_summary (pd.DataFrame): DataFrame containing edge_vs_pinnacle, vig_diff, etc.
        edge_threshold (float): Minimum edge percentage (Pinnacle - Overtime) to trigger a signal.
        vig_tolerance (float): Max acceptable absolute vig difference between books.

    Returns:
        pd.DataFrame: edge_summary with added signal columns.
    """
    df = edge_summary.copy()

    # Updated scoring components based on convergence logic:
    score_edge_vs_pinnacle = (df["edge_vs_pinnacle"] > edge_threshold).astype(int)  # True price gap
    score_vig_alignment = (df["vig_diff"].abs() < vig_tolerance).astype(int)        # Signal clarity (vig not distorting)
    score_price_advantage = (df["edge_vs_pinnacle"] > 0).astype(int)                # Overtime paying better than fair
    #score_internal_ignore = 0  # We no longer trust overtime's own normalized probs for signal

    # Total score
    df["pinnacle_bet_signal_score"] = score_edge_vs_pinnacle + score_price_advantage + score_vig_alignment

    # Signal fires when at least 2 of 3 criteria are met
    df["pinnacle_bet_signal"] = df["pinnacle_bet_signal_score"] >= 2

    return generate_internal_mispricing_signals(df)

def generate_internal_mispricing_signals(edge_summary: pd.DataFrame, internal_edge_threshold: float = 1.5, max_implied_raw: float = 60.0, max_internal_vig: float = 5.0) -> pd.DataFrame:
    """
    Generates betting signals based only on Overtime's internal odds data.

    Args:
        edge_summary (pd.DataFrame): DataFrame with implied probabilities calculated.
        internal_edge_threshold (float): Minimum internal edge % to consider a value bet.
        max_implied_raw (float): Max implied probability to avoid longshots.
        max_internal_vig (float): Max acceptable overround (implied prob sum - 100).

    Returns:
        pd.DataFrame: DataFrame with added signal score and flag columns.
    """
    df = edge_summary.copy()

    # Ensure internal_edge exists
    if "internal_edge" not in df.columns:
        df["internal_edge"] = df["implied_normalized"] - df["implied_raw"]

    # Compute internal vig per market group
    group_keys = ["unified_market_type", "normalized_line"]
    overrounds = (
        df.groupby(group_keys)["implied_raw"]
        .sum()
        .reset_index(name="overround")
    )
    overrounds["internal_vig"] = overrounds["overround"] - 100

    # Merge internal vig back to the odds
    df = pd.merge(df, overrounds[group_keys + ["internal_vig"]], on=group_keys, how="left")

    # Score components
    score_internal_edge = (df["internal_edge"] > internal_edge_threshold).astype(int)
    score_reasonable_odds = (df["implied_raw"] < max_implied_raw).astype(int)
    score_vig_clarity = (df["internal_vig"] < max_internal_vig).astype(int)

    # Combine into a score
    df["internal_signal_score"] = score_internal_edge + score_reasonable_odds + score_vig_clarity
    df["internal_bet_signal"] = df["internal_signal_score"] >= 2

    return df
