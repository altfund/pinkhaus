#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 20:42:00 2025

@author: ess
"""

#from db_queries import get_upcoming_game
DB_NAME = "sport_odds.db"

from get_oracle_odds import get_upcoming_matched_markets
from odds_formatting_helpers import *

import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import json
import logging


def plot_odds_history(df: pd.DataFrame, y_axis="odds"):
    """
    Plots the odds history for a DataFrame containing merged odds data.

    Args:
        df (pd.DataFrame): DataFrame containing odds history with columns
            ["time", "decimal_odds", "bookmaker", "unified_market_type", "normalized_outcome", "normalized_line"].
    """
    import matplotlib.cm as cm
    import numpy as np

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values("time")
    df["normalized_line"] = df["line"].fillna(0)

    unique_markets = df["unified_market_type"].unique()

    # Fixed line+marker style per bookmaker
    bookmakers = df["bookmaker"].unique()
    marker_styles = ["o", "s", "D", "^", "v", "<", ">"]
    line_styles = ["-", "--", "-.", ":"]
    bookmaker_styles = {
        book: {"marker": marker_styles[i % len(marker_styles)], "linestyle": line_styles[i % len(line_styles)]}
        for i, book in enumerate(bookmakers)
    }

    fig, axes = plt.subplots(nrows=len(unique_markets), ncols=1, figsize=(12, 5 * len(unique_markets)), sharex=True)
    if len(unique_markets) == 1:
        axes = [axes]  # ensure iterable

    # Consistent color for outcome+line
    df["key"] = df.apply(lambda x: f"{x['normalized_outcome']} | {x['normalized_line']}", axis=1)
    unique_keys = sorted(df["key"].unique())
    color_map = dict(zip(unique_keys, cm.get_cmap("tab20")(np.linspace(0, 1, len(unique_keys)))))

    for ax, market in zip(axes, unique_markets):
        sub_df = df[df["unified_market_type"] == market]
        sub_df = sub_df.sort_values(by=["normalized_outcome", "normalized_line", "bookmaker"])

        handles = []
        labels = []

        for (bookmaker, outcome, line), group in sub_df.groupby(["bookmaker", "normalized_outcome", "normalized_line"], sort=False):
            key = f"{outcome} | {line}"
            label = f"{bookmaker} | {key}"
            style = bookmaker_styles.get(bookmaker, {"marker": "o", "linestyle": "-"})

            line_obj, = ax.plot(
                group["time"],
                group[y_axis],
                label=label,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=color_map.get(key, "gray")
            )
            handles.append(line_obj)
            labels.append(label)

        # Sort legend entries by outcome and line
        sorted_legend = sorted(zip(labels, handles), key=lambda x: (x[0].split("|")[1].strip(), float(x[0].split("|")[2].strip())))
        sorted_labels, sorted_handles = zip(*sorted_legend)

        ax.set_title(f"Market: {market}")
        ax.set_ylabel(y_axis)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(sorted_handles, sorted_labels, loc='best', fontsize="small")

    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()

    
def filter_odds_history(
    df: pd.DataFrame,
    filters: dict = None,
    sort_by: str = "time",
    ascending: bool = False
) -> pd.DataFrame:
    """
    Filters and sorts the odds history DataFrame based on flexible parameters.

    Args:
        df (pd.DataFrame): The odds history data.
        filters (dict, optional): Dictionary of column-value pairs to filter on.
                                  Values can be single items or lists/tuples.
                                  e.g., {"bookmaker": ["pinnacle", "overtime_markets"]}
        sort_by (str): Column to sort by (default is "time").
        ascending (bool): Sort order (False = most recent first).

    Returns:
        pd.DataFrame: Filtered and sorted odds history.
    """
    filtered = df.copy()

    # Apply filters
    if filters:
        for column, value in filters.items():
            if column not in filtered.columns:
                print(f"Warning: '{column}' not in DataFrame columns.")
                continue

            if isinstance(value, (list, tuple, set)):
                filtered = filtered[filtered[column].isin(value)]
            else:
                filtered = filtered[filtered[column] == value]

    # Sort the DataFrame
    if sort_by in filtered.columns:
        filtered[sort_by] = pd.to_datetime(filtered[sort_by], errors="coerce")
        filtered = filtered.sort_values(by=sort_by, ascending=ascending)

    return filtered.reset_index(drop=True)




def fetch_odds_history(match_id: int) -> pd.DataFrame:
    """
    Fetches the odds history for both sources tied to a given match ID.

    Args:
        match_id (int): The match ID to fetch odds for.

    Returns:
        pd.DataFrame: Combined DataFrame of odds history with market metadata.
    """
    try:
        with sqlite3.connect(DB_NAME) as conn:
            # Step 1: Lookup source IDs from match
            source_query = """
                SELECT overtime_source_id, odds_api_source_id
                FROM match
                WHERE match_id = ?;
            """
            source_row = pd.read_sql_query(source_query, conn, params=(match_id,))

            if source_row.empty:
                print(f"No match found with ID {match_id}")
                return pd.DataFrame()

            overtime_source_id = source_row.at[0, "overtime_source_id"]
            odds_api_source_id = source_row.at[0, "odds_api_source_id"]

            # Validate IDs
            if pd.isna(overtime_source_id) and pd.isna(odds_api_source_id):
                print(f"Match {match_id} has no valid source IDs.")
                return pd.DataFrame()

            valid_ids = [str(sid) for sid in [overtime_source_id, odds_api_source_id] if pd.notna(sid)]

            # Step 2: Fetch odds and join market metadata
            odds_query = f"""
                SELECT 
                    o.updated_at AS time,
                    o.decimal_odds AS odds,
                    o.outcome,
                    o.source,
                    o.bookmaker,
                    o.market_type,
                    o.line AS line,
                    m.home_team,
                    m.away_team,
                    m.league_name,
                    m.market_type AS match_market_type,
                    m.maturity_date
                FROM odd o
                JOIN market m ON o.source_id = m.source_id
                WHERE o.source_id IN ({','.join(['?'] * len(valid_ids))})
                ORDER BY o.updated_at ASC;
            """
            return pd.read_sql_query(odds_query, conn, params=valid_ids)

    except Exception as e:
        print(f"Error fetching odds history: {e}")
        return pd.DataFrame()



def get_next_match_with_odds(db_name: str, limit: int = 1) -> pd.DataFrame:
    """
    Fetches the next upcoming match from the database that has odds available.

    Args:
        db_name (str): Name of the database file.
        limit (int): Number of rows to fetch.

    Returns:
        pd.DataFrame: DataFrame containing the queried match(es).
    """
    query = f"""
        SELECT mt.match_id, mt.updated_at, m.home_team, m.away_team
        FROM match mt
        JOIN odd o ON o.source_id = mt.match_id
        LEFT JOIN market m ON m.source_id = mt.overtime_source_id OR m.source_id = mt.odds_api_source_id
        GROUP BY mt.match_id
        ORDER BY mt.updated_at ASC
        LIMIT {limit};
    """
    try:
        with sqlite3.connect(db_name) as conn:
            return pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Database query failed: {e}")
        return pd.DataFrame()
    
# Display home win odds for the next maturing market match
def display_next_maturing_home_win_odds(db_name: str, limit: int = 1) -> pd.DataFrame:
    with sqlite3.connect(db_name) as conn:
        query = """
            SELECT m.home_team, m.away_team, m.maturity_date, o.bookmaker, o.markets
            FROM matches AS mt
            JOIN markets AS m ON mt.overtime_game_id = m.game_id OR mt.odds_event_id = m.game_id
            JOIN odds AS o ON mt.match_id = o.match_id
            WHERE m.maturity_date >= ?
            ORDER BY m.maturity_date ASC
            LIMIT 1
        """
        result = pd.read_sql_query(query, conn, params=(pd.Timestamp.now(),))
        if not result.empty:
            row = result.iloc[0]
            odds = json.loads(row['markets'])
            home_win_odds = next((o for o in odds if o['key'] == 'home'), None)
            logging.info(f"Next Match: {row['home_team']} vs {row['away_team']} (Maturity: {row['maturity_date']})")
            logging.info(f"Bookmaker: {row['bookmaker']}, Home Win Odds: {home_win_odds}")
        else:
            logging.info("No upcoming matches with odds available.")




def diagnose_alignment_issues(merged_odds: pd.DataFrame) -> dict:
    """
    Runs a series of diagnostics on the merged_odds DataFrame to help explain misalignments
    between Overtime and Pinnacle's implied probabilities.

    Returns a dictionary of diagnostic outputs for further review.
    """
    diagnostics = {}

    # 1. Check for null unified_market_type
    diagnostics["null_unified_market_type"] = merged_odds[merged_odds["unified_market_type"].isnull()]

    # 2. Unique lines and data types
    diagnostics["line_type_check"] = merged_odds["normalized_line"].apply(type).value_counts()

    # 3. Unique values for normalized_outcome per bookmaker
    diagnostics["outcome_by_bookmaker"] = (
        merged_odds.groupby("bookmaker")["normalized_outcome"].value_counts().unstack(fill_value=0)
    )

    # 4. Line mismatches by market + outcome
    diagnostics["group_counts"] = (
        merged_odds.groupby(["unified_market_type", "normalized_outcome", "normalized_line"])["bookmaker"]
        .nunique()
        .reset_index(name="bookmaker_count")
        .sort_values("bookmaker_count", ascending=False)
    )

    # 5. Compare line values by bookmaker and market
    diagnostics["lines_by_bookmaker"] = (
        merged_odds.groupby(["unified_market_type", "bookmaker"])["normalized_line"]
        .value_counts()
        .unstack(fill_value=0)
    )

    # 6. Outcome names that mismatch across bookmakers
    outcome_overlap = (
        merged_odds.groupby(["unified_market_type", "normalized_line", "normalized_outcome"])["bookmaker"]
        .nunique()
        .reset_index()
    )
    diagnostics["mismatched_outcomes"] = outcome_overlap[outcome_overlap["bookmaker"] < 2]

    return diagnostics

def check_for_arbitrage_opportunities(limit: int = 5):
    """
    Entry point for querying multiple upcoming matches and displaying arbitrage opportunities.
    """
    upcoming_matches = get_upcoming_matched_markets(limit=limit)

    if upcoming_matches.empty:
        print("No upcoming matches found.")
        return

    print("Upcoming matches:")
    print(upcoming_matches)

    match_ids = upcoming_matches["match_id"].tolist()

    # Fetch and combine odds data for all matches
    all_odds = pd.concat([
        fetch_odds_history(mid).assign(match_id=mid) for mid in match_ids
    ], ignore_index=True)


    # Filter for relevant bookmakers
    filtered_odds = filter_odds_history(
        all_odds,
        filters={"bookmaker": ["overtime_markets", "pinnacle"]},
        sort_by="time",
        ascending=False
    )

    # Map market_type to unified type
    market_type_map = pd.DataFrame([
        {"bookmaker": "overtime_markets", "market_type": "winner", "unified_market_type": "h2h"},
        {"bookmaker": "pinnacle", "market_type": "h2h", "unified_market_type": "h2h"},
        {"bookmaker": "overtime_markets", "market_type": "spread", "unified_market_type": "spread"},
        {"bookmaker": "pinnacle", "market_type": "spreads", "unified_market_type": "spread"},
        {"bookmaker": "overtime_markets", "market_type": "total", "unified_market_type": "total"},
        {"bookmaker": "pinnacle", "market_type": "totals", "unified_market_type": "total"},
    ])

    # Merge unified market_type
    merged_odds = pd.merge(
        filtered_odds,
        market_type_map,
        how="inner",
        on=["bookmaker", "market_type"]
    )

    # Normalize outcomes
    merged_odds["normalized_outcome"] = merged_odds.apply(
        lambda row: normalize_outcome(row["bookmaker"], row["market_type"], row["outcome"]),
        axis=1
    )
    merged_odds["normalized_outcome"] = merged_odds.apply(normalize_team_outcome, axis=1)
    merged_odds["normalized_line"] = pd.to_numeric(merged_odds["line"], errors="coerce").fillna(0.0)

    # Calculate implied probabilities
    merged_odds = add_implied_probabilities(merged_odds)
    
    # Run diagnostics on the sample
    diagnostics = diagnose_alignment_issues(merged_odds)

    latest_odds = merged_odds.drop_duplicates(
        subset=["match_id", "bookmaker", "unified_market_type", "normalized_outcome", "normalized_line"]
    )

    # Evaluate edge and generate signals
    edge_summary = evaluate_odds_edges(latest_odds, edge_threshold=0.01)
    signal_summary = generate_signals(edge_summary)

    # Keep only outcomes with multiple bookmakers
    group_counts = (
        merged_odds
        .groupby(["match_id", "unified_market_type", "normalized_outcome", "normalized_line"])["bookmaker"]
        .nunique()
        .reset_index(name="bookmaker_count")
    )
    valid_groups = group_counts[group_counts["bookmaker_count"] >= 2]

    arb_odds = pd.merge(
        merged_odds,
        valid_groups[["match_id", "unified_market_type", "normalized_outcome", "normalized_line"]],
        on=["match_id", "unified_market_type", "normalized_outcome", "normalized_line"],
        how="inner"
    )

    # Get most recent odds per bookmaker-market-outcome-line combo
    arb_odds["time"] = pd.to_datetime(arb_odds["time"], errors="coerce")
    arb_odds = arb_odds.sort_values("time", ascending=False)

    market_type_summary = (
        merged_odds[["bookmaker", "market_type"]]
        .drop_duplicates()
        .pivot_table(
            index="market_type",
            columns="bookmaker",
            aggfunc=lambda x: True,
            fill_value=False
        )
    )

    if not arb_odds.empty:
        plot_odds_history(arb_odds, y_axis="odds") #implied_normalized _raw #odds
    else:
        print(f"No odds data found")




if __name__ == "__main__":
    check_for_arbitrage_opportunities()