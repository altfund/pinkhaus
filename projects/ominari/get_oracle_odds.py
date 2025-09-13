#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 22:48:33 2025

@author: ess
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import logging
import requests

from free_data_pull import insert_odds

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports"
ODDS_API_KEY = "2232ba79d0318e421f1b86ac63a9c92a"
DB_NAME = "sport_odds.db"


# Fetch odds data from Odds API using the /odds endpoint
def fetch_odds_api_odds(
    odds_api_sport="upcoming",
    regions=["eu"],
    market_types=["h2h", "spreads", "totals"],
    api_key=ODDS_API_KEY,
):
    regions_str = ",".join(regions)
    markets = ",".join(market_types)

    try:
        url = f"{ODDS_API_URL}/{odds_api_sport}/odds/?apiKey={ODDS_API_KEY}&regions={regions_str}&markets={markets}"
        logging.info(
            f"Fetching odds using the Odds API /odds endpoint for sport={odds_api_sport}..."
        )
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 422:
            logging.warning(
                f"Skipping unsupported sport_key: {odds_api_sport}. Full URL: {url}"
            )
        else:
            logging.error(
                f"HTTP Error fetching events from Odds API: {http_err}. Full URL: {url}"
            )
        return None
    except requests.exceptions.RequestException as e:
        logging.error(
            f"Error fetching events from Odds API /events endpoint: {e}. Full URL: {url}"
        )
        return None


# Function to extract odds data from the Odds API response
def extract_odds_api_odds(odds_api_json):
    results = []

    for event in odds_api_json:
        event_id = event.get("id")

        for bookmaker in event.get("bookmakers", []):
            bookmaker_key = bookmaker.get("key")
            # last_update = bookmaker.get("last_update")

            for market in bookmaker.get("markets", []):
                market_type = market.get("key")

                for outcome in market.get("outcomes", []):
                    results.append(
                        {
                            "source_id": event_id,
                            "source": "odds_api",
                            "bookmaker": bookmaker_key,
                            "market_type": market_type,
                            "outcome": outcome.get("name"),
                            "line": outcome.get("point"),
                            "decimal_odds": outcome.get("price"),
                        }
                    )

    return pd.DataFrame(results)


# Save odds to the database
def save_odds_to_db(match_id, odds_data, source):
    if odds_data is None:
        logging.warning(f"No odds data provided for match_id {match_id}. Skipping.")
        return
    with sqlite3.connect(DB_NAME) as conn:
        odds_records = []
        for event in odds_data:
            bookmakers = event.get("bookmakers", [])
            for bookmaker in bookmakers:
                odds_records.append(
                    (
                        match_id,
                        source,
                        bookmaker.get("title"),
                        json.dumps(bookmaker.get("market")),
                        pd.Timestamp.now().isoformat(),
                    )
                )

        conn.executemany(
            """
            INSERT INTO odds (match_id, source, bookmaker, market, retrieved_at)
            VALUES (?, ?, ?, ?, ?)
        """,
            odds_records,
        )


DB_NAME = "sport_odds.db"


# Function to get upcoming matched markets
def get_upcoming_matched_markets(limit=5):
    """Fetch the next upcoming matched markets from the database and structure them for comparison across sources."""
    query = """
        SELECT mt.match_id, mt.confidence, m.maturity_date AS overtime_maturity_date, mo.maturity_date AS odds_maturity_date,
               m.home_team AS overtime_home, mo.home_team AS odds_home, 
               m.away_team AS overtime_away, mo.away_team AS odds_away,
               m.sport AS overtime_sport, mo.sport AS odds_sport, 
               m.league_name AS overtime_league, mo.league_name AS odds_league, 
               m.market_type AS overtime_market_type, mo.market_type AS odds_market_type,
               mt.overtime_source_id AS overtime_source_id, mt.odds_api_source_id AS odds_api_source_id
        FROM match mt
        JOIN market m ON mt.overtime_source_id = m.source_id
        JOIN market mo ON mt.odds_api_source_id = mo.source_id
        WHERE 
            m.maturity_date >= datetime('now', 'utc') AND
            overtime_sport = "Soccer"
        ORDER BY m.maturity_date ASC
    """
    if limit:
        query += " LIMIT ?;"
        params = (limit,)
    else:
        query += ";"
        params = ()

    with sqlite3.connect(DB_NAME) as conn:
        return pd.read_sql_query(query, conn, params=params)


# Main function
if __name__ == "__main__":
    # Fetch the data
    upcoming_matches = get_upcoming_matched_markets(limit=None)

    # Display the results in a structured format
    if not upcoming_matches.empty:
        # grouped_matches = upcoming_matches.groupby("match_id").first().reset_index()
        print(upcoming_matches)  # grouped_matches)

        sports_in_focus = upcoming_matches["odds_sport"].dropna().unique()[0]
        sports_in_focus = np.atleast_1d(sports_in_focus)

        for odds_api_sport in sports_in_focus:
            odds_data_json = fetch_odds_api_odds(odds_api_sport=odds_api_sport)
            if odds_data_json:
                odds_data_df = extract_odds_api_odds(odds_data_json)
                logging.info(
                    f"Fetched {len(odds_data_df)} odds records for sport {odds_api_sport}."
                )
                if not odds_data_df.empty:
                    insert_odds(DB_NAME, odds_data_df)
                    logging.info(
                        f"Added {len(odds_data_df)} odds records for sport {odds_api_sport}."
                    )

    else:
        print("No upcoming matched markets found.")
