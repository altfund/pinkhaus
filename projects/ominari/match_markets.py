#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 21:43:19 2025

@author: ess
"""

import logging
import sqlite3
import pandas as pd

from free_data_pull import upsert_table

DB_NAME = "sport_odds.db"


# Function to match markets exactly and upsert results
def match_markets():
    logging.info("Matching markets from Overtime and Odds API...")
    with sqlite3.connect(DB_NAME) as conn:
        query = """
            SELECT o1.source_id AS overtime_source_id, o2.source_id AS odds_api_source_id--,
                   --o1.home_team, o1.away_team, o1.maturity_date, o2.maturity_date AS odds_maturity_date
            FROM market o1
            JOIN market o2
            ON o1.home_team = o2.home_team AND o1.away_team = o2.away_team
            AND ABS(strftime('%s', o1.maturity_date) - strftime('%s', o2.maturity_date)) = 0 --3600
            WHERE o1.source = 'overtime_markets' AND o2.source = 'odds_api'
        """
        matches = pd.read_sql_query(query, conn)

        if matches.empty:
            logging.warning("No exact matches found between Overtime and Odds API.")
            return

        matches["confidence"] = 1.00
        matches["updated_at"] = pd.Timestamp.now().isoformat()
        upsert_table(
            DB_NAME,
            matches,
            "match",
            key_columns=["overtime_source_id", "odds_api_source_id"],
        )
        logging.info(f"Upserted {len(matches)} exact matches into match table.")


# Main function
if __name__ == "__main__":
    # Match markets and store results in match table
    match_markets()
    logging.info(
        "Exact markets matched and upserted in match table with 100% confidence."
    )
