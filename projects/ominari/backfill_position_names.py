#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 02:44:54 2025

@author: ess
"""

# backfill_position_names.py
import sqlite3
import json
import pandas as pd
from free_data_pull import get_all_overtime_markets, get_overtime_markets_markets

DB = "sport_odds.db"

def backfill_position_names():
    # 1) Fetch and build a DataFrame of every market
    raw = get_all_overtime_markets()
    full_md  = get_overtime_markets_markets(raw)

    # 2) Narrow to just the PK + the new column
    backfill_df = full_md[["source_id", "position_names"]]

    # 3) Upsert only that column
    conn = sqlite3.connect(DB)
    from free_data_pull import upsert_table
    upsert_table(
        DB,
        table_name="market",
        df=backfill_df,
        key_columns=["source_id"],
        update_columns=["position_names"]
    )
    conn.close()
    print("✅ position_names backfill complete.")

if __name__ == "__main__":
    backfill_position_names()
