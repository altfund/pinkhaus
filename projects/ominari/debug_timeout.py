#!/usr/bin/env python3
"""Debug the timeout issue by checking data availability."""

import sys
import os
sys.path.insert(0, os.path.abspath('../pinkhaus-models'))

import sqlite3

# Check what data we have around the test timestamp
conn = sqlite3.connect('sport_odds.db')

# The timestamp used in test_quick_backtest.py
test_time = "2025-04-09 12:00:00"

print(f"Checking data around {test_time}...")

# Check if we have any data at that time
query = """
SELECT COUNT(DISTINCT source_id) as market_count,
       MIN(updated_at) as earliest,
       MAX(updated_at) as latest
FROM odd
WHERE updated_at BETWEEN datetime(?, '-1 day') AND datetime(?, '+1 day')
"""

cursor = conn.cursor()
cursor.execute(query, (test_time, test_time))
result = cursor.fetchone()

print(f"\nMarkets within ±1 day of test time: {result[0]}")
print(f"Earliest: {result[1]}")
print(f"Latest: {result[2]}")

# Get the actual date range of the data
cursor.execute("SELECT MIN(updated_at), MAX(updated_at) FROM odd")
min_date, max_date = cursor.fetchone()
print("\nActual data range in database:")
print(f"From: {min_date}")
print(f"To: {max_date}")

# Find a good test timestamp with data
cursor.execute("""
SELECT updated_at, COUNT(DISTINCT source_id) as cnt
FROM odd
WHERE updated_at >= datetime('now', '-30 days')
GROUP BY updated_at
ORDER BY cnt DESC
LIMIT 5
""")

print("\nBest timestamps to test (most markets):")
for ts, cnt in cursor.fetchall():
    print(f"  {ts}: {cnt} markets")

conn.close()