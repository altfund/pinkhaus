#!/usr/bin/env python3
"""
Check what blockchain markets we have
"""

import sqlite3

conn = sqlite3.connect("sport_odds.db")
cursor = conn.cursor()

print("📊 BLOCKCHAIN MARKETS ANALYSIS")
print("=" * 60)

# Check market sources
cursor.execute("""
    SELECT source, COUNT(*) as count
    FROM market
    WHERE source LIKE '%blockchain%'
    GROUP BY source
    ORDER BY count DESC
""")

print("\nMarkets by source:")
for source, count in cursor.fetchall():
    print(f"  {source}: {count} markets")

# Check sample market IDs
cursor.execute("""
    SELECT source_id, home_team, away_team
    FROM market
    WHERE source LIKE '%blockchain%'
    LIMIT 10
""")

print("\n\nSample market IDs:")
for source_id, home, away in cursor.fetchall():
    print(f"  {source_id}")
    print(f"    {home} vs {away}")

# Check if we have odds
cursor.execute("""
    SELECT COUNT(DISTINCT m.source_id) as markets_with_odds
    FROM market m
    JOIN odd o ON m.source_id = o.source_id
    WHERE m.source LIKE '%blockchain%'
""")

markets_with_odds = cursor.fetchone()[0]
print(f"\n\nMarkets with odds: {markets_with_odds}")

# Check recent blockchain odds
cursor.execute("""
    SELECT m.home_team, m.away_team, o.decimal_odds, o.outcome, o.updated_at
    FROM market m
    JOIN odd o ON m.source_id = o.source_id
    WHERE m.source LIKE '%blockchain%'
    ORDER BY o.updated_at DESC
    LIMIT 10
""")

print("\n\nRecent blockchain odds:")
for home, away, odds, outcome, updated in cursor.fetchall():
    print(f"  {home} vs {away}: {outcome} = {odds} (updated: {updated})")

conn.close()