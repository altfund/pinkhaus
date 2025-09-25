#!/usr/bin/env python3
"""
Check what Unknown markets actually are
"""

import sqlite3

DB_PATH = "sport_odds.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("🔍 Analyzing Unknown markets...")
print("=" * 80)

# Get sample of unknown markets
cursor.execute("""
    SELECT home_team, away_team, league_name 
    FROM market 
    WHERE source = 'api_live_real' AND sport = 'Unknown'
    LIMIT 30
""")

print("\nSample Unknown markets:")
for home, away, league in cursor.fetchall():
    print(f"  {home} vs {away} [{league or 'No league'}]")

# Analyze patterns
cursor.execute("""
    SELECT home_team, COUNT(*) as count 
    FROM market 
    WHERE source = 'api_live_real' AND sport = 'Unknown'
    GROUP BY home_team
    ORDER BY count DESC
    LIMIT 20
""")

print("\n📊 Most common home teams in Unknown:")
for team, count in cursor.fetchall():
    print(f"  {team}: {count} markets")

conn.close()