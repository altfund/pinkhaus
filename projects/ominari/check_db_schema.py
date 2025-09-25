#!/usr/bin/env python3
"""Check database schema"""
import sqlite3

conn = sqlite3.connect("sport_odds.db")
cursor = conn.cursor()

# Get market table schema
cursor.execute("PRAGMA table_info(market)")
print("Market table columns:")
for col in cursor.fetchall():
    print(f"  {col[1]} ({col[2]})")

print("\nOdd table columns:")
cursor.execute("PRAGMA table_info(odd)")
for col in cursor.fetchall():
    print(f"  {col[1]} ({col[2]})")

conn.close()