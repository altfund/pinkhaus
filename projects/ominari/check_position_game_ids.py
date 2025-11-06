#!/usr/bin/env python3
"""Check the game IDs in our positions"""

import os
import psycopg2

# Set PostgreSQL environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

conn = psycopg2.connect(
    host=os.environ['PG_HOST'],
    port=os.environ['PG_PORT'],
    user=os.environ['PG_USER'],
    password=os.environ['PG_PASSWORD'],
    database=os.environ['PG_DB']
)

try:
    with conn.cursor() as cur:
        # First check what tables exist
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE '%position%'
            OR table_name LIKE '%bet%'
            OR table_name LIKE '%paper%'
        """)
        
        tables = cur.fetchall()
        print("Available tables:")
        for table in tables:
            print(f"  - {table[0]}")
            
        # Check column names
        print("\nChecking paper_trading_positions columns:")
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns 
            WHERE table_name = 'paper_trading_positions'
            ORDER BY ordinal_position
        """)
        
        columns = cur.fetchall()
        for col in columns:
            print(f"  - {col[0]} ({col[1]})")
            
        # Now try with correct column names
        cur.execute("""
            SELECT 
                match_id,
                home_team,
                away_team,
                bet_on,
                kickoff_time,
                placed_at,
                stake,
                odds
            FROM paper_trading_positions
            WHERE status = 'pending'
            ORDER BY kickoff_time
            LIMIT 10
        """)
        
        positions = cur.fetchall()
        
        print(f"\nSample position Game IDs ({len(positions)} positions):")
        for pos in positions:
            print(f"\nMatch ID: {pos[0]}")
            print(f"  {pos[1]} vs {pos[2]}")
            print(f"  Bet on: {pos[3]}")
            print(f"  Kickoff: {pos[4]}")
            print(f"  Placed: {pos[5]}")
            print(f"  Stake: ${pos[6]:.2f} @ {pos[7]:.2f}")
            
finally:
    conn.close()