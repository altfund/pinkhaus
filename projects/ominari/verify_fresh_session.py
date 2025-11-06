#!/usr/bin/env python3
"""Verify fresh session is active"""

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
        # Get all sessions
        cur.execute("""
            SELECT session_id, session_name, initial_bankroll, status, created_at
            FROM paper_trading_sessions
            ORDER BY created_at DESC
        """)
        
        sessions = cur.fetchall()
        print(f"Found {len(sessions)} sessions:")
        
        for session in sessions:
            print(f"\n📊 Session: {session[0]}")
            print(f"   Name: {session[1]}")
            print(f"   Bankroll: ${session[2]:,.2f}")
            print(f"   Status: {session[3]}")
            print(f"   Created: {session[4]}")
            
        # Check positions for latest session
        if sessions:
            latest_session = sessions[0][0]
            cur.execute("""
                SELECT COUNT(*) as count
                FROM paper_trading_positions
                WHERE session_id = %s
            """, (latest_session,))
            
            count = cur.fetchone()[0]
            print(f"\n📈 Positions in latest session: {count}")
            
finally:
    conn.close()