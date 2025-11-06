#!/usr/bin/env python3
"""Reset paper trading to start fresh with correct dates"""

import os
import psycopg2
from datetime import datetime, timezone

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
        # Delete all positions
        cur.execute("DELETE FROM paper_trading_positions")
        positions_deleted = cur.rowcount
        print(f"❌ Deleted {positions_deleted} positions")
        
        # Delete all sessions
        cur.execute("DELETE FROM paper_trading_sessions")
        sessions_deleted = cur.rowcount
        print(f"❌ Deleted {sessions_deleted} sessions")
        
        # Create a fresh session
        new_session_id = datetime.now(timezone.utc).strftime('session_%Y%m%d_%H%M%S')
        starting_bankroll = 10000.0
        
        # Check columns first
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'paper_trading_sessions'
            ORDER BY ordinal_position
        """)
        
        columns = [row[0] for row in cur.fetchall()]
        print(f"Session columns: {columns}")
        
        # Insert with correct columns
        cur.execute("""
            INSERT INTO paper_trading_sessions (
                session_id,
                session_name,
                initial_bankroll,
                created_at,
                status,
                strategy_config
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            new_session_id,
            'Fresh Start - November 2024',
            starting_bankroll,
            datetime.now(timezone.utc),
            'active',
            '{}'
        ))
        
        conn.commit()
        
        print(f"\n✅ Created fresh session: {new_session_id}")
        print(f"   Starting bankroll: ${starting_bankroll:,.2f}")
        print(f"   Status: Active")
        print("\n🎯 System is ready for fresh paper trading with correct dates!")
        
except Exception as e:
    print(f"Error: {e}")
    conn.rollback()
finally:
    conn.close()