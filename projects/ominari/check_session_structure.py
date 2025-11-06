#!/usr/bin/env python3
"""Check session and snapshot structure"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor

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
    database=os.environ['PG_DB'],
    cursor_factory=RealDictCursor
)

try:
    with conn.cursor() as cur:
        # Check sessions
        cur.execute("""
            SELECT * FROM paper_trading_sessions LIMIT 1
        """)
        
        session = cur.fetchone()
        if session:
            print("Session columns:")
            for key, value in session.items():
                print(f"  {key}: {value} (type: {type(value).__name__})")
        
        # Check snapshots
        print("\n\nSnapshot table:")
        cur.execute("""
            SELECT * FROM paper_trading_snapshots LIMIT 1
        """)
        
        snapshot = cur.fetchone()
        if snapshot:
            print("Snapshot columns:")
            for key, value in snapshot.items():
                print(f"  {key}: {value}")
        else:
            print("No snapshots found")
            
        # Check if snapshots table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'paper_trading_snapshots'
            )
        """)
        
        exists = cur.fetchone()['exists']
        print(f"\nSnapshots table exists: {exists}")
            
finally:
    conn.close()