#!/usr/bin/env python3
"""Reset paper trading session"""
import os
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999', 
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})

import psycopg2
from paper_trading_postgres_integrated import PaperTradingSessionManager

def reset_session():
    """Reset paper trading to a new session"""
    conn = psycopg2.connect(
        host=os.environ['PG_HOST'],
        port=os.environ['PG_PORT'],
        user=os.environ['PG_USER'],
        password=os.environ['PG_PASSWORD'],
        database=os.environ['PG_DB']
    )
    
    cur = conn.cursor()
    
    # Deactivate old sessions (check if status column exists)
    try:
        cur.execute("""
            UPDATE paper_trading_sessions 
            SET status = 'archived' 
            WHERE status = 'active' OR status = 'ACTIVE'
        """)
        affected = cur.rowcount
        print(f"Archived {affected} active sessions")
    except psycopg2.errors.UndefinedColumn:
        print("Status column not found, skipping archive")
    
    conn.commit()
    conn.close()
    
    # Create new session
    manager = PaperTradingSessionManager()
    new_session_id = manager.create_session(initial_bankroll=10000)
    print(f"Created new session: {new_session_id}")
    
    # Verify
    session = manager.get_session(new_session_id)
    print(f"New session details: {session}")
    
    return new_session_id

if __name__ == "__main__":
    reset_session()