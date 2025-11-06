#!/usr/bin/env python3
"""Create initial snapshot for fresh session"""

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
        # Get the fresh session
        cur.execute("""
            SELECT session_id, initial_bankroll
            FROM paper_trading_sessions
            WHERE session_id LIKE 'session_20251102%'
        """)
        
        session = cur.fetchone()
        if session:
            session_id = session[0]
            initial_bankroll = float(session[1])
            
            # Create initial snapshot
            cur.execute("""
                INSERT INTO paper_trading_snapshots (
                    session_id,
                    snapshot_time,
                    cash_balance,
                    positions_value,
                    portfolio_value,
                    total_pnl,
                    daily_pnl,
                    win_count,
                    loss_count,
                    pending_count,
                    max_drawdown,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                session_id,
                datetime.now(timezone.utc),
                initial_bankroll,
                0.0,
                initial_bankroll,
                0.0,
                0.0,
                0,
                0,
                0,
                0.0,
                datetime.now(timezone.utc)
            ))
            
            conn.commit()
            
            print(f"✅ Created initial snapshot for session: {session_id}")
            print(f"   Initial bankroll: ${initial_bankroll:,.2f}")
        else:
            print("❌ No fresh session found")
            
finally:
    conn.close()