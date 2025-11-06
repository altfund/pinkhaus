#!/usr/bin/env python3
"""Update trading snapshot to reflect current positions"""

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
        # Get current session
        cur.execute("""
            SELECT session_id, initial_bankroll
            FROM paper_trading_sessions
            WHERE status = 'active'
            ORDER BY created_at DESC
            LIMIT 1
        """)
        
        session = cur.fetchone()
        if not session:
            print("No active session found")
            exit(1)
            
        session_id = session[0]
        initial_bankroll = float(session[1])
        
        # Calculate current positions value
        cur.execute("""
            SELECT 
                COALESCE(SUM(stake), 0) as positions_value,
                COUNT(*) as position_count
            FROM paper_trading_positions
            WHERE session_id = %s AND status = 'pending'
        """, (session_id,))
        
        result = cur.fetchone()
        positions_value = float(result[0])
        position_count = result[1]
        
        # Calculate cash balance
        cash_balance = initial_bankroll - positions_value
        portfolio_value = initial_bankroll  # No P&L yet
        
        print(f"📊 Updating snapshot for session: {session_id}")
        print(f"   Positions: {position_count} (${positions_value:.2f})")
        print(f"   Cash: ${cash_balance:.2f}")
        print(f"   Portfolio: ${portfolio_value:.2f}")
        
        # Insert new snapshot
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
            cash_balance,
            positions_value,
            portfolio_value,
            0.0,  # total_pnl
            0.0,  # daily_pnl
            0,    # win_count
            0,    # loss_count
            position_count,  # pending_count
            0.0,  # max_drawdown
            datetime.now(timezone.utc)
        ))
        
        conn.commit()
        print("\n✅ Snapshot updated successfully!")
        print("The dashboard should now show the correct values.")
        
except Exception as e:
    print(f"Error: {e}")
    conn.rollback()
finally:
    conn.close()