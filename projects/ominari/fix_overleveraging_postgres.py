#\!/usr/bin/env python3
"""Fix over-leveraging in paper trading by adding proper exposure controls"""
import os
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999', 
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})

import psycopg2
from psycopg2.extras import RealDictCursor

def check_current_exposure():
    """Check current exposure levels"""
    conn = psycopg2.connect(
        host=os.environ['PG_HOST'],
        port=os.environ['PG_PORT'],
        user=os.environ['PG_USER'],
        password=os.environ['PG_PASSWORD'],
        database=os.environ['PG_DB']
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get latest session
    cur.execute("""
        SELECT session_id, initial_bankroll
        FROM paper_trading_sessions
        ORDER BY created_at DESC
        LIMIT 1
    """)
    session = cur.fetchone()
    print(f"Session: {session['session_id']}")
    print(f"Initial bankroll: ${session['initial_bankroll']:,.2f}")
    
    # Get open positions
    cur.execute("""
        SELECT COUNT(*) as count, 
               SUM(stake) as total_stake,
               COUNT(DISTINCT match_id) as unique_markets
        FROM paper_trading_positions
        WHERE session_id = %s 
        AND status IN ('pending', 'open')
    """, (session['session_id'],))
    
    exposure = cur.fetchone()
    print(f"\nOpen positions: {exposure['count']}")
    print(f"Total stake: ${exposure['total_stake']:,.2f}")
    print(f"Unique markets: {exposure['unique_markets']}")
    print(f"Exposure %: {(exposure['total_stake'] / session['initial_bankroll'] * 100):,.1f}%")
    
    # Check for duplicates
    cur.execute("""
        SELECT match_id, bet_on, COUNT(*) as count, SUM(stake) as total_stake
        FROM paper_trading_positions
        WHERE session_id = %s
        AND status IN ('pending', 'open')
        GROUP BY match_id, bet_on
        HAVING COUNT(*) > 1
        ORDER BY COUNT(*) DESC
        LIMIT 10
    """, (session['session_id'],))
    
    duplicates = cur.fetchall()
    if duplicates:
        print("\n⚠️  DUPLICATE BETS FOUND:")
        for dup in duplicates:
            print(f"  Market {dup['match_id']} - {dup['bet_on']}: {dup['count']} bets, ${dup['total_stake']:,.2f} total")
    
    conn.close()

if __name__ == "__main__":
    check_current_exposure()
