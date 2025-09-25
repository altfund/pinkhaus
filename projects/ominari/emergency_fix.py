#!/usr/bin/env python3
"""
EMERGENCY FIX: Reset paper trading session and fix over-leveraging
"""
import psycopg2
from datetime import datetime

def emergency_reset():
    print("🚨 EMERGENCY SYSTEM RESET")
    print("=" * 60)
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="ominari_trading",
            user="ominari_user",
            password="ominari_password"
        )
        cur = conn.cursor()
        
        # 1. Clear all paper trading sessions
        print("1. Clearing paper trading sessions...")
        cur.execute("DELETE FROM paper_trading_session")
        cur.execute("DELETE FROM paper_trade")
        conn.commit()
        print("   ✅ Cleared all sessions")
        
        # 2. Check market data
        print("\n2. Checking market data...")
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN home_odds > 0 THEN 1 ELSE 0 END) as home,
                SUM(CASE WHEN draw_odds > 0 THEN 1 ELSE 0 END) as draw,
                SUM(CASE WHEN away_odds > 0 THEN 1 ELSE 0 END) as away
            FROM (
                SELECT DISTINCT
                    m.source_id,
                    MAX(CASE WHEN o.outcome = 'home' THEN o.decimal_odds ELSE 0 END) as home_odds,
                    MAX(CASE WHEN o.outcome = 'draw' THEN o.decimal_odds ELSE 0 END) as draw_odds,
                    MAX(CASE WHEN o.outcome = 'away' THEN o.decimal_odds ELSE 0 END) as away_odds
                FROM market m
                LEFT JOIN odd o ON o.source_id = m.source_id
                WHERE m.is_finished = false
                GROUP BY m.source_id
            ) as odds_check
        """)
        
        result = cur.fetchone()
        print(f"   Total markets: {result[0]}")
        print(f"   Markets with HOME odds: {result[1]}")
        print(f"   Markets with DRAW odds: {result[2]}")
        print(f"   Markets with AWAY odds: {result[3]}")
        
        if result[2] == 0 and result[3] == 0:
            print("   ⚠️  WARNING: No DRAW or AWAY odds in database!")
            print("   This explains why only HOME bets are placed")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {str(e)}")
        return
    
    print("\n" + "=" * 60)
    print("📋 REQUIRED ACTIONS:")
    print("1. Stop the current web_monitor.py (Ctrl+C)")
    print("2. Start fresh with: python3 web_monitor.py")
    print("3. The chunking display should now appear")
    print("4. New session will start with proper limits")
    print("\n⚡ The fixed limits are:")
    print("   - 2% max per game (was 25%)")
    print("   - 1% max per bet (was 25%)")
    print("   - Risk alerts when exposure > 100%")

if __name__ == "__main__":
    emergency_reset()