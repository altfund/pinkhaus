#!/usr/bin/env python3
"""
Fix over-leveraging issue in PostgreSQL paper trading
Resets the current session and enforces proper risk limits
"""
import os
import psycopg2
from datetime import datetime

# Set environment variables
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999',
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})

def fix_overleveraging():
    """Fix the over-leveraging crisis."""
    print("🚨 EMERGENCY FIX: Addressing Over-Leveraging Crisis")
    print("=" * 70)
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=os.environ['PG_HOST'],
            port=os.environ['PG_PORT'],
            database=os.environ['PG_DB'],
            user=os.environ['PG_USER'],
            password=os.environ['PG_PASSWORD']
        )
        cur = conn.cursor()
        
        # 1. Check current situation
        print("\n1️⃣ Checking current situation...")
        cur.execute("""
            SELECT 
                s.session_id,
                s.session_name,
                s.initial_bankroll,
                snap.cash_balance,
                snap.portfolio_value,
                (SELECT COUNT(*) FROM paper_trading_positions WHERE session_id = s.session_id AND status = 'pending') as open_positions,
                (SELECT COALESCE(SUM(stake), 0) FROM paper_trading_positions WHERE session_id = s.session_id AND status = 'pending') as total_exposure
            FROM paper_trading_sessions s
            LEFT JOIN LATERAL (
                SELECT * FROM paper_trading_snapshots 
                WHERE session_id = s.session_id 
                ORDER BY snapshot_time DESC 
                LIMIT 1
            ) snap ON true
            WHERE s.status = 'active' OR s.status = 'ACTIVE'
            ORDER BY s.created_at DESC
            LIMIT 1
        """)
        
        result = cur.fetchone()
        if result:
            session_id, session_name, initial_bankroll, cash, portfolio, open_pos, exposure = result
            exposure_pct = (float(exposure) / float(initial_bankroll) * 100) if initial_bankroll else 0
            
            print(f"📊 Current session: {session_id}")
            print(f"   - Name: {session_name}")
            print(f"   - Initial bankroll: ${initial_bankroll:,.2f}")
            print(f"   - Current cash: ${cash:,.2f}")
            print(f"   - Open positions: {open_pos}")
            print(f"   - Total exposure: ${exposure:,.2f} ({exposure_pct:.1f}%)")
            
            if exposure_pct > 100:
                print(f"   ⚠️  OVER-LEVERAGED by {exposure_pct - 100:.1f}%!")
        else:
            print("❌ No active session found")
            return
        
        # 2. Ask for confirmation
        print(f"\n⚠️  This will:")
        print(f"   1. Close all {open_pos} open positions")
        print(f"   2. Reset bankroll to ${initial_bankroll:,.2f}")
        print(f"   3. Create a fresh start with proper risk limits")
        
        response = input("\nProceed with fix? (yes/no): ").lower()
        if response != 'yes':
            print("❌ Fix cancelled")
            return
        
        # 3. Close all open positions
        print("\n3️⃣ Closing all open positions...")
        cur.execute("""
            UPDATE paper_trading_positions
            SET status = 'cancelled',
                settled_at = NOW(),
                actual_return = 0,
                pnl = -stake,
                result = 'cancelled'
            WHERE session_id = %s AND status = 'pending'
        """, (session_id,))
        closed_count = cur.rowcount
        print(f"✅ Closed {closed_count} positions")
        
        # 4. Reset session
        print("\n4️⃣ Resetting session...")
        
        # Create final snapshot showing the damage
        cur.execute("""
            INSERT INTO paper_trading_snapshots
            (session_id, cash_balance, positions_value, portfolio_value, 
             total_pnl, win_count, loss_count, pending_count)
            VALUES (%s, %s, 0, %s, %s, 0, 0, 0)
        """, (session_id, cash, cash, float(cash) - float(initial_bankroll)))
        
        # Mark old session as closed
        cur.execute("""
            UPDATE paper_trading_sessions
            SET status = 'closed'
            WHERE session_id = %s
        """, (session_id,))
        
        # 5. Create new session
        print("\n5️⃣ Creating fresh session with proper limits...")
        new_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        cur.execute("""
            INSERT INTO paper_trading_sessions 
            (session_id, session_name, initial_bankroll, status, metadata)
            VALUES (%s, %s, %s, 'active', %s::jsonb)
        """, (
            new_session_id,
            f"Fresh Start - Proper Risk Management",
            10000,
            '{"risk_limits": {"max_per_game": 0.02, "max_per_bet": 0.01, "max_exposure": 1.0}}'
        ))
        
        cur.execute("""
            INSERT INTO paper_trading_snapshots 
            (session_id, cash_balance, portfolio_value)
            VALUES (%s, 10000, 10000)
        """, (new_session_id,))
        
        # 6. Verify odds data
        print("\n6️⃣ Checking odds data...")
        cur.execute("""
            SELECT 
                COUNT(DISTINCT m.id) as total_markets,
                COUNT(DISTINCT CASE WHEN o.outcome = 'home' AND o.decimal_odds > 1 THEN m.id END) as home_odds,
                COUNT(DISTINCT CASE WHEN o.outcome = 'draw' AND o.decimal_odds > 1 THEN m.id END) as draw_odds,
                COUNT(DISTINCT CASE WHEN o.outcome = 'away' AND o.decimal_odds > 1 THEN m.id END) as away_odds
            FROM market m
            LEFT JOIN odd o ON o.market_id = m.id
            WHERE m.is_finished = false
            AND m.kickoff_time > NOW()
        """)
        
        odds_result = cur.fetchone()
        if odds_result:
            total, home, draw, away = odds_result
            print(f"✅ Markets with odds:")
            print(f"   - Total upcoming: {total}")
            print(f"   - HOME odds: {home} ({home/total*100:.1f}%)")
            print(f"   - DRAW odds: {draw} ({draw/total*100:.1f}%)")
            print(f"   - AWAY odds: {away} ({away/total*100:.1f}%)")
            
            if draw == 0 and away == 0:
                print("\n   ⚠️  WARNING: No DRAW or AWAY odds available!")
                print("   This is why only HOME bets are being placed.")
                print("   The odds data needs to be fixed for proper diversification.")
        
        # Commit all changes
        conn.commit()
        conn.close()
        
        print("\n" + "=" * 70)
        print("✅ FIX COMPLETE!")
        print(f"\n📋 Summary:")
        print(f"   - Closed {closed_count} over-leveraged positions")
        print(f"   - Created new session: {new_session_id}")
        print(f"   - Starting bankroll: $10,000")
        print(f"   - Risk limits: 2% per game, 1% per bet")
        
        print("\n🎯 Next steps:")
        print("1. Restart web_monitor.py to use the new session")
        print("2. Monitor with: ./simple_monitor.sh")
        print("3. Dashboard should show proper exposure (<100%)")
        print("\n⚠️  Note: Fix the odds data to enable DRAW/AWAY betting")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_overleveraging()