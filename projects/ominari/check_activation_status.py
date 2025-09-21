#!/usr/bin/env python3
"""
Quick status check of all systems
"""
import os
import subprocess
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from datetime import datetime

# Set environment
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999',
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})

def check_status():
    print("🔍 OMINARI SYSTEM STATUS CHECK")
    print("=" * 50)
    
    # 1. Check web monitor
    print("\n1️⃣ Web Monitor Status:")
    result = subprocess.run("pgrep -f web_monitor.py", shell=True, capture_output=True)
    if result.returncode == 0:
        pid = result.stdout.decode().strip()
        print(f"   ✅ Running (PID: {pid})")
        
        # Check HTTP
        try:
            resp = requests.get("http://localhost:8888", timeout=2)
            print(f"   ✅ HTTP responding (status: {resp.status_code})")
        except:
            print("   ❌ HTTP not responding")
    else:
        print("   ❌ Not running")
    
    # 2. Check PostgreSQL
    print("\n2️⃣ PostgreSQL Status:")
    try:
        conn = psycopg2.connect(
            host=os.environ['PG_HOST'],
            port=os.environ['PG_PORT'],
            database=os.environ['PG_DB'],
            user=os.environ['PG_USER'],
            password=os.environ['PG_PASSWORD'],
            cursor_factory=RealDictCursor
        )
        cur = conn.cursor()
        print("   ✅ Connected to PostgreSQL")
        
        # Check active session
        cur.execute("""
            SELECT 
                s.session_id,
                s.session_name,
                s.initial_bankroll,
                snap.cash_balance,
                snap.portfolio_value,
                (SELECT COUNT(*) FROM paper_trading_positions WHERE session_id = s.session_id AND status = 'pending') as open_positions,
                (SELECT COALESCE(SUM(stake), 0) FROM paper_trading_positions WHERE session_id = s.session_id AND status = 'pending') as exposure
            FROM paper_trading_sessions s
            LEFT JOIN LATERAL (
                SELECT * FROM paper_trading_snapshots 
                WHERE session_id = s.session_id 
                ORDER BY snapshot_time DESC LIMIT 1
            ) snap ON true
            WHERE s.status = 'active' OR s.status = 'ACTIVE'
            ORDER BY s.created_at DESC
            LIMIT 1
        """)
        
        session = cur.fetchone()
        if session:
            exposure_pct = (float(session['exposure']) / float(session['initial_bankroll']) * 100) if session['initial_bankroll'] else 0
            
            print(f"\n   📊 Active Session: {session['session_id']}")
            print(f"      Name: {session['session_name']}")
            print(f"      Cash: ${session['cash_balance']:,.2f}")
            print(f"      Open positions: {session['open_positions']}")
            print(f"      Exposure: ${session['exposure']:,.2f} ({exposure_pct:.1f}%)")
            
            if exposure_pct > 100:
                print(f"      ⚠️  OVER-LEVERAGED!")
            elif exposure_pct > 80:
                print(f"      ⚠️  High exposure!")
            else:
                print(f"      ✅ Risk level OK")
        else:
            print("   ❌ No active session found")
        
        # Check odds availability
        print("\n3️⃣ Odds Data Status:")
        cur.execute("""
            SELECT 
                COUNT(DISTINCT CASE WHEN o.outcome = 'home' AND o.decimal_odds > 1 THEN m.source_id END) as home_markets,
                COUNT(DISTINCT CASE WHEN o.outcome = 'draw' AND o.decimal_odds > 1 THEN m.source_id END) as draw_markets,
                COUNT(DISTINCT CASE WHEN o.outcome = 'away' AND o.decimal_odds > 1 THEN m.source_id END) as away_markets,
                COUNT(DISTINCT m.source_id) as total_markets
            FROM market m
            LEFT JOIN odd o ON o.source_id = m.source_id
            WHERE m.is_finished = FALSE
            AND m.maturity_date > NOW()
            AND m.source = 'api_live_real'
        """)
        
        odds = cur.fetchone()
        if odds and odds['total_markets'] > 0:
            print(f"   Total upcoming markets: {odds['total_markets']}")
            print(f"   Home odds: {odds['home_markets']} ({odds['home_markets']/odds['total_markets']*100:.0f}%)")
            print(f"   Draw odds: {odds['draw_markets']} ({odds['draw_markets']/odds['total_markets']*100:.0f}%)")
            print(f"   Away odds: {odds['away_markets']} ({odds['away_markets']/odds['total_markets']*100:.0f}%)")
            
            if odds['draw_markets'] > 0 or odds['away_markets'] > 0:
                print("   ✅ Draw/Away odds are available!")
            else:
                print("   ⚠️  No Draw/Away odds found")
        
        conn.close()
        
    except Exception as e:
        print(f"   ❌ PostgreSQL error: {str(e)}")
    
    # 4. Check recent logs
    print("\n4️⃣ Recent Activity:")
    try:
        with open('web_monitor_activated.log', 'r') as f:
            lines = f.readlines()[-10:]
            if lines:
                print("   Last log entries:")
                for line in lines[-3:]:
                    print(f"      {line.strip()}")
    except:
        print("   No log file found")
    
    print("\n" + "=" * 50)
    print(f"🕒 Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    check_status()