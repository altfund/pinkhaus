#!/usr/bin/env python3
"""Check why positions haven't settled after matches completed"""

import os
import json
from datetime import datetime, timezone

# Set PostgreSQL environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from paper_trading_postgres_integrated import PaperTradingSessionManager
from db_connection import get_db_connection

# Get connection
conn = get_db_connection()

try:
    with conn.cursor() as cur:
        # Get all pending positions
        cur.execute("""
            SELECT 
                bp.bet_id,
                bp.match_id,
                bp.home_team,
                bp.away_team,
                bp.bet_on,
                bp.odds,
                bp.stake,
                bp.created_at,
                bp.maturity_date,
                bp.status
            FROM betting_positions bp
            WHERE bp.status = 'pending'
            ORDER BY bp.maturity_date ASC
        """)
        
        positions = cur.fetchall()
        # Convert to list of dicts
        positions = [dict(row) for row in positions]
        print(f"📊 Found {len(positions)} pending positions")
        
        if not positions:
            print("No pending positions found")
        else:
            now = datetime.now(timezone.utc)
            
            # Check maturity dates
            expired_positions = []
            upcoming_positions = []
            
            for pos in positions:
                maturity = pos['maturity_date']
                if maturity.tzinfo is None:
                    maturity = maturity.replace(tzinfo=timezone.utc)
                
                if maturity < now:
                    expired_positions.append(pos)
                else:
                    upcoming_positions.append(pos)
            
            print(f"\n⏰ Position Status:")
            print(f"   Matches that should have finished: {len(expired_positions)}")
            print(f"   Matches still upcoming: {len(upcoming_positions)}")
            
            if expired_positions:
                print(f"\n🚨 Positions that should be settled (matches finished):")
                for i, pos in enumerate(expired_positions[:5]):  # Show first 5
                    hours_ago = (now - pos['maturity_date']).total_seconds() / 3600
                    print(f"\n   {i+1}. {pos['home_team']} vs {pos['away_team']}")
                    print(f"      Bet: {pos['bet_on'].upper()} @ {pos['odds']:.2f}")
                    print(f"      Stake: ${pos['stake']:.2f}")
                    print(f"      Match ended: {hours_ago:.1f} hours ago")
                    print(f"      Bet placed: {pos['created_at'].strftime('%Y-%m-%d %H:%M')}")
                
                if len(expired_positions) > 5:
                    print(f"\n   ... and {len(expired_positions) - 5} more expired positions")
            
            # Check if we have match results
            print(f"\n🔍 Checking for match results...")
            
            # Check markets table for is_finished status
            match_ids = [pos['match_id'] for pos in expired_positions[:5]]
            if match_ids:
                placeholders = ','.join(['%s'] * len(match_ids))
                cur.execute(f"""
                    SELECT 
                        match_id,
                        home_team,
                        away_team,
                        is_finished,
                        home_score,
                        away_score,
                        updated_at
                    FROM markets
                    WHERE match_id IN ({placeholders})
                """, match_ids)
                
                market_results = cur.fetchall()
                
                if market_results:
                    print(f"\n📋 Match status in markets table:")
                    for market in market_results:
                        print(f"\n   {market['home_team']} vs {market['away_team']}")
                        print(f"   Match ID: {market['match_id']}")
                        print(f"   Finished: {market['is_finished']}")
                        if market['home_score'] is not None:
                            print(f"   Score: {market['home_score']} - {market['away_score']}")
                        print(f"   Last updated: {market['updated_at']}")
                else:
                    print("   ❌ No market data found for these matches")
            
            # Check if there's a settlement process
            cur.execute("""
                SELECT COUNT(*) as settled_count 
                FROM betting_positions 
                WHERE status IN ('won', 'lost')
            """)
            settled = cur.fetchone()
            
            print(f"\n📈 Settlement History:")
            print(f"   Total settled positions: {settled['settled_count']}")
            
            if settled['settled_count'] == 0:
                print("\n⚠️  No positions have ever been settled!")
                print("   This suggests the settlement process is not running")
                print("\n🔧 Possible solutions:")
                print("   1. Check if paper_trading_postgres_integrated.py has a settlement function")
                print("   2. Look for a separate settlement script")
                print("   3. The Overtime API might need to be queried for match results")
                print("   4. Markets table might need is_finished and scores updated")

finally:
    conn.close()