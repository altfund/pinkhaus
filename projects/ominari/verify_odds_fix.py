#!/usr/bin/env python3
"""
Verify that the odds query fix is working
"""
import os

# Set environment
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999',
    'PG_USER': 'ominari_user', 
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})

from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import text

def verify_odds_fix():
    """Verify the fixed query returns draw/away odds."""
    print("✅ Verifying Odds Fix")
    print("=" * 60)
    
    with db_manager.get_db_session() as db:
        # Test the fixed query
        fixed_query = text("""
            SELECT 
                m.source_id as market_id,
                m.source_id,
                m.home_team,
                m.away_team,
                m.sport,
                m.league_name,
                m.maturity_date,
                m.is_finished,
                m.source,
                MAX(CASE WHEN o.outcome = 'home' THEN o.decimal_odds END) as home_odds,
                MAX(CASE WHEN o.outcome = 'draw' THEN o.decimal_odds END) as draw_odds,
                MAX(CASE WHEN o.outcome = 'away' THEN o.decimal_odds END) as away_odds
            FROM market m
            LEFT JOIN odd o ON o.source_id = m.source_id
            WHERE 
                m.source = 'api_live_real'
                AND m.is_finished = FALSE
                AND m.maturity_date > NOW()
                AND m.sport = :sport
            GROUP BY 
                m.source_id,
                m.home_team,
                m.away_team,
                m.sport,
                m.league_name,
                m.maturity_date,
                m.is_finished,
                m.source
            HAVING 
                MAX(CASE WHEN o.outcome = 'home' THEN o.decimal_odds END) IS NOT NULL
                OR MAX(CASE WHEN o.outcome = 'draw' THEN o.decimal_odds END) IS NOT NULL
                OR MAX(CASE WHEN o.outcome = 'away' THEN o.decimal_odds END) IS NOT NULL
            ORDER BY m.maturity_date ASC
            LIMIT 20
        """)
        
        # Test with Soccer
        print("\n🏈 Testing with Soccer:")
        results = db.execute(fixed_query, {'sport': 'Soccer'}).fetchall()
        
        if not results:
            print("❌ No Soccer markets found")
        else:
            print(f"✅ Found {len(results)} markets\n")
            
            # Count odds availability
            home_count = sum(1 for r in results if r.home_odds is not None and r.home_odds > 0)
            draw_count = sum(1 for r in results if r.draw_odds is not None and r.draw_odds > 0)
            away_count = sum(1 for r in results if r.away_odds is not None and r.away_odds > 0)
            
            print(f"📊 Odds Distribution:")
            print(f"   Home odds: {home_count}/{len(results)} ({home_count/len(results)*100:.0f}%)")
            print(f"   Draw odds: {draw_count}/{len(results)} ({draw_count/len(results)*100:.0f}%)")
            print(f"   Away odds: {away_count}/{len(results)} ({away_count/len(results)*100:.0f}%)")
            
            # Show samples
            print(f"\n📝 Sample markets:")
            for i, row in enumerate(results[:5]):
                print(f"\n{i+1}. {row.home_team} vs {row.away_team}")
                print(f"   Home: {row.home_odds or 'NULL'}")
                print(f"   Draw: {row.draw_odds or 'NULL'}")
                print(f"   Away: {row.away_odds or 'NULL'}")
                
                # Flag if missing draw/away
                if row.draw_odds is None or row.away_odds is None:
                    print(f"   ⚠️  Missing {'draw' if row.draw_odds is None else ''} {'away' if row.away_odds is None else ''} odds!")
        
        # Test with Basketball
        print("\n\n🏀 Testing with Basketball:")
        results = db.execute(fixed_query, {'sport': 'Basketball'}).fetchall()
        
        if results:
            print(f"✅ Found {len(results)} markets")
            # Basketball usually doesn't have draw odds
            home_count = sum(1 for r in results if r.home_odds is not None)
            away_count = sum(1 for r in results if r.away_odds is not None)
            draw_count = sum(1 for r in results if r.draw_odds is not None)
            
            print(f"   Home odds: {home_count}/{len(results)}")
            print(f"   Away odds: {away_count}/{len(results)}")
            print(f"   Draw odds: {draw_count}/{len(results)} (should be 0 for basketball)")
    
    print("\n" + "=" * 60)
    print("🎯 Fix Status:")
    print("   ✅ Query updated to use outcome values instead of position")
    print("   ✅ Should now properly fetch draw/away odds where available")
    print("\n⚠️  If draw/away are still missing, the data might not be in the database")

if __name__ == "__main__":
    verify_odds_fix()