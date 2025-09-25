#!/usr/bin/env python3
"""
Fix the odds query in web_monitor.py to properly fetch draw/away odds
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
from sqlalchemy import func, distinct

def analyze_odds_issue():
    """Analyze why draw/away odds aren't showing."""
    print("🔍 Analyzing Odds Query Issue")
    print("=" * 60)
    
    with db_manager.get_db_session() as db:
        # 1. Check outcome values
        print("\n1️⃣ Checking outcome values in odds table:")
        outcomes = db.query(Odd.outcome, func.count(Odd.id).label('count'))\
            .group_by(Odd.outcome)\
            .order_by('count').all()
        
        for outcome, count in outcomes:
            print(f"   '{outcome}': {count:,} records")
        
        # 2. Check position mapping
        print("\n2️⃣ Checking outcome-position mapping:")
        mappings = db.query(Odd.outcome, Odd.position, func.count(Odd.id))\
            .group_by(Odd.outcome, Odd.position)\
            .order_by(Odd.outcome, Odd.position).all()
        
        current_outcome = None
        for outcome, position, count in mappings:
            if outcome != current_outcome:
                current_outcome = outcome
                print(f"\n   {outcome}:")
            print(f"      Position {position}: {count:,} records")
        
        # 3. Sample complete markets
        print("\n3️⃣ Sampling markets with all outcomes:")
        
        # Find a market with all three outcomes
        sample_market = db.query(Market)\
            .join(Odd, Odd.source_id == Market.source_id)\
            .filter(Market.is_finished == False)\
            .group_by(Market.id, Market.source_id, Market.home_team, Market.away_team)\
            .having(func.count(distinct(Odd.outcome)) >= 3)\
            .first()
        
        if sample_market:
            print(f"\n📊 {sample_market.home_team} vs {sample_market.away_team}")
            odds = db.query(Odd).filter(Odd.source_id == sample_market.source_id).all()
            for odd in odds:
                print(f"   {odd.outcome}: {odd.decimal_odds} (position: {odd.position})")
        
        # 4. Test fixed query
        print("\n4️⃣ Testing fixed query (without position filter):")
        
        results = db.query(
            Market.source_id,
            Market.home_team,
            Market.away_team,
            func.max(func.case((Odd.outcome == 'home', Odd.decimal_odds))).label('home_odds'),
            func.max(func.case((Odd.outcome == 'draw', Odd.decimal_odds))).label('draw_odds'), 
            func.max(func.case((Odd.outcome == 'away', Odd.decimal_odds))).label('away_odds')
        ).join(
            Odd, Odd.source_id == Market.source_id
        ).filter(
            Market.is_finished == False,
            Market.source == 'api_live_real'
        ).group_by(
            Market.source_id,
            Market.home_team,
            Market.away_team
        ).limit(5).all()
        
        print("\n✅ Fixed query results:")
        for row in results:
            print(f"\n{row.home_team} vs {row.away_team}")
            print(f"   Home: {row.home_odds or 'NULL'}")
            print(f"   Draw: {row.draw_odds or 'NULL'}")
            print(f"   Away: {row.away_odds or 'NULL'}")
        
        # 5. Count markets with each outcome type
        print("\n5️⃣ Market coverage by outcome:")
        
        total_markets = db.query(func.count(distinct(Market.source_id)))\
            .filter(Market.is_finished == False).scalar()
        
        home_markets = db.query(func.count(distinct(Odd.source_id)))\
            .join(Market, Market.source_id == Odd.source_id)\
            .filter(Market.is_finished == False, Odd.outcome == 'home').scalar()
        
        draw_markets = db.query(func.count(distinct(Odd.source_id)))\
            .join(Market, Market.source_id == Odd.source_id)\
            .filter(Market.is_finished == False, Odd.outcome == 'draw').scalar()
        
        away_markets = db.query(func.count(distinct(Odd.source_id)))\
            .join(Market, Market.source_id == Odd.source_id)\
            .filter(Market.is_finished == False, Odd.outcome == 'away').scalar()
        
        print(f"\nTotal unfinished markets: {total_markets:,}")
        print(f"Markets with home odds: {home_markets:,} ({home_markets/total_markets*100:.1f}%)")
        print(f"Markets with draw odds: {draw_markets:,} ({draw_markets/total_markets*100:.1f}%)")
        print(f"Markets with away odds: {away_markets:,} ({away_markets/total_markets*100:.1f}%)")

def generate_fixed_query():
    """Generate the corrected query."""
    print("\n\n📝 CORRECTED QUERY FOR WEB_MONITOR.PY:")
    print("=" * 60)
    
    fixed_query = '''query = """
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
                    AND m.sport = %s
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
                LIMIT 100
                """'''
    
    print(fixed_query)
    print("\n⚠️  The issue was using position (0,1,2) instead of outcome values!")
    print("The position field doesn't reliably map to home/draw/away.")

if __name__ == "__main__":
    analyze_odds_issue()
    generate_fixed_query()