#!/usr/bin/env python3
"""Check database for soccer markets with proper data"""

import os
from datetime import datetime, timezone, timedelta

# Set up database environment
os.environ['USE_POSTGRESQL'] = '1'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import and_, or_, func

print("⚽ Checking database for soccer markets...\n")

with db_manager.get_db_session() as db:
    # Get upcoming soccer markets
    now = datetime.now(timezone.utc)
    future = now + timedelta(days=7)
    
    soccer_markets = db.query(Market).filter(
        and_(
            Market.sport.ilike('%soccer%'),
            Market.is_finished == False,
            Market.maturity_date > now,
            Market.maturity_date < future
        )
    ).order_by(Market.maturity_date.asc()).limit(20).all()
    
    print(f"Found {len(soccer_markets)} upcoming soccer markets in next 7 days\n")
    
    if soccer_markets:
        for i, market in enumerate(soccer_markets[:10]):
            print(f"{i+1}. {market.home_team} vs {market.away_team}")
            print(f"   League: {market.league_name}")
            print(f"   Time: {market.maturity_date}")
            print(f"   Source: {market.source}")
            print(f"   Market ID: {market.source_id}")
            
            # Get odds
            odds = db.query(Odd).filter(
                Odd.source_id == market.source_id
            ).order_by(Odd.updated_at.desc()).limit(3).all()
            
            if odds:
                print(f"   Odds ({len(odds)} records):")
                for odd in odds:
                    val = odd.decimal_odds if odd.decimal_odds else odd.american_odds
                    print(f"     {odd.outcome}: {val}")
            else:
                print("   No odds found")
            print()
    
    # Check by league
    print("\n🏆 Soccer leagues in database:")
    leagues = db.query(Market.league_name, func.count(Market.id)).filter(
        and_(
            Market.sport.ilike('%soccer%'),
            Market.is_finished == False
        )
    ).group_by(Market.league_name).order_by(func.count(Market.id).desc()).limit(10).all()
    
    for league, count in leagues:
        print(f"  {league}: {count} markets")
    
    # Check sources with soccer data
    print("\n📡 Sources with soccer markets:")
    sources = db.query(Market.source, func.count(Market.id)).filter(
        and_(
            Market.sport.ilike('%soccer%'),
            Market.is_finished == False
        )
    ).group_by(Market.source).order_by(func.count(Market.id).desc()).all()
    
    for source, count in sources:
        print(f"  {source}: {count} markets")