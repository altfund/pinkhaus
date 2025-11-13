#!/usr/bin/env python3
"""
Test nation filtering functionality
"""
import os

# Set environment for PostgreSQL on port 5999
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

# Test different nation filters
os.environ['ALLOWED_SPORTS'] = 'Soccer'
os.environ['ALLOWED_NATIONS'] = 'England,Spain,Italy'

from database_v2 import db_manager
from models import Market
from sqlalchemy import func

def test_nation_filtering():
    with db_manager.get_db_session() as session:
        # Get nation distribution
        print("=== Nation Distribution ===")
        nation_counts = session.query(
            Market.nation, 
            func.count(Market.source_id)
        ).filter(
            Market.sport == 'Soccer'
        ).group_by(
            Market.nation
        ).order_by(
            func.count(Market.source_id).desc()
        ).limit(20).all()
        
        for nation, count in nation_counts:
            print(f"{nation}: {count} markets")
        
        # Test specific nation filter
        print("\n=== Testing England Filter ===")
        england_markets = session.query(Market).filter(
            Market.nation == 'England',
            Market.sport == 'Soccer'
        ).limit(5).all()
        
        for market in england_markets:
            print(f"- {market.home_team} vs {market.away_team} ({market.league_name}) - {market.nation}")
        
        # Test multiple nation filter
        print("\n=== Testing Multiple Nations (England, Spain, Italy) ===")
        multi_nation_markets = session.query(Market).filter(
            Market.nation.in_(['England', 'Spain', 'Italy']),
            Market.sport == 'Soccer'
        ).limit(10).all()
        
        for market in multi_nation_markets:
            print(f"- {market.home_team} vs {market.away_team} ({market.league_name}) - {market.nation}")

if __name__ == "__main__":
    test_nation_filtering()