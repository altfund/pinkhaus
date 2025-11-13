#!/usr/bin/env python3
"""
Demonstrate nation filtering by querying the database
"""
import os

# Set environment for PostgreSQL on port 5999
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market
from sqlalchemy import func

def demo_nation_filtering():
    with db_manager.get_db_session() as session:
        # Show nation distribution
        print("=== Nation/Governing Body Distribution for Soccer ===")
        print("-" * 60)
        
        nation_stats = session.query(
            Market.nation,
            Market.governing_body,
            func.count(Market.source_id).label('count')
        ).filter(
            Market.sport == 'Soccer'
        ).group_by(
            Market.nation,
            Market.governing_body
        ).order_by(
            func.count(Market.source_id).desc()
        ).limit(15).all()
        
        print(f"{'Nation':<20} {'Governing Body':<30} {'Markets':<10}")
        print("-" * 60)
        for nation, gb, count in nation_stats:
            print(f"{nation or 'N/A':<20} {gb or 'N/A':<30} {count:<10}")
        
        # Show example markets by nation
        print("\n=== Example Markets by Nation ===")
        
        for nation in ['England', 'Spain', 'Italy']:
            print(f"\n{nation} Markets:")
            markets = session.query(Market).filter(
                Market.nation == nation,
                Market.sport == 'Soccer'
            ).limit(3).all()
            
            for market in markets:
                print(f"  - {market.home_team} vs {market.away_team}")
                print(f"    League: {market.league_name}")
                print(f"    Governing Body: {market.governing_body}")
                
        # Show how filtering would work
        print("\n=== Filtering Examples ===")
        
        # Filter for just England
        england_count = session.query(func.count(Market.source_id)).filter(
            Market.nation == 'England',
            Market.sport == 'Soccer'
        ).scalar()
        print(f"Markets in England only: {england_count}")
        
        # Filter for multiple nations
        multi_count = session.query(func.count(Market.source_id)).filter(
            Market.nation.in_(['England', 'Spain', 'Italy', 'Germany', 'France']),
            Market.sport == 'Soccer'
        ).scalar()
        print(f"Markets in top 5 European nations: {multi_count}")
        
        # Filter for international competitions
        intl_count = session.query(func.count(Market.source_id)).filter(
            Market.nation.in_(['Europe', 'International']),
            Market.sport == 'Soccer'
        ).scalar()
        print(f"International/Continental markets: {intl_count}")

if __name__ == "__main__":
    demo_nation_filtering()
    print("\n✅ Nation filtering is now fully implemented!")
    print("To use it, set the ALLOWED_NATIONS environment variable:")
    print("  export ALLOWED_NATIONS=England,Spain,Italy")
    print("Then restart the dashboard to see filtered results.")