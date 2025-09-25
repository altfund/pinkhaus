#!/usr/bin/env python3
"""Check remaining sport misclassifications after fix"""

from database_v2 import db_manager
from models import Market
from sqlalchemy import func
import re

def main():
    with db_manager.get_db_session() as db:
        # Get sport counts
        print("Current Sport Distribution:")
        print("-" * 40)
        
        sport_counts = db.query(
            Market.sport,
            func.count().label('count')
        ).group_by(Market.sport).order_by(func.count().desc()).all()
        
        for sport, count in sport_counts:
            print(f"{sport:20} {count:6d}")
        
        print("\n" + "=" * 60)
        print("Checking for remaining misclassifications...")
        print("=" * 60)
        
        # Golf patterns that might be misclassified
        golf_patterns = ['open', 'masters', 'championship', 'pga', 'tour', 
                        'round 1', 'round 2', 'round 3', 'round 4',
                        'end of round', 'leader', 'golf']
        
        print("\n1. Checking for Golf markets classified as other sports:")
        for sport, _ in sport_counts:
            if sport != 'Golf':
                markets = db.query(Market).filter(
                    Market.sport == sport
                ).limit(1000).all()
                
                golf_like = []
                for market in markets:
                    teams = f"{market.home_team} {market.away_team}".lower()
                    if any(pattern in teams for pattern in golf_patterns):
                        golf_like.append(market)
                
                if golf_like:
                    print(f"\n  {sport} - Found {len(golf_like)} potential golf markets:")
                    for m in golf_like[:5]:
                        print(f"    - {m.home_team} vs {m.away_team}")
                    if len(golf_like) > 5:
                        print(f"    ... and {len(golf_like) - 5} more")

if __name__ == "__main__":
    main()