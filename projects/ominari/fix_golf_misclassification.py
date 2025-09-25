#!/usr/bin/env python3
"""
Fix sport misclassification for golf tournaments
"""

from database_v2 import db_manager
from models import Market
from sqlalchemy import or_
import os

# Golf tournament patterns
GOLF_TOURNAMENTS = [
    'scottish open',
    'masters',
    'us open',
    'british open',
    'pga championship',
    'ryder cup',
    'golf',
    'tournament leader',
    'end of round leader',
    'arnold palmer',
    'memorial tournament',
    'wells fargo',
    'players championship'
]

def fix_golf_misclassification():
    """Fix markets misclassified as other sports when they're actually golf."""
    
    with db_manager.get_db_session() as db:
        # Find all misclassified golf markets
        conditions = []
        for pattern in GOLF_TOURNAMENTS:
            conditions.extend([
                Market.home_team.ilike(f'%{pattern}%'),
                Market.away_team.ilike(f'%{pattern}%'),
                Market.league_name.ilike(f'%{pattern}%')
            ])
        
        misclassified = db.query(Market).filter(
            or_(*conditions),
            Market.sport != 'Golf'  # Not already marked as Golf
        ).all()
        
        print(f"Found {len(misclassified)} misclassified golf markets")
        
        # Group by current sport classification
        sport_counts = {}
        for market in misclassified:
            sport = market.sport or 'Unknown'
            sport_counts[sport] = sport_counts.get(sport, 0) + 1
        
        print("\nCurrent misclassifications:")
        for sport, count in sorted(sport_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sport}: {count}")
        
        # Show some examples
        print("\nExamples of misclassified markets:")
        for market in misclassified[:5]:
            print(f"  {market.sport} -> Golf: {market.home_team} vs {market.away_team}")
        
        # Update to Golf
        if misclassified:
            response = input("\nUpdate these markets to Golf? (y/n): ")
            if response.lower() == 'y':
                for market in misclassified:
                    market.sport = 'Golf'
                
                db.commit()
                print(f"\n✅ Updated {len(misclassified)} markets to Golf")
            else:
                print("❌ Cancelled - no changes made")
        else:
            print("\n✅ No misclassified golf markets found!")
        
        # Verify the update
        print("\n=== Current Sport Distribution ===")
        from sqlalchemy import func
        sports = db.query(Market.sport, func.count(Market.source_id)).group_by(Market.sport).all()
        for sport, count in sorted(sports, key=lambda x: x[1], reverse=True)[:15]:
            print(f"{sport}: {count}")

if __name__ == "__main__":
    # Set the correct database connection
    os.environ['PG_HOST'] = 'localhost'
    os.environ['PG_PORT'] = '5999'
    os.environ['PG_USER'] = 'ominari_user'
    os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
    os.environ['PG_DB'] = 'ominari_production'
    os.environ['USE_POSTGRESQL'] = '1'
    
    fix_golf_misclassification()