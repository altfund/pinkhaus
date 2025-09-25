#!/usr/bin/env python3
"""Fix remaining golf misclassifications"""

from database_v2 import db_manager
from models import Market
from sqlalchemy import func
import re

def main():
    # Enhanced golf patterns - more comprehensive
    golf_patterns = [
        # Tournament names
        'open', 'masters', 'championship', 'pga', 'tour', 'cup',
        'invitational', 'classic', 'memorial', 'players championship',
        
        # Round indicators
        'round 1', 'round 2', 'round 3', 'round 4', 'final round',
        'end of round', 'leader', 'cut line', 'playoff',
        
        # Specific tournaments
        'genesis', 'phoenix', 'farmers', 'waste management', 'honda',
        'arnold palmer', 'byron nelson', 'colonial', 'memorial',
        'travelers', 'john deere', 'barracuda', 'wyndham',
        'northern trust', 'bmw', 'tour championship', 'zozo',
        'cj cup', 'shriners', 'sanderson', 'houston open',
        'bermuda', 'mayakoba', 'rbc', 'wells fargo', 'valero',
        'zurich', 'at&t', 'sony', 'american express', 'sentry',
        
        # Golf-specific terms
        'golf', 'golfer', 'lpga', 'european tour', 'korn ferry',
        'liv golf', 'dp world', 'fedex', 'scottish open',
        'british open', 'us open', 'french open', 'irish open'
    ]
    
    with db_manager.get_db_session() as db:
        print("Checking for golf markets misclassified as other sports...")
        
        # Get all non-golf markets
        non_golf_count = db.query(func.count()).select_from(Market).filter(
            Market.sport != 'Golf'
        ).scalar()
        
        print(f"Total non-Golf markets: {non_golf_count}")
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        total_fixed = 0
        
        for offset in range(0, non_golf_count, batch_size):
            markets = db.query(Market).filter(
                Market.sport != 'Golf'
            ).offset(offset).limit(batch_size).all()
            
            batch_fixed = 0
            for market in markets:
                teams_text = f"{market.home_team} {market.away_team}".lower()
                
                # Check if it matches golf patterns
                if any(pattern in teams_text for pattern in golf_patterns):
                    # Additional checks to avoid false positives
                    # Skip if it's clearly another sport
                    if any(word in teams_text for word in ['fc', 'united', 'city fc', 'yankees', 'dodgers', 'lakers', 'celtics']):
                        continue
                    
                    # Likely golf - update it
                    print(f"Fixing: {market.home_team} vs {market.away_team} (was: {market.sport})")
                    market.sport = 'Golf'
                    batch_fixed += 1
            
            if batch_fixed > 0:
                db.commit()
                total_fixed += batch_fixed
                print(f"Fixed {batch_fixed} markets in this batch (total: {total_fixed})")
        
        print(f"\n✅ Total golf markets fixed: {total_fixed}")
        
        # Show final counts
        print("\nFinal sport distribution:")
        sport_counts = db.query(
            Market.sport,
            func.count().label('count')
        ).group_by(Market.sport).order_by(func.count().desc()).all()
        
        for sport, count in sport_counts:
            print(f"  {sport:20} {count:6d}")

if __name__ == "__main__":
    main()