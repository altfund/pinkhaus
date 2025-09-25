#!/usr/bin/env python3
"""Show remaining Unknown markets"""

from database_v2 import db_manager
from models import Market

def main():
    with db_manager.get_db_session() as db:
        # Get all Unknown markets
        unknown_markets = db.query(Market).filter(
            Market.sport == 'Unknown'
        ).all()
        
        print(f"Remaining {len(unknown_markets)} Unknown markets:")
        print("=" * 80)
        
        for i, market in enumerate(unknown_markets):
            print(f"{i+1:3d}. {market.home_team} vs {market.away_team}")
            
            # Try to identify what sport it might be
            full_text = f"{market.home_team} vs {market.away_team}".lower()
            if i < 20:  # Analyze first 20 in detail
                print(f"     Analysis: ", end="")
                if 'winner' in full_text or 'mvp' in full_text:
                    print("Likely a tournament/award market")
                elif any(char.isdigit() for char in full_text):
                    print("Contains numbers - could be prop bet or special market")
                elif len(market.home_team.split()) == 1 and len(market.away_team.split()) == 1:
                    print("Single word teams - could be any sport")
                else:
                    print("No clear pattern identified")

if __name__ == "__main__":
    main()