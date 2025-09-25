#!/usr/bin/env python3
"""Analyze final unknown markets for patterns"""

from database_v2 import db_manager
from models import Market
from collections import defaultdict
import re

def main():
    with db_manager.get_db_session() as db:
        # Get all Unknown markets
        unknown_markets = db.query(Market).filter(
            Market.sport == 'Unknown'
        ).limit(200).all()  # Analyze first 200
        
        print(f"Analyzing first 200 of Unknown markets...")
        print("=" * 80)
        
        # Check for common patterns in team names
        print("Sample Unknown markets:")
        for i, market in enumerate(unknown_markets[:50]):
            home = market.home_team
            away = market.away_team
            print(f"{i+1:3d}. {home} vs {away}")
        
        # Pattern detection
        patterns = {
            'esports_likely': [],
            'soccer_likely': [],
            'baseball_likely': [],
            'basketball_likely': [],
            'football_likely': [],
            'hockey_likely': [],
            'combat_likely': [],
            'individual_sport': [],
            'prop_bet': []
        }
        
        for market in unknown_markets:
            home = market.home_team
            away = market.away_team
            full = f"{home} vs {away}".lower()
            
            # eSports indicators
            if any(word in full for word in ['dplus', 'fearx', 'elevate', 'daystar', 'soul', 'esport', 'arrival']):
                patterns['esports_likely'].append(f"{home} vs {away}")
            
            # Soccer/Football clubs
            elif any(word in full for word in ['if', 'sk ', 'sv ', 'ca ', 'fc', 'deportivo', 'atletico', 'sporting']):
                patterns['soccer_likely'].append(f"{home} vs {away}")
            
            # Baseball teams
            elif any(word in full for word in ['tigers', 'guardians', 'rangers', 'pirates', 'twins']):
                patterns['baseball_likely'].append(f"{home} vs {away}")
            
            # Combat sports (individual names)
            elif re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+ vs [A-Z][a-z]+ [A-Z][a-z]+$', f"{home} vs {away}"):
                patterns['combat_likely'].append(f"{home} vs {away}")
            
            # Prop bets
            elif any(word in away.lower() for word in ['over', 'under', 'yes', 'no', '+', '-']):
                patterns['prop_bet'].append(f"{home} vs {away}")
        
        print("\n" + "=" * 80)
        print("Pattern Analysis:")
        for pattern, matches in patterns.items():
            if matches:
                print(f"\n{pattern}: {len(matches)} markets")
                for match in matches[:5]:
                    print(f"  - {match}")
                if len(matches) > 5:
                    print(f"  ... and {len(matches) - 5} more")

if __name__ == "__main__":
    main()