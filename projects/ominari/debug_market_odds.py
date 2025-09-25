#!/usr/bin/env python3
"""
Debug market odds to understand why trades aren't happening
"""

import os
from datetime import datetime, timezone, timedelta

# Set PostgreSQL environment first
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market, Odd

def debug_markets():
    """Debug available markets and their odds."""
    print("Debugging markets and odds...\n")
    
    with db_manager.get_db_session() as db:
        # Get upcoming markets
        now = datetime.now(timezone.utc)
        markets = db.query(Market).filter(
            Market.maturity_date > now,
            Market.maturity_date < now + timedelta(hours=24),
            Market.is_finished == False
        ).limit(20).all()
        
        print(f"Found {len(markets)} upcoming markets within 24 hours:\n")
        
        for i, market in enumerate(markets):
            print(f"Market {i+1}: {market.home_team} vs {market.away_team}")
            print(f"  Sport: {market.sport}")
            print(f"  League: {market.league_name}")
            print(f"  Starts at: {market.maturity_date}")
            print(f"  Source ID: {market.source_id}")
            
            # Get odds for this market
            odds = db.query(Odd).filter(
                Odd.source_id == market.source_id
            ).order_by(Odd.updated_at.desc()).limit(10).all()
            
            print(f"  Found {len(odds)} odds records:")
            
            # Group by outcome
            home_odds = [o for o in odds if o.outcome == 'Home']
            away_odds = [o for o in odds if o.outcome == 'Away']
            draw_odds = [o for o in odds if o.outcome == 'Draw']
            
            if home_odds:
                print(f"    Home: {home_odds[0].decimal_odds:.2f} (Updated: {home_odds[0].updated_at})")
            if away_odds:
                print(f"    Away: {away_odds[0].decimal_odds:.2f} (Updated: {away_odds[0].updated_at})")
            if draw_odds:
                print(f"    Draw: {draw_odds[0].decimal_odds:.2f} (Updated: {draw_odds[0].updated_at})")
            
            # Check if this would trigger underdog bet
            if home_odds and away_odds:
                home_decimal = home_odds[0].decimal_odds
                away_decimal = away_odds[0].decimal_odds
                
                if home_decimal > away_decimal and home_decimal > 2.5:
                    print(f"  ⭐ HOME UNDERDOG OPPORTUNITY: {home_decimal:.2f} > 2.5")
                elif away_decimal > home_decimal and away_decimal > 2.5:
                    print(f"  ⭐ AWAY UNDERDOG OPPORTUNITY: {away_decimal:.2f} > 2.5")
            
            print()

if __name__ == "__main__":
    debug_markets()