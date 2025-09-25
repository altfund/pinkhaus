#!/usr/bin/env python3
"""
Check odds structure to understand the data format
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

def check_odds_structure():
    """Check the structure of odds data."""
    print("Checking odds structure...\n")
    
    with db_manager.get_db_session() as db:
        # Get a market with odds
        market = db.query(Market).filter(
            Market.maturity_date > datetime.now(timezone.utc),
            Market.is_finished == False
        ).first()
        
        if not market:
            print("No active markets found")
            return
            
        print(f"Market: {market.home_team} vs {market.away_team}")
        print(f"Source ID: {market.source_id}\n")
        
        # Get ALL fields from odds
        odds = db.query(Odd).filter(
            Odd.source_id == market.source_id
        ).limit(10).all()
        
        print(f"Found {len(odds)} odds records:\n")
        
        for i, odd in enumerate(odds):
            print(f"Odd {i+1}:")
            print(f"  ID: {odd.id}")
            print(f"  Source ID: {odd.source_id}")
            print(f"  Source: {odd.source}")
            print(f"  Bookmaker: {odd.bookmaker}")
            print(f"  Market Type: {odd.market_type}")
            print(f"  Outcome: {odd.outcome}")
            print(f"  Position: {odd.position}")
            print(f"  Line: {odd.line}")
            print(f"  Decimal Odds: {odd.decimal_odds}")
            print(f"  American Odds: {odd.american_odds}")
            print(f"  Normalized Implied: {odd.normalized_implied}")
            print(f"  Updated At: {odd.updated_at}")
            print()
        
        # Check unique outcomes
        unique_outcomes = db.query(Odd.outcome).distinct().limit(20).all()
        print("\nUnique outcomes in database:")
        for outcome in unique_outcomes:
            print(f"  - {outcome[0]}")

if __name__ == "__main__":
    check_odds_structure()