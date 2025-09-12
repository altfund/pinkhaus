#!/usr/bin/env python3
"""Check soccer markets safely using ORM."""

from database_v2 import db_manager
from models import Market, Odd
from datetime import datetime, timezone

def check_markets():
    """Check active soccer markets."""
    with db_manager.get_db_session() as db:
        # Count total active soccer markets
        active_count = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        print(f"Active soccer markets: {active_count}")
        
        # Get sample of upcoming matches
        upcoming = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).order_by(Market.maturity_date).limit(10).all()
        
        print("\nNext 10 upcoming soccer matches:")
        for market in upcoming:
            print(f"- {market.home_team} vs {market.away_team} ({market.league_name}) - {market.maturity_date}")
        
        # Check if odds exist for these markets
        if upcoming:
            first_market = upcoming[0]
            odds_count = db.query(Odd).filter(
                Odd.source_id == first_market.source_id
            ).count()
            print(f"\nOdds records for '{first_market.home_team} vs {first_market.away_team}': {odds_count}")

if __name__ == "__main__":
    check_markets()