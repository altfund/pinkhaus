#!/usr/bin/env python3
"""Check what outcome values we have for a market."""

from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import func

with db_manager.get_db_session() as db:
    # Get first soccer market
    market = db.query(Market).filter(
        Market.sport == 'Soccer',
        Market.is_finished == False
    ).first()
    
    if market:
        print(f"Market: {market.home_team} vs {market.away_team}")
        print(f"Source ID: {market.source_id}")
        
        # Get all outcomes for this market
        outcomes = db.query(Odd.outcome, func.count(Odd.id)).filter(
            Odd.source_id == market.source_id
        ).group_by(Odd.outcome).all()
        
        print(f"\nOutcomes found ({len(outcomes)}):")
        for outcome, count in outcomes:
            print(f"  - '{outcome}': {count} odds")