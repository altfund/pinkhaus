#!/usr/bin/env python3
"""Debug why no opportunities are found"""

import os
from datetime import datetime, timedelta, timezone

os.environ['PG_PORT'] = '5999'
os.environ['PG_HOST'] = 'localhost' 
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import and_

with db_manager.get_db_session() as db:
    # Get markets happening in next 24 hours
    now = datetime.now(timezone.utc)
    future_time = now + timedelta(hours=24)
    
    markets = db.query(Market).filter(
        and_(
            Market.maturity_date > now,
            Market.maturity_date < future_time,
            Market.sport.in_(['Soccer', 'Tennis', 'American Football', 'Basketball'])
        )
    ).limit(5).all()
    
    print(f"Found {len(markets)} markets")
    
    for market in markets:
        print(f"\n{market.home_team} vs {market.away_team}")
        print(f"Sport: {market.sport}, Date: {market.maturity_date}")
        
        # Get odds
        odds_records = db.query(Odd).filter(
            Odd.source_id == market.source_id
        ).all()
        
        print(f"Odds records: {len(odds_records)}")
        
        # Group by outcome
        odds_by_outcome = {}
        for odd in odds_records:
            print(f"  - {odd.outcome}: {odd.decimal_odds} from {odd.bookmaker}")
            if odd.outcome not in odds_by_outcome or odd.decimal_odds > odds_by_outcome[odd.outcome].decimal_odds:
                odds_by_outcome[odd.outcome] = odd
                
        # Calculate fair probabilities
        if len(odds_by_outcome) >= 2:
            total_prob = sum(1/odd.decimal_odds for odd in odds_by_outcome.values())
            print(f"Total implied prob: {total_prob:.2%}")
            
            for outcome, odd in odds_by_outcome.items():
                implied_prob = 1 / odd.decimal_odds
                fair_prob = implied_prob / total_prob
                fair_odds = 1 / fair_prob
                edge = ((odd.decimal_odds / fair_odds) - 1) * 100
                print(f"  {outcome}: Edge={edge:.2f}%, Fair prob={fair_prob:.2%}")