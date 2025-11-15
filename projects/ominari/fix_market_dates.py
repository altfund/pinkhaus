#!/usr/bin/env python3
"""Fix market maturity dates to be in the near future"""

import os
import sys
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from load_env import load_dotenv
load_dotenv()

# Set up environment
os.environ['PG_PORT'] = '5999'

from sqlalchemy import or_
from database_v2 import db_manager
from models import Market, Odd

with db_manager.get_db_session() as db:
    # Find our test markets with positive edges
    test_markets = [
        "Portland Thorns FC",  # Our arbitrage market
        "Barcelona",
        "Real Madrid",
        "Manchester United",
        "Liverpool"
    ]
    
    # Update maturity dates to be 3-6 hours from now
    now = datetime.now(timezone.utc)
    
    for i, team in enumerate(test_markets):
        market = db.query(Market).filter(
            or_(Market.home_team.contains(team), Market.away_team.contains(team))
        ).first()
        
        if market:
            # Set maturity date to 3-6 hours from now
            new_date = now + timedelta(hours=3 + i)
            old_date = market.maturity_date
            market.maturity_date = new_date
            
            print(f"Updated {market.home_team} vs {market.away_team}")
            print(f"  Old date: {old_date}")
            print(f"  New date: {new_date}")
            
            # Check odds
            odds = db.query(Odd).filter(Odd.source_id == market.source_id).all()
            if odds:
                total_prob = sum(1/o.decimal_odds for o in odds)
                edge = ((1/total_prob) - 1) * 100 if total_prob < 1.0 else -((total_prob - 1) / total_prob) * 100
                print(f"  Edge: {edge:.2f}%")
            print()
    
    db.commit()
    print("✅ Updated market maturity dates to be in the next 3-8 hours")