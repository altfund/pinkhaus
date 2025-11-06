#!/usr/bin/env python3
"""Update match times to be within trading window"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market
from datetime import datetime, timezone, timedelta

with db_manager.get_db_session() as db:
    # Update first two matches to be within next 24 hours
    markets = db.query(Market).filter(
        Market.sport == 'Soccer',
        Market.is_finished == False
    ).order_by(Market.maturity_date).limit(2).all()
    
    if markets:
        # First match in 4 hours
        markets[0].maturity_date = datetime.now(timezone.utc) + timedelta(hours=4)
        print(f"Updated {markets[0].home_team} vs {markets[0].away_team} to kick off in 4 hours")
        
        if len(markets) > 1:
            # Second match in 8 hours
            markets[1].maturity_date = datetime.now(timezone.utc) + timedelta(hours=8)
            print(f"Updated {markets[1].home_team} vs {markets[1].away_team} to kick off in 8 hours")
        
        db.commit()
        print("\n✅ Match times updated successfully")