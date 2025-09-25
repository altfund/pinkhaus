#!/usr/bin/env python3
"""
Check what markets can be displayed in the dashboard
"""

import os
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market, Odd
from datetime import datetime, timezone
from sqlalchemy import func

def check_markets():
    with db_manager.get_db_session() as db:
        # Total markets
        total = db.query(Market).count()
        print(f"Total markets: {total}")
        
        # Active markets
        active = db.query(Market).filter(Market.is_finished == False).count()
        print(f"Active markets: {active}")
        
        # Future markets
        now_utc = datetime.now(timezone.utc)
        future = db.query(Market).filter(
            Market.maturity_date > now_utc,
            Market.is_finished == False
        ).count()
        print(f"Future markets: {future}")
        
        # Soccer markets (what the dashboard filters for)
        soccer_future = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False,
            Market.maturity_date > now_utc
        ).count()
        print(f"Future Soccer markets: {soccer_future}")
        
        # All sports breakdown
        print("\nSports breakdown:")
        sports = db.query(
            Market.sport,
            func.count(Market.source_id)
        ).filter(
            Market.is_finished == False,
            Market.maturity_date > now_utc
        ).group_by(Market.sport).all()
        
        for sport, count in sports:
            print(f"  {sport}: {count}")
            
        # Check maturity dates
        print("\nSample market dates:")
        samples = db.query(Market).filter(
            Market.is_finished == False
        ).order_by(Market.maturity_date).limit(5).all()
        
        for m in samples:
            print(f"  {m.home_team} vs {m.away_team}")
            print(f"    Sport: {m.sport}")
            print(f"    Date: {m.maturity_date}")
            print(f"    Is future: {m.maturity_date > now_utc}")

if __name__ == "__main__":
    check_markets()