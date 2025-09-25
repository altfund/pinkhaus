#!/usr/bin/env python3
"""
Check what the web monitor is actually querying
"""
import os
os.environ['PG_PORT'] = '5999'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from database_v2 import db_manager
from models import Market, Odd
from datetime import datetime, timezone, timedelta
from sqlalchemy import desc

def check_web_monitor_query():
    """Replicate the web monitor's query"""
    with db_manager.get_db_session() as db:
        now = datetime.now(timezone.utc)
        future_time = now + timedelta(hours=48)
        
        # This is the exact query from web_monitor_unified.py
        active_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False,
            Market.maturity_date > now,
            Market.maturity_date <= future_time
        ).order_by(Market.maturity_date).limit(50).all()
        
        print(f"Current time: {now}")
        print(f"Query time range: {now} to {future_time}")
        print(f"Found {len(active_markets)} markets")
        print()
        
        # Show first 10 markets
        for i, market in enumerate(active_markets[:10]):
            if market.maturity_date:
                # Ensure timezone aware
                maturity = market.maturity_date
                if maturity.tzinfo is None:
                    maturity = maturity.replace(tzinfo=timezone.utc)
                time_diff = maturity - now
                hours = time_diff.total_seconds() / 3600
            else:
                hours = 0
            
            print(f"{i+1}. {market.home_team} vs {market.away_team}")
            print(f"   Maturity: {market.maturity_date}")
            print(f"   Hours from now: {hours:.1f}")
            print(f"   Source: {market.source}")
            print(f"   Has odds: ", end="")
            
            # Check for odds
            odds_count = db.query(Odd).filter(
                Odd.source_id == market.source_id
            ).count()
            print(f"{odds_count} odds")
            print()

if __name__ == "__main__":
    check_web_monitor_query()