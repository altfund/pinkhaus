#!/usr/bin/env python3
"""
Properly fix all timezone issues in the database
"""

import os
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market, Odd
from datetime import datetime, timezone
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_all_timezones():
    """Fix all timezone issues properly."""
    
    with db_manager.get_db_session() as db:
        # Fix markets using ORM to ensure proper timezone handling
        logger.info("Fixing market timezones...")
        
        markets = db.query(Market).all()
        fixed_count = 0
        
        for market in markets:
            changed = False
            
            # Fix maturity_date
            if market.maturity_date and market.maturity_date.tzinfo is None:
                market.maturity_date = market.maturity_date.replace(tzinfo=timezone.utc)
                changed = True
                
            # Fix updated_at
            if market.updated_at and market.updated_at.tzinfo is None:
                market.updated_at = market.updated_at.replace(tzinfo=timezone.utc)
                changed = True
                
            # Fix start_time if exists
            if hasattr(market, 'start_time') and market.start_time and market.start_time.tzinfo is None:
                market.start_time = market.start_time.replace(tzinfo=timezone.utc)
                changed = True
                
            # Fix last_update if exists
            if hasattr(market, 'last_update') and market.last_update and market.last_update.tzinfo is None:
                market.last_update = market.last_update.replace(tzinfo=timezone.utc)
                changed = True
                
            if changed:
                fixed_count += 1
        
        logger.info(f"Fixed {fixed_count} markets")
        
        # Fix odds
        odds = db.query(Odd).all()
        fixed_odds = 0
        
        for odd in odds:
            if odd.updated_at and odd.updated_at.tzinfo is None:
                odd.updated_at = odd.updated_at.replace(tzinfo=timezone.utc)
                fixed_odds += 1
                
        logger.info(f"Fixed {fixed_odds} odds")
        
        db.commit()
        
        # Verify
        result = db.execute(text("""
            SELECT COUNT(*) FROM market
            WHERE maturity_date IS NOT NULL
            AND maturity_date > NOW()
        """))
        future_count = result.scalar()
        
        logger.info(f"Markets with future dates: {future_count}")
        
        # Show sample
        result = db.execute(text("""
            SELECT home_team, away_team, sport, maturity_date
            FROM market
            WHERE is_finished = false
            ORDER BY maturity_date
            LIMIT 5
        """))
        
        logger.info("\nSample markets:")
        for row in result:
            logger.info(f"  {row[0]} vs {row[1]} ({row[2]}) - {row[3]}")

if __name__ == "__main__":
    fix_all_timezones()