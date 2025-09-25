#!/usr/bin/env python3
"""
Fix timezone issues in blockchain market data
"""

import os
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market, Odd
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_timezones():
    """Ensure all datetime fields are timezone-aware."""
    
    with db_manager.get_db_session() as db:
        # Get all markets
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
                
            if changed:
                fixed_count += 1
                
        # Fix odds too
        odds = db.query(Odd).all()
        for odd in odds:
            if odd.updated_at and odd.updated_at.tzinfo is None:
                odd.updated_at = odd.updated_at.replace(tzinfo=timezone.utc)
                
        db.commit()
        logger.info(f"Fixed {fixed_count} markets with timezone issues")
        
        # Verify
        naive_markets = db.query(Market).filter(
            Market.maturity_date.op('AT TIME ZONE')('UTC') == None
        ).count()
        
        logger.info(f"Markets without timezone: {naive_markets}")

if __name__ == "__main__":
    logger.info("Fixing timezone issues...")
    fix_timezones()
    logger.info("Done!")