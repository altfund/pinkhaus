#!/usr/bin/env python3
"""
Fix timezone issues in the database
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_timezones():
    """Ensure all market dates are timezone-aware."""
    with db_manager.get_db_session() as db:
        markets = db.query(Market).all()
        fixed = 0
        
        for market in markets:
            # Check if maturity_date needs timezone
            if market.maturity_date and market.maturity_date.tzinfo is None:
                # Add UTC timezone
                market.maturity_date = market.maturity_date.replace(tzinfo=timezone.utc)
                fixed += 1
                
            # Check updated_at
            if market.updated_at and market.updated_at.tzinfo is None:
                market.updated_at = market.updated_at.replace(tzinfo=timezone.utc)
                
        db.commit()
        logger.info(f"Fixed timezone for {fixed} markets")
        
        # Verify
        test_market = db.query(Market).first()
        if test_market:
            logger.info(f"Sample market date: {test_market.maturity_date}")
            logger.info(f"Has timezone: {test_market.maturity_date.tzinfo is not None}")

if __name__ == "__main__":
    fix_timezones()