#!/usr/bin/env python3
"""
Fix timezone issues by updating database directly
"""

import os
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_timezones():
    """Fix all timezone issues using direct SQL."""
    
    with db_manager.get_db_session() as db:
        # Convert all timestamps to UTC
        logger.info("Converting all timestamps to UTC...")
        
        # Fix market table
        db.execute(text("""
            UPDATE market 
            SET maturity_date = maturity_date AT TIME ZONE 'UTC',
                updated_at = updated_at AT TIME ZONE 'UTC'
            WHERE maturity_date IS NOT NULL
        """))
        
        # Fix odd table
        db.execute(text("""
            UPDATE odd
            SET updated_at = updated_at AT TIME ZONE 'UTC'
            WHERE updated_at IS NOT NULL
        """))
        
        db.commit()
        
        # Check results
        result = db.execute(text("""
            SELECT COUNT(*) FROM market
            WHERE maturity_date > NOW()
        """))
        future_count = result.scalar()
        
        logger.info(f"Markets with future dates: {future_count}")
        
        # Show sample
        result = db.execute(text("""
            SELECT home_team, away_team, sport, maturity_date
            FROM market
            ORDER BY maturity_date DESC
            LIMIT 5
        """))
        
        logger.info("\nSample markets:")
        for row in result:
            logger.info(f"  {row[0]} vs {row[1]} ({row[2]}) - {row[3]}")

if __name__ == "__main__":
    fix_timezones()