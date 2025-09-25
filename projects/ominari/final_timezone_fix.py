#!/usr/bin/env python3
"""
Final timezone fix - make ALL timestamps timezone-aware
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

def final_timezone_fix():
    """Apply final timezone fix using SQL."""
    
    with db_manager.get_db_session() as db:
        logger.info("Applying final timezone fix...")
        
        # Method 1: Explicit UTC timestamp conversion
        logger.info("Step 1: Converting market timestamps...")
        db.execute(text("""
            UPDATE market 
            SET maturity_date = TIMEZONE('UTC', maturity_date),
                updated_at = TIMEZONE('UTC', updated_at)
            WHERE maturity_date IS NOT NULL
        """))
        
        logger.info("Step 2: Converting odd timestamps...")
        db.execute(text("""
            UPDATE odd
            SET updated_at = TIMEZONE('UTC', updated_at)
            WHERE updated_at IS NOT NULL
        """))
        
        # Method 2: Force timezone using timestamptz type 
        logger.info("Step 3: Ensuring all dates are timestamptz...")
        
        try:
            # This adds timezone if not present
            db.execute(text("""
                UPDATE market 
                SET maturity_date = maturity_date AT TIME ZONE 'UTC'
                WHERE maturity_date IS NOT NULL 
                AND EXTRACT(timezone FROM maturity_date) IS NULL
            """))
        except Exception as e:
            logger.warning(f"Timezone conversion warning: {e}")
            
        db.commit()
        
        # Verify the fix
        logger.info("Verifying timezone fix...")
        
        # Check a sample market
        result = db.execute(text("""
            SELECT home_team, away_team, maturity_date, 
                   EXTRACT(timezone FROM maturity_date) as tz_offset
            FROM market 
            WHERE maturity_date IS NOT NULL
            ORDER BY updated_at DESC
            LIMIT 3
        """))
        
        logger.info("Sample market timestamps:")
        for row in result:
            logger.info(f"  {row[0]} vs {row[1]} - {row[2]} (tz: {row[3]})")
            
        # Test the comparison that was failing
        try:
            result = db.execute(text("""
                SELECT COUNT(*) FROM market
                WHERE maturity_date > NOW()
                AND is_finished = false
            """))
            future_count = result.scalar()
            logger.info(f"✅ Future markets query successful: {future_count} markets")
            
            # Test with Python timezone object
            now_utc = datetime.now(timezone.utc)
            markets = db.query(Market).filter(
                Market.maturity_date > now_utc,
                Market.is_finished == False
            ).limit(3).all()
            
            logger.info(f"✅ Python timezone comparison successful: {len(markets)} markets")
            
            for m in markets:
                logger.info(f"  {m.home_team} vs {m.away_team} - {m.maturity_date}")
                
        except Exception as e:
            logger.error(f"❌ Comparison still failing: {e}")
            
        logger.info("🎯 Final timezone fix complete!")

if __name__ == "__main__":
    final_timezone_fix()