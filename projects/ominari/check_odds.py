#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check if we have odds data in the database.
"""

from database import SessionLocal
from models import Market, Odd
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_odds():
    """Check odds data."""
    db = SessionLocal()
    try:
        # Count total odds
        odds_count = db.query(Odd).count()
        logger.info(f"Total odds in database: {odds_count}")
        
        if odds_count == 0:
            logger.warning("No odds found in database!")
            
            # Check if we need to populate odds
            markets_count = db.query(Market).count()
            logger.info(f"Total markets: {markets_count}")
            
            # Get sample markets
            sample_markets = db.query(Market).filter(
                Market.sport.like('%Soccer%')
            ).limit(5).all()
            
            logger.info("\nSample soccer markets:")
            for market in sample_markets:
                logger.info(f"  {market.source_id}: {market.home_team} vs {market.away_team}")
        else:
            # Get sample odds
            sample_odds = db.query(Odd).limit(5).all()
            logger.info("\nSample odds:")
            for odd in sample_odds:
                logger.info(f"  Market: {odd.source_id}, Position: {odd.position}, Decimal: {odd.decimal_odds}")
                
            # Check recent odds
            recent_count = db.execute(text("""
                SELECT COUNT(*) FROM odd 
                WHERE updated_at > datetime('now', '-1 hour')
            """)).scalar()
            logger.info(f"\nOdds updated in last hour: {recent_count}")
            
    except Exception as e:
        logger.error(f"Error checking odds: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    check_odds()