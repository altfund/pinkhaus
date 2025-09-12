#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug database issues safely using ORM.
"""

from database import SessionLocal
from models import Market
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_maturity_dates():
    """Check maturity date values safely using ORM."""
    db = SessionLocal()
    try:
        # Get a small sample first
        recent_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        # Test query with just one record
        logger.info("Testing query with single record...")
        try:
            single_market = db.query(Market).filter(
                Market.sport.like('%Soccer%')
            ).first()
            if single_market:
                logger.info(f"Single market test: ID={single_market.source_id}, maturity={single_market.maturity_date}")
            else:
                logger.info("No soccer markets found")
        except Exception as e:
            logger.error(f"Error with single record: {e}")
            
        # Check for problematic values
        logger.info("\nChecking for empty string issues...")
        try:
            # Use raw SQL safely with limit
            result = db.execute(text("""
                SELECT source_id, maturity_date 
                FROM market 
                WHERE maturity_date = '' 
                LIMIT 5
            """)).fetchall()
            logger.info(f"Found {len(result)} records with empty string maturity_date")
            for row in result:
                logger.info(f"  {row[0]}: '{row[1]}'")
        except Exception as e:
            logger.error(f"Error checking empty strings: {e}")
            
        # Check recent markets without maturity_date filter
        logger.info("\nChecking recent markets without date filter...")
        try:
            recent_markets = db.query(Market).filter(
                Market.sport.like('%Soccer%')
            ).order_by(Market.updated_at.desc()).limit(5).all()
            
            logger.info(f"Found {len(recent_markets)} recent soccer markets")
            for market in recent_markets:
                logger.info(f"  {market.source_id}: updated={market.updated_at}, maturity={market.maturity_date}")
        except Exception as e:
            logger.error(f"Error fetching recent markets: {e}")
            
    finally:
        db.close()


def test_signal_generation():
    """Test the signal generation query that's failing."""
    db = SessionLocal()
    try:
        recent_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        # Try the exact query that's failing but with error handling
        logger.info(f"\nTesting signal generation query (markets updated since {recent_time})...")
        
        # First count how many we expect
        count = db.query(Market).filter(
            Market.sport.like('%Soccer%')
        ).count()
        logger.info(f"Total soccer markets: {count}")
        
        # Try without the updated_at filter first
        try:
            markets_no_filter = db.query(Market).filter(
                Market.sport.like('%Soccer%')
            ).limit(5).all()
            logger.info(f"Without time filter: {len(markets_no_filter)} markets loaded successfully")
        except Exception as e:
            logger.error(f"Error even without time filter: {e}")
            return
            
        # Now try with the filter
        try:
            markets = db.query(Market).filter(
                Market.updated_at >= recent_time,
                Market.sport.like('%Soccer%')
            ).limit(5).all()
            logger.info(f"With time filter: {len(markets)} markets loaded successfully")
            
            # Try to access maturity_date
            for market in markets:
                try:
                    mat_date = market.maturity_date
                    logger.info(f"  Market {market.source_id}: maturity_date accessible = {mat_date}")
                except Exception as e:
                    logger.error(f"  Market {market.source_id}: Error accessing maturity_date: {e}")
                    
        except Exception as e:
            logger.error(f"Error with time filter: {e}")
            
    finally:
        db.close()


def fix_empty_strings():
    """Fix empty string maturity_dates directly in database."""
    db = SessionLocal()
    try:
        logger.info("\nFixing empty string maturity_dates...")
        
        # First find them
        result = db.execute(text("""
            SELECT COUNT(*) FROM market WHERE maturity_date = ''
        """)).fetchone()
        count = result[0] if result else 0
        logger.info(f"Found {count} records with empty string maturity_date")
        
        if count > 0:
            # Fix them
            db.execute(text("""
                UPDATE market 
                SET maturity_date = NULL 
                WHERE maturity_date = ''
            """))
            db.commit()
            logger.info(f"Updated {count} records to NULL")
            
            # Verify
            result = db.execute(text("""
                SELECT COUNT(*) FROM market WHERE maturity_date = ''
            """)).fetchone()
            remaining = result[0] if result else 0
            logger.info(f"Remaining empty strings: {remaining}")
    except Exception as e:
        logger.error(f"Error fixing empty strings: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    fix_empty_strings()
    check_maturity_dates()
    test_signal_generation()