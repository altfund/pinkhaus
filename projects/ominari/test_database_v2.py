#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the database_v2 module functionality.
"""

from database_v2 import db_manager, with_db_session
from models import Market, Odd
from datetime import datetime, timezone, timedelta
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_query():
    """Test basic query with retry."""
    logger.info("Test 1: Basic query with retry")
    start = time.time()
    
    try:
        with db_manager.get_db_session() as db:
            # Count soccer markets
            count = db.query(Market).filter(
                Market.sport.like('%Soccer%')
            ).count()
            logger.info(f"Found {count} soccer markets")
            
            # Get a sample
            markets = db.query(Market).filter(
                Market.sport.like('%Soccer%')
            ).limit(3).all()
            
            for market in markets:
                logger.info(f"  - {market.home_team} vs {market.away_team}")
                
    except Exception as e:
        logger.error(f"Query failed: {e}")
        
    elapsed = time.time() - start
    logger.info(f"Query completed in {elapsed:.2f} seconds\n")


def test_with_retry():
    """Test the retry mechanism."""
    logger.info("Test 2: Testing retry mechanism")
    
    def query_with_potential_lock(session):
        # This might fail if database is locked
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        count = session.query(Odd).filter(
            Odd.updated_at >= recent
        ).count()
        return count
    
    try:
        start = time.time()
        count = db_manager.execute_with_retry(query_with_potential_lock)
        elapsed = time.time() - start
        logger.info(f"Found {count} recent odds in {elapsed:.2f} seconds")
    except Exception as e:
        logger.error(f"Query failed after retries: {e}")


def test_database_stats():
    """Test database statistics."""
    logger.info("\nTest 3: Database statistics")
    
    stats = db_manager.get_database_stats()
    for key, value in stats.items():
        if 'bytes' in key:
            logger.info(f"  {key}: {value/1e9:.2f} GB")
        elif key == 'cache_hit_rate':
            logger.info(f"  {key}: {value*100:.1f}%")
        else:
            logger.info(f"  {key}: {value}")


def test_performance():
    """Test query performance."""
    logger.info("\nTest 4: Performance test")
    
    @with_db_session
    def count_markets_by_sport(session):
        from sqlalchemy import func
        
        results = session.query(
            Market.sport,
            func.count(Market.source_id).label('count')
        ).group_by(Market.sport).all()
        
        return {sport: count for sport, count in results}
    
    start = time.time()
    sport_counts = count_markets_by_sport()
    elapsed = time.time() - start
    
    logger.info(f"Query completed in {elapsed:.2f} seconds")
    for sport, count in sorted(sport_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"  {sport}: {count:,}")


def main():
    """Run all tests."""
    logger.info("Starting database_v2 tests...\n")
    
    test_basic_query()
    test_with_retry()
    test_database_stats()
    test_performance()
    
    logger.info("\nAll tests completed!")


if __name__ == "__main__":
    main()