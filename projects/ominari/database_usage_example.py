#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of using the enhanced database_v2 module.
"""

from database_v2 import db_manager, with_db_session
from models import Market, Odd
from datetime import datetime, timezone, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_query():
    """Example 1: Basic query with automatic retry."""
    logger.info("Example 1: Basic query with retry")
    
    # Old way:
    # db = SessionLocal()
    # try:
    #     markets = db.query(Market).filter(Market.sport == 'Soccer').limit(5).all()
    # finally:
    #     db.close()
    
    # New way - automatic retry and session management:
    with db_manager.get_db_session() as db:
        markets = db.query(Market).filter(Market.sport.like('%Soccer%')).limit(5).all()
        logger.info(f"Found {len(markets)} markets")
        for market in markets:
            logger.info(f"  - {market.home_team} vs {market.away_team}")


def example_2_with_retry_decorator():
    """Example 2: Using the decorator for automatic session handling."""
    logger.info("\nExample 2: Using decorator")
    
    @with_db_session
    def get_recent_odds(session, hours=24):
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return session.query(Odd).filter(
            Odd.updated_at >= cutoff
        ).count()
    
    # Function automatically gets a session
    count = get_recent_odds(hours=1)
    logger.info(f"Odds updated in last hour: {count}")


def example_3_execute_with_retry():
    """Example 3: Complex operation with retry."""
    logger.info("\nExample 3: Complex operation with retry")
    
    def update_market_stats(session, sport='Soccer'):
        # This will automatically retry if database is locked
        markets = session.query(Market).filter(
            Market.sport.like(f'%{sport}%')
        ).all()
        
        stats = {
            'total': len(markets),
            'active': sum(1 for m in markets if not m.is_finished),
            'finished': sum(1 for m in markets if m.is_finished)
        }
        
        logger.info(f"{sport} stats: {stats}")
        return stats
    
    # Execute with automatic retry
    stats = db_manager.execute_with_retry(update_market_stats, sport='Soccer')


def example_4_bulk_operations():
    """Example 4: Bulk operations with chunking."""
    logger.info("\nExample 4: Bulk operations")
    
    # Simulate creating many odds records
    new_odds = []
    for i in range(100):
        new_odds.append({
            'source_id': f'test_market_{i}',
            'position': 0,
            'market_type': 'moneyline',
            'outcome': f'Team {i}',
            'source': 'test',
            'bookmaker': 'test_book',
            'decimal_odds': 2.0,
            'normalized_implied': 0.5
        })
    
    # This will insert in chunks with retry logic
    # inserted = bulk_insert_with_retry(Odd, new_odds, chunk_size=50)
    # logger.info(f"Inserted {inserted} odds records")
    logger.info("Bulk insert example (commented out to avoid test data)")


def example_5_monitoring():
    """Example 5: Database monitoring."""
    logger.info("\nExample 5: Database monitoring")
    
    # Get database statistics
    stats = db_manager.get_database_stats()
    logger.info("Database statistics:")
    for key, value in stats.items():
        if 'bytes' in key:
            logger.info(f"  {key}: {value/1e6:.1f} MB")
        else:
            logger.info(f"  {key}: {value}")
    
    # Force optimization if needed
    if stats.get('size_bytes', 0) > 100_000_000_000:  # 100GB
        logger.warning("Database is large, consider running optimization")
        # db_manager.optimize_database()
    
    # Check WAL size
    if stats.get('wal_size_bytes', 0) > 100_000_000:  # 100MB
        logger.warning("WAL file is large, consider checkpointing")
        # db_manager.vacuum_wal()


def example_6_handling_locks():
    """Example 6: Handling database locks gracefully."""
    logger.info("\nExample 6: Handling locks")
    
    from sqlalchemy.exc import OperationalError
    
    try:
        with db_manager.get_db_session(retries=5, retry_delay=1.0) as db:
            # Simulate a long-running query
            count = db.query(Market).count()
            logger.info(f"Total markets: {count}")
    except OperationalError as e:
        if "database is locked" in str(e):
            logger.error("Database is locked after all retries!")
            logger.error("Consider running: python database_monitor.py unlock --force")
        else:
            raise


def main():
    """Run all examples."""
    try:
        example_1_basic_query()
        example_2_with_retry_decorator()
        example_3_execute_with_retry()
        example_4_bulk_operations()
        example_5_monitoring()
        example_6_handling_locks()
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()