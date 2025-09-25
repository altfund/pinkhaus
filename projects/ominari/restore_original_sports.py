#!/usr/bin/env python3
"""Restore original sport classifications from SQLite database"""

import sqlite3
import logging
from database_v2 import db_manager
from models import Market
from sqlalchemy import func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Restoring original sport classifications from SQLite...")
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect('sport_odds.db')
    sqlite_cursor = sqlite_conn.cursor()
    
    # Get all markets with their sports from SQLite
    sqlite_cursor.execute("""
        SELECT source_id, sport 
        FROM market 
        WHERE source_id IS NOT NULL
    """)
    
    sqlite_sports = {}
    for source_id, sport in sqlite_cursor.fetchall():
        sqlite_sports[source_id] = sport if sport else 'Unknown'
    
    sqlite_conn.close()
    logger.info(f"Found {len(sqlite_sports)} markets in SQLite")
    
    # Update PostgreSQL
    with db_manager.get_db_session() as db:
        # Process in batches
        batch_size = 1000
        updated = 0
        
        # Get all markets from PostgreSQL
        total_markets = db.query(func.count(Market.source_id)).scalar()
        logger.info(f"Total markets in PostgreSQL: {total_markets}")
        
        for offset in range(0, total_markets, batch_size):
            markets = db.query(Market).offset(offset).limit(batch_size).all()
            
            for market in markets:
                if market.source_id in sqlite_sports:
                    original_sport = sqlite_sports[market.source_id]
                    # Use the original sport value, including Unknown
                    if original_sport != market.sport:
                        logger.debug(f"Restoring {market.source_id}: {market.sport} -> {original_sport}")
                        market.sport = original_sport
                        updated += 1
            
            db.commit()
            logger.info(f"Processed {min(offset + batch_size, total_markets)}/{total_markets} markets, updated {updated} so far")
    
    logger.info(f"\n✅ Restoration complete! Updated {updated} markets")
    
    # Show final distribution
    with db_manager.get_db_session() as db:
        logger.info("\nFinal sport distribution:")
        sport_counts = db.query(
            Market.sport,
            func.count().label('count')
        ).group_by(Market.sport).order_by(func.count().desc()).all()
        
        for sport, count in sport_counts:
            logger.info(f"  {sport:20} {count:6d}")

if __name__ == "__main__":
    main()