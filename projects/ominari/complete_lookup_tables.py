#!/usr/bin/env python3
"""Complete lookup table population."""

import logging
from sqlalchemy import text
from database_v2 import db_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def complete_lookups():
    """Complete populating lookup tables."""
    
    with db_manager.get_db_session() as db:
        # Sources - should be quick
        logger.info("Populating sources...")
        sources = db.execute(text("""
            SELECT DISTINCT source 
            FROM odd 
            WHERE source IS NOT NULL
            LIMIT 255
        """)).fetchall()
        
        for i, (name,) in enumerate(sources):
            db.execute(text("""
                INSERT OR IGNORE INTO lu_sources (id, name) 
                VALUES (:id, :name)
            """), {"id": i, "name": name})
        
        logger.info(f"Inserted {len(sources)} sources")
        
        # Market types - also quick
        logger.info("Populating market types...")
        market_types = db.execute(text("""
            SELECT DISTINCT market_type 
            FROM odd 
            WHERE market_type IS NOT NULL
            LIMIT 255
        """)).fetchall()
        
        for i, (name,) in enumerate(market_types):
            db.execute(text("""
                INSERT OR IGNORE INTO lu_market_types (id, name) 
                VALUES (:id, :name)
            """), {"id": i, "name": name})
        
        logger.info(f"Inserted {len(market_types)} market types")
        
        db.commit()
        
        # Show final counts
        logger.info("\nFinal lookup table counts:")
        for table in ['lu_bookmakers', 'lu_sources', 'lu_market_types', 'lu_outcomes']:
            count = db.execute(text(f'SELECT COUNT(*) FROM {table}')).scalar()
            logger.info(f'  {table}: {count} records')


if __name__ == "__main__":
    complete_lookups()