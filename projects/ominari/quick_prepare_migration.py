#!/usr/bin/env python3
"""Quick preparation for migration."""

import logging
from sqlalchemy import text
from database_v2 import db_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_prepare():
    """Quick preparation using limited queries."""
    
    with db_manager.get_db_session() as db:
        # Get unique values with LIMIT for speed
        logger.info("Quick-loading unique values...")
        
        # Sources - probably just a few
        sources = db.execute(text("""
            SELECT DISTINCT source FROM odd LIMIT 100
        """)).fetchall()
        
        seen_sources = set()
        for i, (name,) in enumerate(sources):
            if name and name not in seen_sources:
                db.execute(text("""
                    INSERT OR IGNORE INTO lu_sources (id, name) 
                    VALUES (:id, :name)
                """), {"id": len(seen_sources), "name": name})
                seen_sources.add(name)
        
        logger.info(f"Added {len(seen_sources)} sources")
        
        # Market types - also limited set
        market_types = db.execute(text("""
            SELECT DISTINCT market_type FROM odd LIMIT 100  
        """)).fetchall()
        
        seen_types = set()
        for i, (name,) in enumerate(market_types):
            if name and name not in seen_types:
                db.execute(text("""
                    INSERT OR IGNORE INTO lu_market_types (id, name) 
                    VALUES (:id, :name)
                """), {"id": len(seen_types), "name": name})
                seen_types.add(name)
        
        logger.info(f"Added {len(seen_types)} market types")
        
        db.commit()
        
        # Show what we have
        logger.info("\nLookup tables status:")
        for table in ['lu_bookmakers', 'lu_sources', 'lu_market_types', 'lu_outcomes']:
            count = db.execute(text(f'SELECT COUNT(*) FROM {table}')).scalar()
            logger.info(f"  {table}: {count} records")

if __name__ == "__main__":
    quick_prepare()
    logger.info("\nReady to run: python migrate_odds_optimized.py")