#!/usr/bin/env python3
"""Prepare for migration by populating all lookup tables."""

import logging
from sqlalchemy import text
from database_v2 import db_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def populate_all_lookups():
    """Populate all lookup tables from existing data."""
    
    with db_manager.get_db_session() as db:
        # Get all unique values using GROUP BY for efficiency
        logger.info("Analyzing unique values in odd table...")
        
        # Sources
        logger.info("Loading unique sources...")
        sources = db.execute(text("""
            SELECT DISTINCT source 
            FROM odd 
            WHERE source IS NOT NULL
            GROUP BY source
            ORDER BY source
        """)).fetchall()
        
        logger.info(f"Found {len(sources)} unique sources")
        for i, (name,) in enumerate(sources):
            if i < 255:  # TINYINT limit
                db.execute(text("""
                    INSERT OR IGNORE INTO lu_sources (id, name) 
                    VALUES (:id, :name)
                """), {"id": i, "name": name})
        
        # Market types
        logger.info("Loading unique market types...")
        market_types = db.execute(text("""
            SELECT DISTINCT market_type 
            FROM odd 
            WHERE market_type IS NOT NULL
            GROUP BY market_type
            ORDER BY market_type
        """)).fetchall()
        
        logger.info(f"Found {len(market_types)} unique market types")
        for i, (name,) in enumerate(market_types):
            if i < 255:
                db.execute(text("""
                    INSERT OR IGNORE INTO lu_market_types (id, name) 
                    VALUES (:id, :name)
                """), {"id": i, "name": name})
        
        # Bookmakers already populated
        bookmaker_count = db.execute(text(
            "SELECT COUNT(*) FROM lu_bookmakers"
        )).scalar()
        logger.info(f"Bookmakers already populated: {bookmaker_count}")
        
        db.commit()
        
        # Show final summary
        logger.info("\nLookup tables ready:")
        for table in ['lu_bookmakers', 'lu_sources', 'lu_market_types', 'lu_outcomes']:
            count = db.execute(text(f'SELECT COUNT(*) FROM {table}')).scalar()
            
            # Show sample values
            samples = db.execute(text(f"""
                SELECT id, name FROM {table} 
                ORDER BY id LIMIT 5
            """)).fetchall()
            
            logger.info(f"\n{table}: {count} records")
            for id, name in samples:
                logger.info(f"  {id}: {name}")
            if count > 5:
                logger.info(f"  ... and {count - 5} more")

def verify_ready():
    """Verify system is ready for migration."""
    logger.info("\nVerifying migration readiness...")
    
    with db_manager.get_db_session() as db:
        # Check tables exist
        tables = ['odds_normalized', 'lu_bookmakers', 'lu_sources', 
                  'lu_market_types', 'lu_outcomes']
        
        for table in tables:
            result = db.execute(text(f"""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='{table}'
            """)).fetchone()
            
            if result:
                logger.info(f"✓ {table} exists")
            else:
                logger.error(f"✗ {table} missing!")
                return False
        
        # Check indexes
        indexes = db.execute(text("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND tbl_name='odds_normalized'
        """)).fetchall()
        
        logger.info(f"✓ {len(indexes)} indexes on odds_normalized")
        
        # Check if any data already migrated
        migrated = db.execute(text(
            "SELECT COUNT(*) FROM odds_normalized"
        )).scalar()
        
        if migrated > 0:
            logger.info(f"ℹ {migrated:,} records already migrated (will resume)")
        else:
            logger.info("ℹ Starting fresh migration")
        
        # Check available disk space
        import shutil
        stat = shutil.disk_usage('.')
        free_gb = stat.free / (1024**3)
        logger.info(f"✓ Free disk space: {free_gb:.1f} GB")
        
        if free_gb < 50:
            logger.warning("⚠ Low disk space! Migration needs temp space.")
        
        return True

def optimize_database():
    """Run ANALYZE to update query planner statistics."""
    logger.info("\nOptimizing database...")
    
    with db_manager.get_db_session() as db:
        db.execute(text("ANALYZE"))
        logger.info("✓ Database statistics updated")

if __name__ == "__main__":
    logger.info("Preparing for migration...")
    
    # Step 1: Populate lookups
    populate_all_lookups()
    
    # Step 2: Verify readiness
    if verify_ready():
        # Step 3: Optimize
        optimize_database()
        
        logger.info("\n" + "="*80)
        logger.info("READY FOR MIGRATION!")
        logger.info("Run: python migrate_odds_optimized.py")
        logger.info("Monitor with: python monitor_migration.py")
    else:
        logger.error("System not ready for migration!")