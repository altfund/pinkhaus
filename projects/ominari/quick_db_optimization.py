#!/usr/bin/env python3
"""
Quick Database Optimization

Fast, targeted optimization for immediate performance gains on the 216GB database.
Focuses on the most critical indexes and settings.
"""

import sqlite3
import logging
import time
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_db_size(db_path: str) -> dict:
    """Get database file size quickly."""
    main_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    wal_path = db_path + "-wal"
    wal_size = os.path.getsize(wal_path) if os.path.exists(wal_path) else 0
    
    return {
        'main_gb': main_size / (1024**3),
        'wal_mb': wal_size / (1024**2),
        'total_gb': (main_size + wal_size) / (1024**3)
    }


def quick_optimize_database(db_path: str = "sport_odds.db"):
    """Run quick database optimization."""
    logger.info("🚀 Quick Database Optimization")
    logger.info("=" * 50)
    
    # Check database size
    size_info = get_db_size(db_path)
    logger.info(f"Database: {size_info['main_gb']:.1f} GB (WAL: {size_info['wal_mb']:.1f} MB)")
    
    conn = sqlite3.connect(db_path, timeout=30.0)
    
    try:
        # 1. Critical settings optimization
        logger.info("1. Optimizing critical settings...")
        
        settings = [
            ("cache_size", "PRAGMA cache_size = -2000000"),  # 2GB cache
            ("temp_store", "PRAGMA temp_store = MEMORY"),
            ("synchronous", "PRAGMA synchronous = NORMAL"),
            ("journal_mode", "PRAGMA journal_mode = WAL"),
            ("wal_autocheckpoint", "PRAGMA wal_autocheckpoint = 1000"),
        ]
        
        for name, sql in settings:
            try:
                conn.execute(sql)
                logger.info(f"   ✅ Set {name}")
            except Exception as e:
                logger.error(f"   ❌ Failed {name}: {e}")
        
        # 2. Most critical indexes only
        logger.info("2. Creating critical indexes...")
        
        critical_indexes = [
            ("idx_market_source_sport", "CREATE INDEX IF NOT EXISTS idx_market_source_sport ON market(source, sport)"),
            ("idx_market_finished", "CREATE INDEX IF NOT EXISTS idx_market_finished ON market(is_finished, maturity_date)"),
            ("idx_odd_source_updated", "CREATE INDEX IF NOT EXISTS idx_odd_source_updated ON odd(source_id, updated_at)"),
            ("idx_odd_bookmaker", "CREATE INDEX IF NOT EXISTS idx_odd_bookmaker ON odd(bookmaker)"),
        ]
        
        created_count = 0
        for name, sql in critical_indexes:
            try:
                start = time.time()
                conn.execute(sql)
                duration = time.time() - start
                logger.info(f"   ✅ {name} ({duration:.2f}s)")
                created_count += 1
            except sqlite3.Error as e:
                if "already exists" in str(e):
                    logger.info(f"   ⏭️ {name} (already exists)")
                else:
                    logger.error(f"   ❌ {name}: {e}")
        
        # 3. Quick analyze
        logger.info("3. Updating statistics...")
        try:
            start = time.time()
            conn.execute("ANALYZE")
            duration = time.time() - start
            logger.info(f"   ✅ ANALYZE completed ({duration:.2f}s)")
        except Exception as e:
            logger.error(f"   ❌ ANALYZE failed: {e}")
        
        conn.commit()
        
        # 4. Quick performance test
        logger.info("4. Testing performance...")
        
        test_queries = [
            ("market_count", "SELECT COUNT(*) FROM market"),
            ("blockchain_markets", "SELECT COUNT(*) FROM market WHERE source LIKE 'blockchain_%'"),
            ("recent_odds", "SELECT COUNT(*) FROM odd WHERE updated_at > datetime('now', '-1 day')"),
        ]
        
        for name, sql in test_queries:
            try:
                start = time.time()
                result = conn.execute(sql).fetchone()
                duration = time.time() - start
                count = result[0] if result else 0
                logger.info(f"   ✅ {name}: {count:,} rows ({duration:.3f}s)")
            except Exception as e:
                logger.error(f"   ❌ {name}: {e}")
        
        logger.info("\n✅ Quick optimization complete!")
        logger.info(f"   Critical indexes created: {created_count}")
        logger.info(f"   Database ready for improved performance")
        
        return True
        
    finally:
        conn.close()


def check_existing_indexes(db_path: str = "sport_odds.db"):
    """Check what indexes already exist."""
    logger.info("🔍 Checking existing indexes...")
    
    conn = sqlite3.connect(db_path)
    
    try:
        indexes = conn.execute("""
            SELECT name, tbl_name, sql 
            FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
            ORDER BY tbl_name, name
        """).fetchall()
        
        if indexes:
            logger.info(f"Found {len(indexes)} custom indexes:")
            for name, table, sql in indexes:
                logger.info(f"   {table}.{name}")
        else:
            logger.info("No custom indexes found")
            
        return indexes
        
    finally:
        conn.close()


def main():
    """Run quick optimization."""
    start_time = time.time()
    
    # Check existing indexes first
    existing_indexes = check_existing_indexes()
    
    # Run quick optimization
    success = quick_optimize_database()
    
    end_time = time.time()
    
    if success:
        logger.info(f"\n🎯 Optimization completed in {end_time - start_time:.2f} seconds")
        logger.info("Ready for blockchain trading system!")
    else:
        logger.error("Optimization failed")


if __name__ == "__main__":
    main()