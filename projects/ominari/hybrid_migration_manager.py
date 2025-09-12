#!/usr/bin/env python3
"""
Hybrid Migration Manager

Manages data migration for hybrid SQLite/PostgreSQL architecture.
Only migrates recent data to PostgreSQL while keeping historical in SQLite.
"""

import sqlite3
import logging
from datetime import datetime, timedelta
import os
import signal
import sys
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hybrid_migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HybridMigrationManager:
    """Manages hybrid database migration strategy."""
    
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.running = True
        self.conn = None
        self.setup_signal_handlers()
        
        # Migration cutoff - only migrate data newer than this
        self.cutoff_date = datetime.now() - timedelta(days=180)  # 6 months
        logger.info(f"Migration cutoff date: {self.cutoff_date}")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown."""
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
    
    def shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("Received shutdown signal, saving state...")
        self.running = False
    
    def connect(self):
        """Connect to SQLite with optimizations."""
        self.conn = sqlite3.connect('sport_odds.db', timeout=300.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=20000")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA mmap_size=2147483648")  # 2GB mmap
    
    def analyze_migration_scope(self):
        """Analyze what needs to be migrated."""
        cursor = self.conn.cursor()
        
        logger.info("Analyzing migration scope...")
        
        # Count recent markets
        cursor.execute("""
            SELECT COUNT(*) FROM market 
            WHERE datetime(substr(maturity_date, 1, 19)) >= ?
        """, (self.cutoff_date.isoformat(),))
        recent_markets = cursor.fetchone()[0]
        
        # Count recent odds (estimate with small sample)
        cursor.execute("""
            SELECT COUNT(*) FROM odd 
            WHERE source_id IN (
                SELECT source_id FROM market 
                WHERE datetime(substr(maturity_date, 1, 19)) >= ?
                LIMIT 100
            )
        """, (self.cutoff_date.isoformat(),))
        
        logger.info(f"Recent markets to migrate: {recent_markets:,}")
        
        # Get total database stats
        cursor.execute("SELECT COUNT(*) FROM market")
        total_markets = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM odd WHERE rowid < 1000000")  # Sample
        sample_odds = cursor.fetchone()[0]
        
        logger.info(f"Total markets: {total_markets:,}")
        logger.info(f"Sample odds (first 1M): {sample_odds:,}")
        
        return {
            'recent_markets': recent_markets,
            'total_markets': total_markets,
            'migration_percentage': (recent_markets / total_markets * 100) if total_markets > 0 else 0
        }
    
    def create_archive_tables(self):
        """Create archive tables for old data."""
        cursor = self.conn.cursor()
        
        logger.info("Creating archive tables...")
        
        # Archive old odds
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odd_archive AS
            SELECT * FROM odd WHERE 1=0
        """)
        
        # Archive old markets
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_archive AS
            SELECT * FROM market WHERE 1=0
        """)
        
        # Create indexes on archive tables
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_odd_archive_market 
            ON odd_archive(market_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_archive_date 
            ON market_archive(starts_at)
        """)
        
        self.conn.commit()
        logger.info("Archive tables created")
    
    def archive_old_data(self):
        """Move old data to archive tables."""
        cursor = self.conn.cursor()
        
        logger.info("Archiving old data...")
        
        # Archive old markets in batches
        archived = 0
        while self.running:
            cursor.execute("""
                INSERT OR IGNORE INTO market_archive
                SELECT * FROM market 
                WHERE datetime(substr(maturity_date, 1, 19)) < ?
                  AND source_id NOT IN (SELECT source_id FROM market_archive)
                LIMIT ?
            """, (self.cutoff_date.isoformat(), self.batch_size))
            
            if cursor.rowcount == 0:
                break
            
            archived += cursor.rowcount
            self.conn.commit()
            
            if archived % 10000 == 0:
                logger.info(f"Archived {archived:,} old markets...")
        
        logger.info(f"Total markets archived: {archived:,}")
    
    def optimize_for_recent_data(self):
        """Optimize SQLite for recent data access."""
        cursor = self.conn.cursor()
        
        logger.info("Creating optimized indexes for recent data...")
        
        # Create partial indexes for recent data
        indexes = [
            """CREATE INDEX IF NOT EXISTS idx_market_recent 
               ON market(maturity_date) 
               WHERE datetime(substr(maturity_date, 1, 19)) >= date('now', '-180 days')""",
            
            """CREATE INDEX IF NOT EXISTS idx_market_sport_recent 
               ON market(sport, maturity_date) 
               WHERE datetime(substr(maturity_date, 1, 19)) >= date('now', '-180 days')""",
        ]
        
        for idx in indexes:
            try:
                cursor.execute(idx)
                logger.info(f"Created index: {idx.split('ON')[0].strip()}")
            except sqlite3.Error as e:
                logger.warning(f"Index creation warning: {e}")
        
        # Analyze tables for query planner
        cursor.execute("ANALYZE market")
        cursor.execute("ANALYZE sqlite_master")
        
        self.conn.commit()
        logger.info("Optimization complete")
    
    def create_summary_tables(self):
        """Create summary tables for fast queries."""
        cursor = self.conn.cursor()
        
        logger.info("Creating summary tables...")
        
        # Daily market summary
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_daily_summary AS
            SELECT 
                date(maturity_date) as date,
                sport,
                COUNT(*) as market_count,
                COUNT(DISTINCT home_team || '_' || away_team) as unique_matches
            FROM market
            WHERE datetime(substr(starts_at, 1, 19)) >= date('now', '-365 days')
            GROUP BY date(maturity_date), sport
        """)
        
        # Sport statistics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sport_stats AS
            SELECT 
                sport,
                COUNT(*) as total_markets,
                COUNT(CASE WHEN is_finished = 1 THEN 1 END) as finished_markets,
                MIN(maturity_date) as earliest_market,
                MAX(maturity_date) as latest_market
            FROM market
            GROUP BY sport
        """)
        
        self.conn.commit()
        logger.info("Summary tables created")
    
    def run(self):
        """Run the hybrid migration process."""
        try:
            self.connect()
            
            # Analyze scope
            scope = self.analyze_migration_scope()
            logger.info(f"Migration will cover {scope['migration_percentage']:.1f}% of data")
            
            # Create required structures
            self.create_archive_tables()
            self.create_summary_tables()
            
            # Optimize for recent data
            self.optimize_for_recent_data()
            
            # Archive old data (optional, can be run separately)
            if os.getenv('ARCHIVE_OLD_DATA', 'false').lower() == 'true':
                self.archive_old_data()
            
            logger.info("✅ Hybrid migration preparation complete")
            logger.info("📝 Next steps:")
            logger.info("   1. Run setup_postgresql_hybrid.py to create PostgreSQL schema")
            logger.info("   2. Start blockchain readers to populate PostgreSQL")
            logger.info("   3. Update unified data system to query both databases")
            
        except Exception as e:
            logger.error(f"Migration error: {e}")
            raise
        finally:
            if self.conn:
                self.conn.close()


if __name__ == "__main__":
    manager = HybridMigrationManager()
    manager.run()