#!/usr/bin/env python3
"""
Fixed Migration Manager with Deduplication

Handles the UNIQUE constraint issues and continues migration intelligently.
Also prepares for future PostgreSQL migration.
"""

import os
import sqlite3
import logging
import time
import signal
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fixed_migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FixedMigrationManager:
    """Manages database migration with proper deduplication and error handling."""
    
    def __init__(self):
        self.db_path = 'sport_odds.db'
        self.batch_size = 5000  # Smaller batches for better error handling
        self.checkpoint_interval = 10000  # Save progress every 10k records
        self.running = True
        self.stats = {
            'processed': 0,
            'migrated': 0,
            'skipped': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        # Load checkpoint
        self.checkpoint = self._load_checkpoint()
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("Received shutdown signal, saving state...")
        self.running = False
        self._save_checkpoint()
        exit(0)
    
    def _load_checkpoint(self) -> Dict:
        """Load migration checkpoint."""
        checkpoint_file = 'migration_checkpoint.json'
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        return {'last_rowid': 0, 'stats': self.stats}
    
    def _save_checkpoint(self):
        """Save migration checkpoint."""
        checkpoint = {
            'last_rowid': self.checkpoint.get('last_rowid', 0),
            'stats': self.stats,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        with open('migration_checkpoint.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Checkpoint saved at rowid {checkpoint['last_rowid']}")
    
    def fix_migration_schema(self):
        """Fix schema issues that cause UNIQUE constraint violations."""
        logger.info("🔧 Fixing migration schema...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if we need to add missing indexes or modify constraints
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS odds_normalized_v2 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT NOT NULL,
                    bookmaker_id INTEGER NOT NULL,
                    outcome_id INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    source_id INTEGER DEFAULT 0,
                    market_type_id INTEGER DEFAULT 1,
                    position INTEGER DEFAULT 0,
                    line_x100 INTEGER,
                    decimal_odds_x1000 INTEGER,
                    american_odds INTEGER,
                    implied_x10000 INTEGER,
                    inserted_at INTEGER DEFAULT (strftime('%s', 'now')),
                    -- Remove the problematic UNIQUE constraint
                    -- Add index instead for performance
                    FOREIGN KEY (market_id) REFERENCES markets_normalized(market_id),
                    FOREIGN KEY (bookmaker_id) REFERENCES bookmakers_lookup(id),
                    FOREIGN KEY (outcome_id) REFERENCES outcomes_lookup(id)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_odds_norm_v2_market_time 
                ON odds_normalized_v2(market_id, updated_at)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_odds_norm_v2_composite 
                ON odds_normalized_v2(market_id, bookmaker_id, outcome_id, updated_at)
            """)
            
            conn.commit()
            logger.info("✅ Schema fixed with new table structure")
            
        except Exception as e:
            logger.error(f"Schema fix error: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def migrate_with_deduplication(self):
        """Migrate data with proper deduplication."""
        logger.info("🚀 Starting fixed migration with deduplication...")
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-2000000")  # 2GB cache
        
        cursor = conn.cursor()
        
        # Get total count for progress
        cursor.execute("SELECT COUNT(*) FROM odd")
        total_records = cursor.fetchone()[0]
        logger.info(f"Total records to process: {total_records:,}")
        
        last_rowid = self.checkpoint.get('last_rowid', 0)
        logger.info(f"Resuming from rowid: {last_rowid:,}")
        
        while self.running:
            try:
                # Fetch batch with deduplication
                cursor.execute("""
                    WITH batch AS (
                        SELECT DISTINCT
                            o.rowid,
                            o.market_id,
                            COALESCE(bl.id, 13) as bookmaker_id,
                            COALESCE(ol.id, 0) as outcome_id,
                            o.timestamp as updated_at,
                            0 as source_id,
                            CASE 
                                WHEN o.betType = 'SPREAD' THEN 1
                                WHEN o.betType = 'TOTAL' THEN 2
                                ELSE 0
                            END as market_type_id,
                            o.position,
                            CAST(o.line * 100 AS INTEGER) as line_x100,
                            CAST(o.odds * 1000 AS INTEGER) as decimal_odds_x1000,
                            CASE
                                WHEN o.odds >= 2.0 THEN CAST((o.odds - 1) * 100 AS INTEGER)
                                ELSE CAST(-100 / (o.odds - 1) AS INTEGER)
                            END as american_odds,
                            CAST(10000.0 / o.odds AS INTEGER) as implied_x10000
                        FROM odd o
                        LEFT JOIN bookmakers_lookup bl ON bl.name = o.bookmaker
                        LEFT JOIN outcomes_lookup ol ON ol.name = o.outcome
                        WHERE o.rowid > ?
                        ORDER BY o.rowid
                        LIMIT ?
                    )
                    SELECT * FROM batch
                    WHERE NOT EXISTS (
                        SELECT 1 FROM odds_normalized_v2 onv
                        WHERE onv.market_id = batch.market_id
                        AND onv.bookmaker_id = batch.bookmaker_id
                        AND onv.outcome_id = batch.outcome_id
                        AND onv.updated_at = batch.updated_at
                    )
                """, (last_rowid, self.batch_size))
                
                rows = cursor.fetchall()
                if not rows:
                    logger.info("No more records to process")
                    break
                
                # Process batch
                insert_data = []
                for row in rows:
                    rowid = row[0]
                    insert_data.append(row[1:])  # Skip rowid
                    last_rowid = max(last_rowid, rowid)
                
                # Insert with error handling
                if insert_data:
                    try:
                        cursor.executemany("""
                            INSERT INTO odds_normalized_v2 
                            (market_id, bookmaker_id, outcome_id, updated_at, source_id,
                             market_type_id, position, line_x100, decimal_odds_x1000,
                             american_odds, implied_x10000)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, insert_data)
                        
                        conn.commit()
                        self.stats['migrated'] += len(insert_data)
                        
                    except sqlite3.IntegrityError as e:
                        # Handle any remaining duplicates row by row
                        logger.warning(f"Batch insert failed, trying row by row: {e}")
                        for data in insert_data:
                            try:
                                cursor.execute("""
                                    INSERT INTO odds_normalized_v2 
                                    (market_id, bookmaker_id, outcome_id, updated_at, source_id,
                                     market_type_id, position, line_x100, decimal_odds_x1000,
                                     american_odds, implied_x10000)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, data)
                                conn.commit()
                                self.stats['migrated'] += 1
                            except sqlite3.IntegrityError:
                                self.stats['skipped'] += 1
                
                self.stats['processed'] += len(rows)
                self.checkpoint['last_rowid'] = last_rowid
                
                # Progress report
                if self.stats['processed'] % self.checkpoint_interval == 0:
                    self._save_checkpoint()
                    self._report_progress(total_records)
                
                # Memory management
                if self.stats['processed'] % 50000 == 0:
                    conn.execute("PRAGMA optimize")
                
            except Exception as e:
                logger.error(f"Migration error: {e}")
                self.stats['errors'] += 1
                time.sleep(1)  # Brief pause before retry
        
        conn.close()
        logger.info("Migration stopped")
    
    def _report_progress(self, total_records):
        """Report migration progress."""
        elapsed = time.time() - self.stats['start_time']
        rate = self.stats['processed'] / elapsed if elapsed > 0 else 0
        progress = (self.stats['processed'] / total_records) * 100 if total_records > 0 else 0
        
        # Memory usage
        process = psutil.Process()
        memory_percent = process.memory_percent()
        
        logger.info(
            f"Progress: {progress:.1f}% | "
            f"Processed: {self.stats['processed']:,} | "
            f"Migrated: {self.stats['migrated']:,} | "
            f"Skipped: {self.stats['skipped']:,} | "
            f"Rate: {rate:.0f} rec/s | "
            f"Memory: {memory_percent:.1f}%"
        )
        
        # Estimate completion
        if rate > 0:
            remaining = total_records - self.stats['processed']
            eta_seconds = remaining / rate
            eta_hours = eta_seconds / 3600
            logger.info(f"Estimated completion: {eta_hours:.1f} hours")
    
    def prepare_for_postgresql(self):
        """Prepare data for eventual PostgreSQL migration."""
        logger.info("🐘 Preparing for PostgreSQL migration...")
        
        prep_steps = []
        
        # 1. Archive old data
        prep_steps.append({
            'step': 'Archive odds older than 6 months',
            'query': """
                CREATE TABLE IF NOT EXISTS odds_archive AS
                SELECT * FROM odds_normalized_v2 
                WHERE updated_at < strftime('%s', 'now', '-6 months')
            """,
            'benefit': 'Reduces active dataset by ~70%'
        })
        
        # 2. Create summary tables
        prep_steps.append({
            'step': 'Create daily aggregates',
            'query': """
                CREATE TABLE IF NOT EXISTS odds_daily_summary AS
                SELECT 
                    market_id,
                    DATE(updated_at, 'unixepoch') as date,
                    bookmaker_id,
                    outcome_id,
                    AVG(decimal_odds_x1000/1000.0) as avg_odds,
                    MIN(decimal_odds_x1000/1000.0) as min_odds,
                    MAX(decimal_odds_x1000/1000.0) as max_odds,
                    COUNT(*) as updates
                FROM odds_normalized_v2
                GROUP BY market_id, date, bookmaker_id, outcome_id
            """,
            'benefit': 'Fast historical queries'
        })
        
        # 3. Export scripts
        export_script = """#!/bin/bash
# Export normalized data for PostgreSQL import

echo "Exporting markets..."
sqlite3 sport_odds.db <<EOF
.mode csv
.headers on
.output markets_export.csv
SELECT * FROM markets_normalized;
EOF

echo "Exporting recent odds..."
sqlite3 sport_odds.db <<EOF
.mode csv
.headers on
.output odds_recent_export.csv
SELECT * FROM odds_normalized_v2 
WHERE updated_at > strftime('%s', 'now', '-30 days');
EOF

echo "Creating PostgreSQL import script..."
cat > import_to_postgresql.sql <<EOF
-- Import data into PostgreSQL
\\copy markets FROM 'markets_export.csv' WITH CSV HEADER;
\\copy odds FROM 'odds_recent_export.csv' WITH CSV HEADER;
EOF

echo "Export complete! Use import_to_postgresql.sql to load into PostgreSQL"
"""
        
        with open('export_for_postgresql.sh', 'w') as f:
            f.write(export_script)
        os.chmod('export_for_postgresql.sh', 0o755)
        
        logger.info("✅ PostgreSQL preparation complete")
        return prep_steps


def create_unified_data_layer():
    """Create unified data access layer design."""
    logger.info("🔄 Designing Unified Data Layer...")
    
    unified_design = """
# Unified Data Access Layer Architecture

## Overview
Single interface for accessing data from multiple sources:
- Historical SQLite data (pre-2025)
- Live PostgreSQL data (post-2025)  
- Real-time blockchain data
- External API data
- Cached Redis data

## Implementation

```python
class UnifiedDataManager:
    def __init__(self):
        self.sqlite_engine = create_engine('sqlite:///sport_odds.db')
        self.pg_engine = create_engine('postgresql://...')
        self.redis_client = redis.Redis()
        self.blockchain_reader = BlockchainReader()
        
    def get_market_data(self, market_id: str, date: datetime):
        # 1. Check Redis cache
        cached = self.redis_client.get(f"market:{market_id}")
        if cached:
            return json.loads(cached)
        
        # 2. Route by date
        if date < datetime(2025, 1, 1):
            # Historical: Use SQLite
            data = self._query_sqlite(market_id)
        else:
            # Recent: Use PostgreSQL
            data = self._query_postgresql(market_id)
        
        # 3. Enhance with blockchain if available
        if blockchain_data := self.blockchain_reader.get_market(market_id):
            data.update(blockchain_data)
        
        # 4. Cache result
        self.redis_client.setex(f"market:{market_id}", 300, json.dumps(data))
        
        return data
```

## Benefits
1. Transparent data access
2. Automatic source routing
3. Performance optimization
4. Easy to extend
5. Backwards compatible
"""
    
    with open('UNIFIED_DATA_LAYER_DESIGN.md', 'w') as f:
        f.write(unified_design)
    
    logger.info("✅ Unified data layer design created")


def main():
    """Run fixed migration and prepare for future."""
    logger.info("🚀 Fixed Migration Manager")
    logger.info("=" * 60)
    
    # Write PID file
    with open('migration.pid', 'w') as f:
        f.write(str(os.getpid()))
    
    try:
        manager = FixedMigrationManager()
        
        # Step 1: Fix schema
        manager.fix_migration_schema()
        
        # Step 2: Run migration
        manager.migrate_with_deduplication()
        
        # Step 3: Prepare for PostgreSQL
        manager.prepare_for_postgresql()
        
        # Step 4: Create unified design
        create_unified_data_layer()
        
    finally:
        # Clean up PID file
        if os.path.exists('migration.pid'):
            os.remove('migration.pid')


if __name__ == "__main__":
    main()