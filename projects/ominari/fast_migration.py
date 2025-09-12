#!/usr/bin/env python3
"""
Fast migration using direct SQL approach.
Bypasses ORM for maximum speed.
"""

import sqlite3
import logging
import time
import json
import os
from datetime import datetime
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FastMigration:
    def __init__(self, source_db='sport_odds.db', batch_size=50000):
        self.source_db = source_db
        self.batch_size = batch_size
        self.stats = {
            'start_time': time.time(),
            'records_processed': 0,
            'records_migrated': 0,
            'last_id': 0
        }
        
    def run(self):
        """Run fast migration using direct SQL."""
        logger.info("Starting fast migration...")
        
        # Connect to source database
        source_conn = sqlite3.connect(self.source_db)
        source_conn.row_factory = sqlite3.Row
        
        # Connect to destination (create normalized table if needed)
        dest_conn = sqlite3.connect('sport_odds_normalized.db')
        
        # Create normalized table
        dest_conn.execute("""
            CREATE TABLE IF NOT EXISTS odds_normalized (
                market_id TEXT NOT NULL,
                bookmaker_id INTEGER NOT NULL,
                outcome_id INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                source_id INTEGER NOT NULL,
                market_type_id INTEGER NOT NULL,
                position INTEGER NOT NULL,
                line_x100 INTEGER,
                decimal_odds_x1000 INTEGER NOT NULL,
                american_odds INTEGER,
                implied_x10000 INTEGER,
                PRIMARY KEY (market_id, bookmaker_id, outcome_id, updated_at)
            )
        """)
        
        # Create indexes for performance
        dest_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_updated 
            ON odds_normalized(market_id, updated_at DESC)
        """)
        
        dest_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_updated_at 
            ON odds_normalized(updated_at DESC)
        """)
        
        # Load lookup tables
        lookups = self._load_lookups()
        
        # Get last processed ID
        last_id = self.stats.get('last_id', 0)
        
        # Count total records
        total_count = source_conn.execute(
            "SELECT COUNT(*) FROM odd WHERE id > ?", (last_id,)
        ).fetchone()[0]
        
        logger.info(f"Total records to process: {total_count:,}")
        
        # Process in batches
        batch_count = 0
        
        while True:
            # Check resources
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                logger.warning(f"High memory usage: {memory.percent}%, pausing...")
                time.sleep(30)
                continue
                
            # Fetch batch
            cursor = source_conn.execute("""
                SELECT id, source_id, source, bookmaker, market_type, outcome, 
                       position, line, decimal_odds, american_odds, normalized_implied,
                       updated_at
                FROM odd
                WHERE id > ?
                ORDER BY id
                LIMIT ?
            """, (last_id, self.batch_size))
            
            rows = cursor.fetchall()
            
            if not rows:
                break
                
            # Process batch
            normalized_records = []
            
            for row in rows:
                try:
                    # Normalize record
                    normalized = self._normalize_record(row, lookups)
                    if normalized:
                        normalized_records.append(normalized)
                except Exception as e:
                    logger.error(f"Error normalizing record {row['id']}: {e}")
                    
            # Insert batch
            if normalized_records:
                try:
                    dest_conn.executemany("""
                        INSERT OR IGNORE INTO odds_normalized (
                            market_id, bookmaker_id, outcome_id, updated_at,
                            source_id, market_type_id, position, line_x100,
                            decimal_odds_x1000, american_odds, implied_x10000
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, normalized_records)
                    
                    dest_conn.commit()
                    self.stats['records_migrated'] += len(normalized_records)
                except Exception as e:
                    logger.error(f"Error inserting batch: {e}")
                    
            # Update stats
            last_id = rows[-1]['id']
            self.stats['last_id'] = last_id
            self.stats['records_processed'] += len(rows)
            batch_count += 1
            
            # Report progress
            if batch_count % 10 == 0:
                elapsed = time.time() - self.stats['start_time']
                rate = self.stats['records_processed'] / elapsed
                pct = (self.stats['records_processed'] / total_count) * 100
                
                logger.info(
                    f"Progress: {pct:.2f}% | "
                    f"Processed: {self.stats['records_processed']:,} | "
                    f"Migrated: {self.stats['records_migrated']:,} | "
                    f"Rate: {rate:.0f} rec/sec"
                )
                
                # Save checkpoint
                self._save_checkpoint()
                
            # Small delay to avoid overwhelming system
            time.sleep(0.01)
            
        # Final stats
        self._report_final_stats()
        
        # Close connections
        source_conn.close()
        dest_conn.close()
        
    def _load_lookups(self):
        """Load lookup tables."""
        # Simplified lookups - in production, load from actual lookup tables
        return {
            'source': {
                'overtime': 1,
                'free_data': 2,
                'blockchain': 3,
                'manual': 4
            },
            'bookmaker': {
                'fanduel': 1,
                'draftkings': 2,
                'betmgm': 3,
                'caesars': 4,
                'pointsbet': 5,
                'bet365': 6,
                'betrivers': 7,
                'unibet': 8,
                'barstool': 9,
                'wynn': 10,
                'overtime': 11,
                'overtime_v1': 12,
                'overtime_v2': 13,
                'default': 99
            },
            'market_type': {
                'h2h': 1,
                'moneyline': 1,
                'spreads': 2,
                'totals': 3,
                'draw_no_bet': 4,
                'double_chance': 5
            },
            'outcome': {
                'home': 1,
                'away': 2,
                'draw': 3,
                'over': 4,
                'under': 5,
                'yes': 6,
                'no': 7,
                'option_0': 1,
                'option_1': 2,
                'option_2': 3
            }
        }
        
    def _normalize_record(self, row, lookups):
        """Normalize a single record."""
        # Get IDs from lookups
        source_id = lookups['source'].get(row['source'], 4)  # Default to manual
        bookmaker_id = lookups['bookmaker'].get(row['bookmaker'], 99)
        market_type_id = lookups['market_type'].get(row['market_type'], 1)
        outcome_id = lookups['outcome'].get(row['outcome'], 1)
        
        # Convert values to integers
        line_x100 = int(row['line'] * 100) if row['line'] else None
        decimal_odds_x1000 = int(row['decimal_odds'] * 1000)
        american_odds = int(row['american_odds']) if row['american_odds'] else None
        implied_x10000 = int(row['normalized_implied'] * 10000) if row['normalized_implied'] else None
        
        # Create market_id (simplified - in production use proper hashing)
        market_id = f"{row['source_id']}_{row['market_type']}".replace(' ', '_')[:68]
        
        # Convert timestamp
        if isinstance(row['updated_at'], str):
            try:
                dt = datetime.fromisoformat(row['updated_at'].replace('Z', '+00:00'))
                updated_at = int(dt.timestamp())
            except:
                updated_at = int(time.time())
        else:
            updated_at = int(row['updated_at'])
            
        return (
            market_id,
            bookmaker_id,
            outcome_id,
            updated_at,
            source_id,
            market_type_id,
            row['position'] or 0,
            line_x100,
            decimal_odds_x1000,
            american_odds,
            implied_x10000
        )
        
    def _save_checkpoint(self):
        """Save migration checkpoint."""
        with open('fast_migration_checkpoint.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
            
    def _report_final_stats(self):
        """Report final statistics."""
        elapsed = time.time() - self.stats['start_time']
        
        logger.info("=" * 60)
        logger.info("Migration Complete!")
        logger.info(f"Total processed: {self.stats['records_processed']:,}")
        logger.info(f"Total migrated: {self.stats['records_migrated']:,}")
        logger.info(f"Time elapsed: {elapsed/3600:.2f} hours")
        logger.info(f"Average rate: {self.stats['records_processed']/elapsed:.0f} records/sec")
        logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast database migration')
    parser.add_argument('--batch-size', type=int, default=50000, help='Batch size')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    migrator = FastMigration(batch_size=args.batch_size)
    
    if args.resume and os.path.exists('fast_migration_checkpoint.json'):
        with open('fast_migration_checkpoint.json', 'r') as f:
            migrator.stats = json.load(f)
        logger.info(f"Resuming from record {migrator.stats['last_id']:,}")
        
    migrator.run()