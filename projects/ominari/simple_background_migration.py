#!/usr/bin/env python3
"""Simple background migration that works with existing lookup tables."""

import logging
import time
import os
import signal
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from typing import Dict
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SimpleOddsMigrator:
    """Simple migrator that uses existing lookup values."""
    
    def __init__(self, batch_size=5000):
        self.batch_size = batch_size
        self.lookup_cache = {}
        self.running = True
        self.state_file = 'simple_migration_state.json'
        self.stats = self._load_state()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Create engine with optimizations
        self.engine = create_engine(
            "sqlite:///sport_odds.db",
            connect_args={
                'timeout': 300,
                'check_same_thread': False,
            }
        )
        
        self._load_existing_lookups()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received shutdown signal, saving state...")
        self.running = False
        self._save_state()
        sys.exit(0)
    
    def _load_existing_lookups(self):
        """Load only existing lookup values - no expansion."""
        with self.engine.connect() as conn:
            # Load existing bookmakers
            result = conn.execute(text("SELECT id, name FROM lu_bookmakers"))
            self.lookup_cache['bookmaker'] = {name: id for id, name in result}
            
            # Load existing sources
            result = conn.execute(text("SELECT id, name FROM lu_sources"))
            self.lookup_cache['source'] = {name: id for id, name in result}
            
            # Load existing market types
            result = conn.execute(text("SELECT id, name FROM lu_market_types"))
            self.lookup_cache['market_type'] = {name: id for id, name in result}
            
            logger.info(f"Loaded lookups: {len(self.lookup_cache['bookmaker'])} bookmakers, "
                       f"{len(self.lookup_cache['source'])} sources, "
                       f"{len(self.lookup_cache['market_type'])} market types")
    
    def _load_state(self) -> Dict:
        """Load migration state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Resumed from saved state: {state['total_migrated']:,} records migrated")
                return state
            except:
                pass
        
        return {
            'last_id': 0,
            'total_processed': 0,
            'total_migrated': 0,
            'total_skipped': 0,
            'start_time': time.time()
        }
    
    def _save_state(self):
        """Save current migration state."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.stats, f)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def run_migration(self):
        """Run the migration with simple batch processing."""
        logger.info("Starting simple migration...")
        logger.info(f"Batch size: {self.batch_size}")
        
        with self.engine.begin() as conn:
            # Set pragmas for better performance
            conn.execute(text("PRAGMA journal_mode = WAL"))
            conn.execute(text("PRAGMA synchronous = NORMAL"))
            conn.execute(text("PRAGMA cache_size = -2000000"))  # 2GB cache
            conn.execute(text("PRAGMA temp_store = MEMORY"))
            
            # Get total count
            result = conn.execute(text("SELECT COUNT(*) FROM odd WHERE id > :last_id"), 
                                {"last_id": self.stats['last_id']})
            remaining_count = result.scalar()
            logger.info(f"Records remaining to migrate: {remaining_count:,}")
        
        while self.running:
            batch_start_time = time.time()
            
            # Process a batch
            migrated, skipped = self._process_batch()
            
            if migrated == 0 and skipped == 0:
                logger.info("Migration complete!")
                break
            
            # Update stats
            self.stats['total_migrated'] += migrated
            self.stats['total_skipped'] += skipped
            self.stats['total_processed'] += migrated + skipped
            
            # Calculate rate
            batch_time = time.time() - batch_start_time
            rate = (migrated + skipped) / batch_time if batch_time > 0 else 0
            
            # Progress update every 10 batches
            if self.stats['total_processed'] % (self.batch_size * 10) == 0:
                elapsed = time.time() - self.stats['start_time']
                overall_rate = self.stats['total_processed'] / elapsed if elapsed > 0 else 0
                
                logger.info(f"Processed: {self.stats['total_processed']:,} | "
                           f"Migrated: {self.stats['total_migrated']:,} | "
                           f"Skipped: {self.stats['total_skipped']:,} | "
                           f"Rate: {overall_rate:.0f} rec/s")
                
                # Save state
                self._save_state()
            
            # Brief pause to avoid overwhelming the system
            time.sleep(0.1)
        
        # Final stats
        elapsed = time.time() - self.stats['start_time']
        logger.info("="*80)
        logger.info("MIGRATION COMPLETE!")
        logger.info(f"Total migrated: {self.stats['total_migrated']:,}")
        logger.info(f"Total skipped: {self.stats['total_skipped']:,}")
        logger.info(f"Total time: {elapsed/3600:.1f} hours")
        logger.info(f"Average rate: {self.stats['total_processed']/elapsed:.0f} records/sec")
    
    def _process_batch(self) -> tuple[int, int]:
        """Process a single batch of odds records."""
        migrated = 0
        skipped = 0
        
        with self.engine.begin() as conn:
            # Get batch of odds
            result = conn.execute(text("""
                SELECT id, source_id, market_type, bookmaker, source, outcome, 
                       position, line, decimal_odds, american_odds, normalized_implied, updated_at
                FROM odd
                WHERE id > :last_id
                ORDER BY id
                LIMIT :batch_size
            """), {"last_id": self.stats['last_id'], "batch_size": self.batch_size})
            
            rows = result.fetchall()
            if not rows:
                return 0, 0
            
            records_to_insert = []
            
            for row in rows:
                try:
                    # Check if we have lookup values
                    bookmaker_id = self.lookup_cache['bookmaker'].get(row[3])
                    source_id = self.lookup_cache['source'].get(row[4])
                    market_type_id = self.lookup_cache['market_type'].get(row[2])
                    
                    if bookmaker_id is None or source_id is None or market_type_id is None:
                        skipped += 1
                        continue
                    
                    # Map outcome
                    outcome_mapping = {
                        'option_1': 0, 'option_2': 1, 'option_3': 2,
                        'home': 0, 'draw': 1, 'away': 2,
                        'yes': 0, 'no': 1,
                        'over': 0, 'under': 1
                    }
                    
                    outcome = str(row[5]).lower() if row[5] else ''
                    outcome_id = outcome_mapping.get(outcome)
                    
                    if outcome_id is None:
                        # Try numeric parsing
                        try:
                            outcome_id = int(outcome.replace('option_', '')) - 1
                        except:
                            skipped += 1
                            continue
                    
                    # Convert timestamp
                    if isinstance(row[11], str):
                        timestamp = int(datetime.fromisoformat(row[11].replace('Z', '+00:00')).timestamp())
                    elif isinstance(row[11], datetime):
                        timestamp = int(row[11].timestamp())
                    else:
                        timestamp = int(datetime.now().timestamp())
                    
                    # Create record
                    record = {
                        'market_id': row[1],
                        'bookmaker_id': bookmaker_id,
                        'source_id': source_id,
                        'market_type_id': market_type_id,
                        'outcome_id': outcome_id,
                        'position': row[6] if row[6] is not None else 0,
                        'line_x100': int(row[7] * 100) if row[7] else None,
                        'decimal_odds_x1000': int(row[8] * 1000) if row[8] else None,
                        'american_odds': int(row[9]) if row[9] else None,
                        'implied_x10000': int(row[10] * 10000) if row[10] else None,
                        'updated_at': timestamp
                    }
                    
                    records_to_insert.append(record)
                    
                except Exception as e:
                    logger.debug(f"Failed to process row {row[0]}: {e}")
                    skipped += 1
            
            # Update last ID
            self.stats['last_id'] = rows[-1][0]
            
            # Insert records
            if records_to_insert:
                try:
                    # Use INSERT OR IGNORE to handle duplicates
                    for record in records_to_insert:
                        conn.execute(text("""
                            INSERT OR IGNORE INTO odds_normalized 
                            (market_id, bookmaker_id, source_id, market_type_id, outcome_id, 
                             position, line_x100, decimal_odds_x1000, american_odds, implied_x10000, updated_at)
                            VALUES 
                            (:market_id, :bookmaker_id, :source_id, :market_type_id, :outcome_id,
                             :position, :line_x100, :decimal_odds_x1000, :american_odds, :implied_x10000, :updated_at)
                        """), record)
                    
                    migrated = len(records_to_insert)
                    
                except Exception as e:
                    logger.error(f"Failed to insert batch: {e}")
                    # Try one by one as fallback
                    for record in records_to_insert:
                        try:
                            conn.execute(text("""
                                INSERT OR IGNORE INTO odds_normalized 
                                (market_id, bookmaker_id, source_id, market_type_id, outcome_id, 
                                 position, line_x100, decimal_odds_x1000, american_odds, implied_x10000, updated_at)
                                VALUES 
                                (:market_id, :bookmaker_id, :source_id, :market_type_id, :outcome_id,
                                 :position, :line_x100, :decimal_odds_x1000, :american_odds, :implied_x10000, :updated_at)
                            """), record)
                            migrated += 1
                        except:
                            skipped += 1
        
        return migrated, skipped


def main():
    """Run simple migration."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple background migration')
    parser.add_argument('--batch-size', type=int, default=5000,
                        help='Records per batch (default: 5000)')
    args = parser.parse_args()
    
    migrator = SimpleOddsMigrator(batch_size=args.batch_size)
    migrator.run_migration()


if __name__ == "__main__":
    main()