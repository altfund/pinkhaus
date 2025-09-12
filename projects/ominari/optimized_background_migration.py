#!/usr/bin/env python3
"""Optimized background migration with duplicate handling and performance improvements."""

import logging
import time
import os
import gc
import signal
import sys
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, and_, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import NullPool
from models import Odd, Base
from database_v2 import db_manager
import psutil
from typing import Dict, Optional
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimized_migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OddNormalized(Base):
    """Normalized odds model."""
    __tablename__ = "odds_normalized"
    
    from sqlalchemy import Column, String, SmallInteger, Integer
    
    market_id = Column(String(68), primary_key=True)
    bookmaker_id = Column(SmallInteger, primary_key=True)
    outcome_id = Column(SmallInteger, primary_key=True)
    updated_at = Column(Integer, primary_key=True)
    
    source_id = Column(SmallInteger, nullable=False)
    market_type_id = Column(SmallInteger, nullable=False)
    position = Column(SmallInteger, nullable=False)
    line_x100 = Column(SmallInteger)
    decimal_odds_x1000 = Column(SmallInteger, nullable=False)
    american_odds = Column(SmallInteger)
    implied_x10000 = Column(SmallInteger)


class OptimizedOddsMigrator:
    """Optimized migrator with duplicate handling and parallel processing."""
    
    def __init__(self, batch_size=10000, sleep_interval=0.1, nice_level=19, num_threads=4):
        self.batch_size = batch_size
        self.sleep_interval = sleep_interval
        self.nice_level = nice_level
        self.num_threads = num_threads
        self.lookup_cache = {}
        self.running = True
        self.state_file = 'optimized_migration_state.json'
        self.stats = self._load_state()
        self.stats_lock = threading.Lock()
        
        # Set process priority to lowest
        try:
            os.nice(nice_level)
            logger.info(f"Set process nice level to {nice_level}")
        except:
            logger.warning("Could not set process priority")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._setup_db_sessions()
        self._expand_and_load_lookups()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received shutdown signal, saving state...")
        self.running = False
        self._save_state()
        sys.exit(0)
    
    def _setup_db_sessions(self):
        """Setup database sessions with minimal resource usage."""
        # Create engine with no connection pooling
        self.engine = create_engine(
            "sqlite:///sport_odds.db",
            poolclass=NullPool,
            connect_args={
                'timeout': 60,
                'check_same_thread': False,
                'isolation_level': 'DEFERRED'
            }
        )
        
        # Create session factory
        self.Session = scoped_session(sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False
        ))
    
    def _expand_and_load_lookups(self):
        """Expand lookup tables with all unique values and load them."""
        session = self.Session()
        try:
            logger.info("Expanding lookup tables with all unique values...")
            
            # Get all unique bookmakers
            result = session.execute(text("""
                INSERT OR IGNORE INTO lu_bookmakers (name)
                SELECT DISTINCT bookmaker 
                FROM odd 
                WHERE bookmaker IS NOT NULL 
                AND bookmaker NOT IN (SELECT name FROM lu_bookmakers)
                LIMIT 1000
            """))
            session.commit()
            logger.info(f"Added {result.rowcount} new bookmakers")
            
            # Get all unique sources
            result = session.execute(text("""
                INSERT OR IGNORE INTO lu_sources (name)
                SELECT DISTINCT source 
                FROM odd 
                WHERE source IS NOT NULL 
                AND source NOT IN (SELECT name FROM lu_sources)
                LIMIT 100
            """))
            session.commit()
            logger.info(f"Added {result.rowcount} new sources")
            
            # Get all unique market types
            result = session.execute(text("""
                INSERT OR IGNORE INTO lu_market_types (name)
                SELECT DISTINCT market_type 
                FROM odd 
                WHERE market_type IS NOT NULL 
                AND market_type NOT IN (SELECT name FROM lu_market_types)
                LIMIT 100
            """))
            session.commit()
            logger.info(f"Added {result.rowcount} new market types")
            
            # Load lookups into memory
            result = session.execute(text("SELECT id, name FROM lu_bookmakers"))
            self.lookup_cache['bookmaker'] = {name: id for id, name in result}
            
            result = session.execute(text("SELECT id, name FROM lu_sources"))
            self.lookup_cache['source'] = {name: id for id, name in result}
            
            result = session.execute(text("SELECT id, name FROM lu_market_types"))
            self.lookup_cache['market_type'] = {name: id for id, name in result}
            
            logger.info(f"Loaded lookups: {len(self.lookup_cache['bookmaker'])} bookmakers, "
                       f"{len(self.lookup_cache['source'])} sources, "
                       f"{len(self.lookup_cache['market_type'])} market types")
        finally:
            session.close()
    
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
            'total_duplicates': 0,
            'start_time': time.time()
        }
    
    def _save_state(self):
        """Save current migration state."""
        try:
            with self.stats_lock:
                with open(self.state_file, 'w') as f:
                    json.dump(self.stats, f)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _check_resources(self) -> bool:
        """Check system resources and throttle if needed."""
        # Memory check
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Disk I/O check
        disk_usage = psutil.disk_usage('/')
        
        # Adaptive throttling
        if memory_percent > 85:
            logger.warning(f"High memory usage: {memory_percent}%")
            gc.collect()
            time.sleep(10)
            return False
        
        if cpu_percent > 90:
            time.sleep(2)
        elif cpu_percent > 70:
            time.sleep(0.5)
        
        if disk_usage.percent > 95:
            logger.error("Disk space critically low!")
            return False
        
        return True
    
    def _migrate_batch(self, batch_start_id: int, batch_end_id: int) -> tuple[int, int, int]:
        """Migrate a batch of records with duplicate handling."""
        session = self.Session()
        
        try:
            # Query batch using ORM with explicit ID range
            odds = session.query(Odd).filter(
                and_(Odd.id > batch_start_id, Odd.id <= batch_end_id)
            ).all()
            
            if not odds:
                return 0, 0, 0
            
            migrated = 0
            skipped = 0
            duplicates = 0
            
            # Group records by unique key to handle duplicates
            unique_records = {}
            
            for odd in odds:
                try:
                    # Get lookup IDs
                    bookmaker_id = self.lookup_cache['bookmaker'].get(odd.bookmaker)
                    source_id = self.lookup_cache['source'].get(odd.source)
                    market_type_id = self.lookup_cache['market_type'].get(odd.market_type)
                    
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
                    outcome_id = outcome_mapping.get(odd.outcome.lower() if odd.outcome else None)
                    if outcome_id is None:
                        # Try numeric parsing
                        try:
                            outcome_id = int(odd.outcome.replace('option_', '')) - 1
                        except:
                            skipped += 1
                            continue
                    
                    # Create unique key
                    timestamp = int(odd.updated_at.timestamp()) if odd.updated_at else 0
                    unique_key = (odd.source_id, bookmaker_id, outcome_id, timestamp)
                    
                    # Keep only the latest odds for each unique key
                    if unique_key in unique_records:
                        duplicates += 1
                        # Keep the one with higher ID (more recent)
                        if odd.id <= unique_records[unique_key]['id']:
                            continue
                    
                    # Store record data
                    unique_records[unique_key] = {
                        'id': odd.id,
                        'market_id': odd.source_id,
                        'bookmaker_id': bookmaker_id,
                        'source_id': source_id,
                        'market_type_id': market_type_id,
                        'outcome_id': outcome_id,
                        'position': odd.position if odd.position is not None else 0,
                        'line_x100': int(odd.line * 100) if odd.line else None,
                        'decimal_odds_x1000': int(odd.decimal_odds * 1000) if odd.decimal_odds else 0,
                        'american_odds': int(odd.american_odds) if odd.american_odds else None,
                        'implied_x10000': int(odd.normalized_implied * 10000) if odd.normalized_implied else None,
                        'updated_at': timestamp
                    }
                    
                except Exception as e:
                    logger.debug(f"Failed to process odd {odd.id}: {e}")
                    skipped += 1
            
            # Bulk insert unique records
            if unique_records:
                records_to_insert = list(unique_records.values())
                
                # Use raw SQL for better performance with ON CONFLICT
                insert_sql = text("""
                    INSERT OR IGNORE INTO odds_normalized 
                    (market_id, bookmaker_id, source_id, market_type_id, outcome_id, 
                     position, line_x100, decimal_odds_x1000, american_odds, implied_x10000, updated_at)
                    VALUES 
                    (:market_id, :bookmaker_id, :source_id, :market_type_id, :outcome_id,
                     :position, :line_x100, :decimal_odds_x1000, :american_odds, :implied_x10000, :updated_at)
                """)
                
                # Execute in chunks to avoid too many parameters
                chunk_size = 1000
                for i in range(0, len(records_to_insert), chunk_size):
                    chunk = records_to_insert[i:i + chunk_size]
                    try:
                        for record in chunk:
                            session.execute(insert_sql, record)
                        session.commit()
                        migrated += len(chunk)
                    except Exception as e:
                        logger.error(f"Insert chunk failed: {e}")
                        session.rollback()
            
            # Clear ORM session to free memory
            session.expire_all()
            
            return migrated, skipped, duplicates
            
        finally:
            session.close()
    
    def _process_id_range(self, start_id: int, end_id: int, thread_id: int) -> tuple[int, int, int]:
        """Process a range of IDs in a thread."""
        total_migrated = 0
        total_skipped = 0
        total_duplicates = 0
        
        current_id = start_id
        while current_id < end_id and self.running:
            if not self._check_resources():
                time.sleep(5)
                continue
            
            batch_end = min(current_id + self.batch_size, end_id)
            
            # Process batch
            migrated, skipped, duplicates = self._migrate_batch(current_id, batch_end)
            
            total_migrated += migrated
            total_skipped += skipped
            total_duplicates += duplicates
            
            # Update global stats
            with self.stats_lock:
                self.stats['total_migrated'] += migrated
                self.stats['total_skipped'] += skipped
                self.stats['total_duplicates'] += duplicates
                self.stats['total_processed'] += migrated + skipped
                self.stats['last_id'] = max(self.stats['last_id'], batch_end)
            
            current_id = batch_end
            
            # Brief sleep to prevent overwhelming the system
            time.sleep(self.sleep_interval)
        
        return total_migrated, total_skipped, total_duplicates
    
    def run_optimized_migration(self):
        """Run migration with parallel processing."""
        logger.info("Starting optimized migration...")
        logger.info(f"Batch size: {self.batch_size}, Threads: {self.num_threads}")
        
        session = self.Session()
        
        try:
            # Get total count and max ID
            total_count = session.query(Odd).count()
            max_id_result = session.query(Odd.id).order_by(Odd.id.desc()).first()
            max_id = max_id_result[0] if max_id_result else 0
            
            logger.info(f"Total records to migrate: {total_count:,}")
            logger.info(f"Max ID: {max_id:,}")
            
            # Temporarily disable indexes for faster inserts
            logger.info("Optimizing table for bulk inserts...")
            session.execute(text("PRAGMA journal_mode = WAL"))
            session.execute(text("PRAGMA synchronous = NORMAL"))
            session.execute(text("PRAGMA cache_size = -2000000"))  # 2GB cache
            session.execute(text("PRAGMA temp_store = MEMORY"))
            session.commit()
            
            # Calculate ID ranges for parallel processing
            records_per_thread = (max_id - self.stats['last_id']) // self.num_threads
            
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = []
                
                for i in range(self.num_threads):
                    start_id = self.stats['last_id'] + (i * records_per_thread)
                    end_id = start_id + records_per_thread if i < self.num_threads - 1 else max_id
                    
                    future = executor.submit(self._process_id_range, start_id, end_id, i)
                    futures.append(future)
                
                # Monitor progress
                last_log_time = time.time()
                while any(not f.done() for f in futures):
                    time.sleep(10)
                    
                    # Progress update every 30 seconds
                    if time.time() - last_log_time > 30:
                        with self.stats_lock:
                            progress = self.stats['total_processed'] / total_count * 100 if total_count > 0 else 0
                            elapsed = time.time() - self.stats['start_time']
                            overall_rate = self.stats['total_migrated'] / elapsed if elapsed > 0 else 0
                            
                            logger.info(f"Progress: {progress:.1f}% | "
                                       f"Migrated: {self.stats['total_migrated']:,} | "
                                       f"Duplicates: {self.stats['total_duplicates']:,} | "
                                       f"Rate: {overall_rate:.0f} rec/s | "
                                       f"Memory: {psutil.virtual_memory().percent}%")
                            
                            # Save state periodically
                            self._save_state()
                        
                        last_log_time = time.time()
                
                # Get results from all threads
                for future in as_completed(futures):
                    try:
                        migrated, skipped, duplicates = future.result()
                        logger.info(f"Thread completed: {migrated:,} migrated, {skipped:,} skipped, {duplicates:,} duplicates")
                    except Exception as e:
                        logger.error(f"Thread failed: {e}")
        
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            self._save_state()
            raise
        
        finally:
            session.close()
            self.Session.remove()
        
        # Final stats
        elapsed = time.time() - self.stats['start_time']
        logger.info("="*80)
        logger.info("MIGRATION COMPLETE!")
        logger.info(f"Total migrated: {self.stats['total_migrated']:,}")
        logger.info(f"Total skipped: {self.stats['total_skipped']:,}")
        logger.info(f"Total duplicates handled: {self.stats['total_duplicates']:,}")
        logger.info(f"Total time: {elapsed/3600:.1f} hours")
        logger.info(f"Average rate: {self.stats['total_migrated']/elapsed:.0f} records/sec")
        
        # Clean up state file
        if os.path.exists(self.state_file):
            os.remove(self.state_file)


def main():
    """Run optimized migration with configurable options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized background migration')
    parser.add_argument('--batch-size', type=int, default=10000,
                        help='Records per batch (default: 10000)')
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of parallel threads (default: 4)')
    parser.add_argument('--sleep', type=float, default=0.1,
                        help='Sleep interval between batches in seconds (default: 0.1)')
    parser.add_argument('--nice', type=int, default=19,
                        help='Process nice level 0-19 (default: 19 = lowest priority)')
    args = parser.parse_args()
    
    migrator = OptimizedOddsMigrator(
        batch_size=args.batch_size,
        sleep_interval=args.sleep,
        nice_level=args.nice,
        num_threads=args.threads
    )
    
    migrator.run_optimized_migration()


if __name__ == "__main__":
    main()