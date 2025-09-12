#!/usr/bin/env python3
"""Background migration using ORM models with automatic resource management."""

import logging
import time
import os
import gc
import signal
import sys
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import NullPool
from models import Odd, Base
from database_v2 import db_manager
import psutil
from typing import Dict, Optional
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('background_migration.log'),
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


class BackgroundOddsMigrator:
    """Background migrator using ORM models with automatic throttling."""
    
    def __init__(self, batch_size=1000, sleep_interval=1, nice_level=19):
        self.batch_size = batch_size
        self.sleep_interval = sleep_interval
        self.nice_level = nice_level
        self.lookup_cache = {}
        self.running = True
        self.state_file = 'migration_state.json'
        self.stats = self._load_state()
        
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
        self._load_lookups()
    
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
                'timeout': 30,
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
    
    def _load_lookups(self):
        """Load lookup tables using ORM."""
        session = self.Session()
        try:
            # Use raw SQL for lookups since we don't have models for them
            from sqlalchemy import text
            
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
            'start_time': time.time()
        }
    
    def _save_state(self):
        """Save current migration state."""
        try:
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
        if memory_percent > 80:
            logger.warning(f"High memory usage: {memory_percent}%")
            gc.collect()
            time.sleep(5)
            return False
        
        if cpu_percent > 80:
            time.sleep(2)
        elif cpu_percent > 60:
            time.sleep(0.5)
        
        if disk_usage.percent > 90:
            logger.error("Disk space critically low!")
            return False
        
        return True
    
    def _migrate_batch(self, session) -> tuple[int, int]:
        """Migrate a batch of records using ORM."""
        # Query batch using ORM
        odds = session.query(Odd).filter(
            Odd.id > self.stats['last_id']
        ).limit(self.batch_size).all()
        
        if not odds:
            return 0, 0
        
        migrated = 0
        skipped = 0
        normalized_records = []
        
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
                outcome_id = {'option_1': 0, 'option_2': 1, 'option_3': 2}.get(odd.outcome)
                if outcome_id is None:
                    skipped += 1
                    continue
                
                # Create normalized record
                norm_odd = OddNormalized(
                    market_id=odd.source_id,
                    bookmaker_id=bookmaker_id,
                    source_id=source_id,
                    market_type_id=market_type_id,
                    outcome_id=outcome_id,
                    position=odd.position if odd.position is not None else 0,
                    line_x100=int(odd.line * 100) if odd.line else None,
                    decimal_odds_x1000=int(odd.decimal_odds * 1000) if odd.decimal_odds else 0,
                    american_odds=int(odd.american_odds) if odd.american_odds else None,
                    implied_x10000=int(odd.normalized_implied * 10000) if odd.normalized_implied else None,
                    updated_at=int(odd.updated_at.timestamp()) if odd.updated_at else 0
                )
                
                normalized_records.append(norm_odd)
                migrated += 1
                
            except Exception as e:
                logger.debug(f"Failed to migrate odd {odd.id}: {e}")
                skipped += 1
        
        # Bulk insert normalized records
        if normalized_records:
            try:
                # Use bulk_insert_mappings for efficiency
                session.bulk_insert_mappings(
                    OddNormalized,
                    [
                        {
                            'market_id': r.market_id,
                            'bookmaker_id': r.bookmaker_id,
                            'source_id': r.source_id,
                            'market_type_id': r.market_type_id,
                            'outcome_id': r.outcome_id,
                            'position': r.position,
                            'line_x100': r.line_x100,
                            'decimal_odds_x1000': r.decimal_odds_x1000,
                            'american_odds': r.american_odds,
                            'implied_x10000': r.implied_x10000,
                            'updated_at': r.updated_at
                        }
                        for r in normalized_records
                    ]
                )
                session.commit()
            except Exception as e:
                logger.error(f"Bulk insert failed: {e}")
                session.rollback()
                # Try one by one as fallback
                for r in normalized_records:
                    try:
                        session.merge(r)
                        session.commit()
                    except:
                        session.rollback()
        
        # Update last processed ID
        if odds:
            self.stats['last_id'] = odds[-1].id
        
        # Clear ORM session to free memory
        session.expire_all()
        
        return migrated, skipped
    
    def run_background_migration(self):
        """Run migration in background with automatic resource management."""
        logger.info("Starting background migration...")
        logger.info(f"Batch size: {self.batch_size}, Sleep interval: {self.sleep_interval}s")
        
        session = self.Session()
        
        try:
            # Get total count for progress
            total_count = session.query(Odd).count()
            logger.info(f"Total records to migrate: {total_count:,}")
            
            while self.running:
                # Check resources
                if not self._check_resources():
                    time.sleep(10)
                    continue
                
                # Migrate batch
                batch_start = time.time()
                migrated, skipped = self._migrate_batch(session)
                batch_time = time.time() - batch_start
                
                if migrated == 0 and skipped == 0:
                    logger.info("Migration complete!")
                    break
                
                # Update stats
                self.stats['total_migrated'] += migrated
                self.stats['total_skipped'] += skipped
                self.stats['total_processed'] += migrated + skipped
                
                # Calculate rate
                rate = migrated / batch_time if batch_time > 0 else 0
                
                # Progress update every 100 batches
                if self.stats['total_processed'] % (self.batch_size * 100) == 0:
                    progress = self.stats['total_processed'] / total_count * 100
                    elapsed = time.time() - self.stats['start_time']
                    overall_rate = self.stats['total_migrated'] / elapsed if elapsed > 0 else 0
                    
                    logger.info(f"Progress: {progress:.1f}% | "
                               f"Migrated: {self.stats['total_migrated']:,} | "
                               f"Rate: {rate:.0f} rec/s | "
                               f"Avg: {overall_rate:.0f} rec/s | "
                               f"Memory: {psutil.virtual_memory().percent}%")
                    
                    # Save state periodically
                    self._save_state()
                
                # Sleep to reduce system load
                time.sleep(self.sleep_interval)
                
                # Periodic garbage collection
                if self.stats['total_processed'] % 10000 == 0:
                    gc.collect()
        
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
        logger.info(f"Total time: {elapsed/3600:.1f} hours")
        logger.info(f"Average rate: {self.stats['total_migrated']/elapsed:.0f} records/sec")
        
        # Clean up state file
        if os.path.exists(self.state_file):
            os.remove(self.state_file)


def main():
    """Run background migration with configurable options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Background ORM-based migration')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Records per batch (default: 1000)')
    parser.add_argument('--sleep', type=float, default=1.0,
                        help='Sleep interval between batches in seconds (default: 1.0)')
    parser.add_argument('--nice', type=int, default=19,
                        help='Process nice level 0-19 (default: 19 = lowest priority)')
    args = parser.parse_args()
    
    migrator = BackgroundOddsMigrator(
        batch_size=args.batch_size,
        sleep_interval=args.sleep,
        nice_level=args.nice
    )
    
    migrator.run_background_migration()


if __name__ == "__main__":
    main()