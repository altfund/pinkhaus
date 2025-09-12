#!/usr/bin/env python3
"""Optimized migration script for odds normalization - runs as fast as possible."""

import logging
import time
import os
import psutil
from datetime import datetime, timedelta, timezone
from sqlalchemy import text, create_engine
from sqlalchemy.pool import NullPool
from database_v2 import db_manager
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedOddsMigrator:
    """High-performance odds migrator with resource management."""
    
    def __init__(self, chunk_size=500000, max_workers=None, memory_limit_gb=4):
        self.chunk_size = chunk_size
        self.max_workers = max_workers or min(mp.cpu_count() - 1, 4)
        self.memory_limit_gb = memory_limit_gb
        self.lookup_cache = {}
        self.stats = {
            'total_processed': 0,
            'total_migrated': 0,
            'start_time': time.time()
        }
        self._load_lookups()
        
    def _load_lookups(self):
        """Load lookup tables into memory once."""
        logger.info("Loading lookup tables...")
        
        with db_manager.get_db_session() as db:
            # Bookmakers
            bookmakers = db.execute(text(
                "SELECT id, name FROM lu_bookmakers"
            )).fetchall()
            self.lookup_cache['bookmaker'] = {name: id for id, name in bookmakers}
            
            # Sources
            sources = db.execute(text(
                "SELECT id, name FROM lu_sources"
            )).fetchall()
            self.lookup_cache['source'] = {name: id for id, name in sources}
            
            # Market types
            market_types = db.execute(text(
                "SELECT id, name FROM lu_market_types"
            )).fetchall()
            self.lookup_cache['market_type'] = {name: id for id, name in market_types}
            
            # Add any missing values dynamically
            self.next_ids = {
                'bookmaker': max(self.lookup_cache['bookmaker'].values()) + 1 if self.lookup_cache['bookmaker'] else 0,
                'source': max(self.lookup_cache['source'].values()) + 1 if self.lookup_cache['source'] else 0,
                'market_type': max(self.lookup_cache['market_type'].values()) + 1 if self.lookup_cache['market_type'] else 0,
            }
            
        logger.info(f"Loaded lookups - Bookmakers: {len(self.lookup_cache['bookmaker'])}, "
                   f"Sources: {len(self.lookup_cache['source'])}, "
                   f"Market Types: {len(self.lookup_cache['market_type'])}")
    
    def _get_or_create_lookup_id(self, table, value):
        """Get lookup ID, creating if necessary."""
        if value in self.lookup_cache[table]:
            return self.lookup_cache[table][value]
        
        # Add new value
        new_id = self.next_ids[table]
        if new_id < 255:  # TINYINT limit
            self.lookup_cache[table][value] = new_id
            self.next_ids[table] += 1
            
            # Insert into database
            with db_manager.get_db_session() as db:
                db.execute(text(f"""
                    INSERT OR IGNORE INTO lu_{table}s (id, name) 
                    VALUES (:id, :name)
                """), {"id": new_id, "name": value})
                db.commit()
            
            return new_id
        else:
            logger.warning(f"Lookup table lu_{table}s full, skipping value: {value}")
            return None
    
    def _check_resources(self):
        """Check if we have enough resources to continue."""
        # Memory check
        memory = psutil.virtual_memory()
        memory_used_gb = (memory.total - memory.available) / (1024**3)
        
        if memory_used_gb > self.memory_limit_gb:
            logger.warning(f"Memory usage high: {memory_used_gb:.1f}GB / {self.memory_limit_gb}GB limit")
            return False
            
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > 90:
            logger.warning(f"CPU usage high: {cpu_percent}%")
            time.sleep(1)  # Brief pause
            
        return True
    
    def _migrate_chunk(self, offset, limit, date_range):
        """Migrate a single chunk of data."""
        start_date, end_date = date_range
        
        # Create separate connection for this thread
        engine = create_engine(f"sqlite:///{db_manager.database_path}", poolclass=NullPool)
        
        with engine.begin() as conn:
            # Fetch chunk
            records = conn.execute(text("""
                SELECT 
                    source_id,
                    bookmaker,
                    source,
                    market_type,
                    outcome,
                    COALESCE(position, 0) as position,
                    line,
                    decimal_odds,
                    american_odds,
                    normalized_implied,
                    updated_at
                FROM odd
                WHERE updated_at >= :start AND updated_at < :end
                ORDER BY updated_at
                LIMIT :limit OFFSET :offset
            """), {
                "start": start_date,
                "end": end_date,
                "limit": limit,
                "offset": offset
            }).fetchall()
            
            if not records:
                return 0, 0
            
            # Transform records
            normalized_records = []
            skipped = 0
            
            for record in records:
                try:
                    # Get lookup IDs
                    bookmaker_id = self._get_or_create_lookup_id('bookmaker', record[1])
                    source_id = self._get_or_create_lookup_id('source', record[2])
                    market_type_id = self._get_or_create_lookup_id('market_type', record[3])
                    
                    if bookmaker_id is None or source_id is None or market_type_id is None:
                        skipped += 1
                        continue
                    
                    # Map outcome
                    outcome_id = {'option_1': 0, 'option_2': 1, 'option_3': 2}.get(record[4])
                    if outcome_id is None:
                        skipped += 1
                        continue
                    
                    normalized_records.append({
                        'market_id': record[0],
                        'bookmaker_id': bookmaker_id,
                        'source_id': source_id,
                        'market_type_id': market_type_id,
                        'outcome_id': outcome_id,
                        'position': record[5],
                        'line_x100': int(record[6] * 100) if record[6] else None,
                        'decimal_odds_x1000': int(record[7] * 1000) if record[7] else 0,
                        'american_odds': int(record[8]) if record[8] else None,
                        'implied_x10000': int(record[9] * 10000) if record[9] else None,
                        'updated_at': int(record[10].timestamp())
                    })
                except Exception as e:
                    skipped += 1
                    continue
            
            # Bulk insert using raw SQL for speed
            if normalized_records:
                # Build multi-row insert
                values_list = []
                params = {}
                
                for i, rec in enumerate(normalized_records):
                    values_list.append(f"""(
                        :market_id{i}, :bookmaker_id{i}, :source_id{i}, 
                        :market_type_id{i}, :outcome_id{i}, :position{i},
                        :line_x100_{i}, :decimal_odds_x1000_{i}, 
                        :american_odds{i}, :implied_x10000_{i}, :updated_at{i}
                    )""")
                    
                    for key, value in rec.items():
                        params[f"{key}{i}"] = value
                
                # Execute multi-row insert
                conn.execute(text(f"""
                    INSERT OR IGNORE INTO odds_normalized
                    (market_id, bookmaker_id, source_id, market_type_id, 
                     outcome_id, position, line_x100, decimal_odds_x1000,
                     american_odds, implied_x10000, updated_at)
                    VALUES {','.join(values_list)}
                """), params)
            
            return len(normalized_records), skipped
    
    def migrate_date_range(self, start_date, end_date):
        """Migrate a date range using parallel processing."""
        logger.info(f"Migrating {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
        
        # Count total records
        with db_manager.get_db_session() as db:
            total_count = db.execute(text("""
                SELECT COUNT(*) FROM odd 
                WHERE updated_at >= :start AND updated_at < :end
            """), {"start": start_date, "end": end_date}).scalar()
            
        if total_count == 0:
            logger.info("No records in this range")
            return
            
        logger.info(f"Processing {total_count:,} records with {self.max_workers} workers")
        
        # Process in parallel chunks
        migrated_total = 0
        skipped_total = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks
            futures = []
            for offset in range(0, total_count, self.chunk_size):
                if not self._check_resources():
                    logger.warning("Resource limit reached, waiting...")
                    time.sleep(5)
                    
                future = executor.submit(
                    self._migrate_chunk, 
                    offset, 
                    self.chunk_size,
                    (start_date, end_date)
                )
                futures.append(future)
            
            # Process results as they complete
            for i, future in enumerate(as_completed(futures)):
                try:
                    migrated, skipped = future.result()
                    migrated_total += migrated
                    skipped_total += skipped
                    
                    # Progress update every 10 chunks
                    if i % 10 == 0:
                        progress = (i + 1) / len(futures) * 100
                        elapsed = time.time() - self.stats['start_time']
                        rate = migrated_total / elapsed if elapsed > 0 else 0
                        
                        logger.info(f"Progress: {progress:.1f}% | "
                                  f"Migrated: {migrated_total:,} | "
                                  f"Rate: {rate:,.0f} records/sec | "
                                  f"Memory: {psutil.virtual_memory().percent}%")
                        
                except Exception as e:
                    logger.error(f"Chunk failed: {e}")
                    
        self.stats['total_processed'] += total_count
        self.stats['total_migrated'] += migrated_total
        
        logger.info(f"Range complete - Migrated: {migrated_total:,}, Skipped: {skipped_total:,}")
    
    def run_full_migration(self):
        """Run the complete migration."""
        logger.info("Starting optimized migration...")
        
        # Get date range
        with db_manager.get_db_session() as db:
            # Check progress
            last_migrated = db.execute(text(
                "SELECT MAX(updated_at) FROM odds_normalized"
            )).scalar()
            
            if last_migrated:
                min_date = datetime.fromtimestamp(last_migrated + 1, tz=timezone.utc)
                logger.info(f"Resuming from {min_date}")
            else:
                min_result = db.execute(text(
                    "SELECT MIN(updated_at) FROM odd WHERE updated_at IS NOT NULL"
                )).scalar()
                min_date = min_result if min_result else datetime(2020, 1, 1, tzinfo=timezone.utc)
            
            max_result = db.execute(text(
                "SELECT MAX(updated_at) FROM odd WHERE updated_at IS NOT NULL"
            )).scalar()
            max_date = max_result if max_result else datetime.now(timezone.utc)
        
        logger.info(f"Migration range: {min_date} to {max_date}")
        
        # Process in hourly batches
        current_date = min_date
        while current_date < max_date:
            batch_end = min(current_date + timedelta(hours=1), max_date)
            
            self.migrate_date_range(current_date, batch_end)
            
            current_date = batch_end
            
            # Overall progress
            progress = (current_date - min_date) / (max_date - min_date) * 100
            elapsed = time.time() - self.stats['start_time']
            
            logger.info(f"\nOVERALL PROGRESS: {progress:.1f}%")
            logger.info(f"Total migrated: {self.stats['total_migrated']:,}")
            logger.info(f"Elapsed time: {elapsed/3600:.1f} hours")
            logger.info(f"Average rate: {self.stats['total_migrated']/elapsed:,.0f} records/sec\n")
        
        # Final report
        logger.info("="*80)
        logger.info("MIGRATION COMPLETE!")
        logger.info(f"Total records processed: {self.stats['total_processed']:,}")
        logger.info(f"Total records migrated: {self.stats['total_migrated']:,}")
        logger.info(f"Total time: {elapsed/3600:.1f} hours")
        logger.info(f"Average rate: {self.stats['total_migrated']/elapsed:,.0f} records/sec")
        
        # Analyze space savings
        self._analyze_results()
    
    def _analyze_results(self):
        """Analyze the migration results."""
        logger.info("\nAnalyzing results...")
        
        with db_manager.get_db_session() as db:
            # Count normalized records
            norm_count = db.execute(text(
                "SELECT COUNT(*) FROM odds_normalized"
            )).scalar()
            
            # Estimate space savings
            original_size = self.stats['total_processed'] * 186  # bytes per record
            new_size = norm_count * 85  # optimized size
            saved = (original_size - new_size) / (1024**3)
            
            logger.info(f"Records in normalized table: {norm_count:,}")
            logger.info(f"Estimated space saved: {saved:.1f} GB")
            logger.info(f"Reduction: {(1 - new_size/original_size)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='High-performance odds migration')
    parser.add_argument('--chunk-size', type=int, default=500000,
                        help='Records per chunk (default: 500k)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: auto)')
    parser.add_argument('--memory-limit', type=float, default=4.0,
                        help='Memory limit in GB (default: 4)')
    args = parser.parse_args()
    
    migrator = OptimizedOddsMigrator(
        chunk_size=args.chunk_size,
        max_workers=args.workers,
        memory_limit_gb=args.memory_limit
    )
    
    migrator.run_full_migration()


if __name__ == "__main__":
    main()