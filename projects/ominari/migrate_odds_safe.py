#!/usr/bin/env python3
"""Safe, memory-efficient migration script for odds normalization."""

import logging
import time
import gc
from datetime import datetime, timedelta, timezone
from sqlalchemy import text, create_engine
from sqlalchemy.pool import NullPool
import argparse
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SafeOddsMigrator:
    """Memory-safe odds migrator that processes in small chunks."""
    
    def __init__(self, chunk_size=50000, batch_hours=1):
        self.chunk_size = chunk_size  # Smaller chunks to avoid memory issues
        self.batch_hours = batch_hours
        self.lookup_cache = {}
        self.stats = {
            'total_processed': 0,
            'total_migrated': 0,
            'total_skipped': 0,
            'start_time': time.time()
        }
        self.db_path = 'sport_odds.db'
        self._load_lookups()
        
    def _load_lookups(self):
        """Load lookup tables into memory."""
        logger.info("Loading lookup tables...")
        
        # Use direct connection to avoid session overhead
        engine = create_engine(f"sqlite:///{self.db_path}", poolclass=NullPool)
        
        with engine.connect() as conn:
            # Load lookups
            bookmakers = conn.execute(text("SELECT id, name FROM lu_bookmakers")).fetchall()
            self.lookup_cache['bookmaker'] = {name: id for id, name in bookmakers}
            
            sources = conn.execute(text("SELECT id, name FROM lu_sources")).fetchall()
            self.lookup_cache['source'] = {name: id for id, name in sources}
            
            market_types = conn.execute(text("SELECT id, name FROM lu_market_types")).fetchall()
            self.lookup_cache['market_type'] = {name: id for id, name in market_types}
            
        engine.dispose()
        
        logger.info(f"Loaded {len(self.lookup_cache['bookmaker'])} bookmakers, "
                   f"{len(self.lookup_cache['source'])} sources, "
                   f"{len(self.lookup_cache['market_type'])} market types")
    
    def _check_memory(self):
        """Check available memory and pause if needed."""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb < 1.0:  # Less than 1GB available
            logger.warning(f"Low memory: {available_gb:.2f}GB available. Pausing...")
            gc.collect()
            time.sleep(10)
            return False
        return True
    
    def _migrate_chunk(self, start_date, end_date, offset):
        """Migrate a single chunk of data."""
        # Create new connection for this chunk
        engine = create_engine(f"sqlite:///{self.db_path}", poolclass=NullPool)
        
        try:
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
                    "limit": self.chunk_size,
                    "offset": offset
                }).fetchall()
                
                if not records:
                    return 0, 0
                
                # Process records one by one to minimize memory
                migrated = 0
                skipped = 0
                
                # Build batch insert values
                values_sql = []
                params = {}
                param_index = 0
                
                for record in records:
                    try:
                        # Get lookup IDs
                        bookmaker_id = self.lookup_cache['bookmaker'].get(record[1])
                        source_id = self.lookup_cache['source'].get(record[2])
                        market_type_id = self.lookup_cache['market_type'].get(record[3])
                        
                        if bookmaker_id is None or source_id is None or market_type_id is None:
                            skipped += 1
                            continue
                        
                        # Map outcome
                        outcome_id = {'option_1': 0, 'option_2': 1, 'option_3': 2}.get(record[4])
                        if outcome_id is None:
                            skipped += 1
                            continue
                        
                        # Add to batch
                        values_sql.append(f"""(
                            :market_id{param_index}, :bookmaker_id{param_index}, :source_id{param_index},
                            :market_type_id{param_index}, :outcome_id{param_index}, :position{param_index},
                            :line_x100_{param_index}, :decimal_odds_x1000_{param_index},
                            :american_odds{param_index}, :implied_x10000_{param_index}, :updated_at{param_index}
                        )""")
                        
                        params[f'market_id{param_index}'] = record[0]
                        params[f'bookmaker_id{param_index}'] = bookmaker_id
                        params[f'source_id{param_index}'] = source_id
                        params[f'market_type_id{param_index}'] = market_type_id
                        params[f'outcome_id{param_index}'] = outcome_id
                        params[f'position{param_index}'] = record[5]
                        params[f'line_x100_{param_index}'] = int(record[6] * 100) if record[6] else None
                        params[f'decimal_odds_x1000_{param_index}'] = int(record[7] * 1000) if record[7] else 0
                        params[f'american_odds{param_index}'] = int(record[8]) if record[8] else None
                        params[f'implied_x10000_{param_index}'] = int(record[9] * 10000) if record[9] else None
                        params[f'updated_at{param_index}'] = int(record[10].timestamp())
                        
                        param_index += 1
                        migrated += 1
                        
                        # Insert in batches of 1000 to avoid SQL length limits
                        if param_index >= 1000:
                            conn.execute(text(f"""
                                INSERT OR IGNORE INTO odds_normalized
                                (market_id, bookmaker_id, source_id, market_type_id,
                                 outcome_id, position, line_x100, decimal_odds_x1000,
                                 american_odds, implied_x10000, updated_at)
                                VALUES {','.join(values_sql)}
                            """), params)
                            
                            values_sql = []
                            params = {}
                            param_index = 0
                            
                    except Exception as e:
                        skipped += 1
                        continue
                
                # Insert remaining records
                if values_sql:
                    conn.execute(text(f"""
                        INSERT OR IGNORE INTO odds_normalized
                        (market_id, bookmaker_id, source_id, market_type_id,
                         outcome_id, position, line_x100, decimal_odds_x1000,
                         american_odds, implied_x10000, updated_at)
                        VALUES {','.join(values_sql)}
                    """), params)
                
                return migrated, skipped
                
        finally:
            engine.dispose()
            gc.collect()  # Force garbage collection
    
    def migrate_date_range(self, start_date, end_date):
        """Migrate a date range in chunks."""
        logger.info(f"Migrating {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
        
        # Count total records
        engine = create_engine(f"sqlite:///{self.db_path}", poolclass=NullPool)
        with engine.connect() as conn:
            total_count = conn.execute(text("""
                SELECT COUNT(*) FROM odd 
                WHERE updated_at >= :start AND updated_at < :end
            """), {"start": start_date, "end": end_date}).scalar()
        engine.dispose()
        
        if total_count == 0:
            logger.info("No records in this range")
            return
        
        logger.info(f"Processing {total_count:,} records in chunks of {self.chunk_size:,}")
        
        # Process chunks sequentially to avoid memory issues
        offset = 0
        range_migrated = 0
        range_skipped = 0
        
        while offset < total_count:
            if not self._check_memory():
                continue
                
            chunk_start = time.time()
            migrated, skipped = self._migrate_chunk(start_date, end_date, offset)
            chunk_time = time.time() - chunk_start
            
            range_migrated += migrated
            range_skipped += skipped
            offset += self.chunk_size
            
            # Progress update
            progress = min(offset / total_count * 100, 100)
            rate = migrated / chunk_time if chunk_time > 0 else 0
            
            logger.info(f"Chunk progress: {progress:.1f}% | "
                       f"Migrated: {migrated:,} | "
                       f"Rate: {rate:,.0f} records/sec | "
                       f"Memory: {psutil.virtual_memory().percent}%")
        
        self.stats['total_processed'] += total_count
        self.stats['total_migrated'] += range_migrated
        self.stats['total_skipped'] += range_skipped
        
        logger.info(f"Range complete - Migrated: {range_migrated:,}, Skipped: {range_skipped:,}")
    
    def run_migration(self):
        """Run the complete migration."""
        logger.info("Starting safe migration...")
        
        # Get date range
        engine = create_engine(f"sqlite:///{self.db_path}", poolclass=NullPool)
        with engine.connect() as conn:
            # Check if resuming
            last_migrated = conn.execute(text(
                "SELECT MAX(updated_at) FROM odds_normalized"
            )).scalar()
            
            if last_migrated:
                min_date = datetime.fromtimestamp(last_migrated + 1, tz=timezone.utc)
                logger.info(f"Resuming from {min_date}")
            else:
                min_result = conn.execute(text(
                    "SELECT MIN(updated_at) FROM odd WHERE updated_at IS NOT NULL"
                )).scalar()
                min_date = min_result if min_result else datetime(2020, 1, 1, tzinfo=timezone.utc)
            
            max_result = conn.execute(text(
                "SELECT MAX(updated_at) FROM odd WHERE updated_at IS NOT NULL"
            )).scalar()
            max_date = max_result if max_result else datetime.now(timezone.utc)
        engine.dispose()
        
        logger.info(f"Migration range: {min_date} to {max_date}")
        
        # Process in batches
        current_date = min_date
        while current_date < max_date:
            batch_end = min(current_date + timedelta(hours=self.batch_hours), max_date)
            
            self.migrate_date_range(current_date, batch_end)
            
            current_date = batch_end
            
            # Overall progress
            progress = (current_date - min_date) / (max_date - min_date) * 100
            elapsed = time.time() - self.stats['start_time']
            overall_rate = self.stats['total_migrated'] / elapsed if elapsed > 0 else 0
            
            logger.info(f"\nOVERALL PROGRESS: {progress:.1f}%")
            logger.info(f"Total migrated: {self.stats['total_migrated']:,}")
            logger.info(f"Total skipped: {self.stats['total_skipped']:,}")
            logger.info(f"Elapsed time: {elapsed/3600:.1f} hours")
            logger.info(f"Average rate: {overall_rate:,.0f} records/sec")
            
            # Estimate time remaining
            if overall_rate > 0:
                records_left = self.stats['total_processed'] / progress * (100 - progress) if progress > 0 else 0
                eta_hours = records_left / overall_rate / 3600
                logger.info(f"Estimated time remaining: {eta_hours:.1f} hours\n")
        
        logger.info("="*80)
        logger.info("MIGRATION COMPLETE!")
        logger.info(f"Total processed: {self.stats['total_processed']:,}")
        logger.info(f"Total migrated: {self.stats['total_migrated']:,}")
        logger.info(f"Total skipped: {self.stats['total_skipped']:,}")
        logger.info(f"Success rate: {self.stats['total_migrated']/self.stats['total_processed']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Safe memory-efficient odds migration')
    parser.add_argument('--chunk-size', type=int, default=50000,
                        help='Records per chunk (default: 50k)')
    parser.add_argument('--batch-hours', type=int, default=1,
                        help='Hours per batch (default: 1)')
    args = parser.parse_args()
    
    migrator = SafeOddsMigrator(
        chunk_size=args.chunk_size,
        batch_hours=args.batch_hours
    )
    
    migrator.run_migration()


if __name__ == "__main__":
    main()