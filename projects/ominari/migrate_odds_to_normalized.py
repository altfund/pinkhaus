#!/usr/bin/env python3
"""Migrate odds data to normalized schema."""

import logging
import time
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
from database_v2 import db_manager
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OddsMigrator:
    """Migrate odds to normalized schema."""
    
    def __init__(self, batch_days=1, start_date=None):
        self.batch_days = batch_days
        self.start_date = start_date
        self.lookup_cache = {}
        self._load_lookups()
        
    def _load_lookups(self):
        """Load lookup tables into memory."""
        logger.info("Loading lookup tables...")
        
        with db_manager.get_db_session() as db:
            # Load bookmakers
            bookmakers = db.execute(text(
                "SELECT id, name FROM lu_bookmakers"
            )).fetchall()
            self.lookup_cache['bookmaker'] = {name: id for id, name in bookmakers}
            
            # Load sources
            sources = db.execute(text(
                "SELECT id, name FROM lu_sources"
            )).fetchall()
            self.lookup_cache['source'] = {name: id for id, name in sources}
            
            # Load market types
            market_types = db.execute(text(
                "SELECT id, name FROM lu_market_types"
            )).fetchall()
            self.lookup_cache['market_type'] = {name: id for id, name in market_types}
            
        logger.info(f"Loaded {len(self.lookup_cache['bookmaker'])} bookmakers")
        logger.info(f"Loaded {len(self.lookup_cache['source'])} sources")
        logger.info(f"Loaded {len(self.lookup_cache['market_type'])} market types")
    
    def get_migration_bounds(self):
        """Get date bounds for migration."""
        with db_manager.get_db_session() as db:
            if self.start_date:
                min_date = self.start_date
            else:
                # Check if we have any migrated data
                result = db.execute(text(
                    "SELECT MAX(updated_at) FROM odds_normalized"
                )).scalar()
                
                if result:
                    # Resume from last migrated
                    min_date = datetime.fromtimestamp(result, tz=timezone.utc)
                    logger.info(f"Resuming from {min_date}")
                else:
                    # Start from beginning
                    result = db.execute(text(
                        "SELECT MIN(updated_at) FROM odd WHERE updated_at IS NOT NULL"
                    )).scalar()
                    min_date = result if result else datetime.now(timezone.utc) - timedelta(days=365)
            
            # Get max date
            result = db.execute(text(
                "SELECT MAX(updated_at) FROM odd WHERE updated_at IS NOT NULL"
            )).scalar()
            max_date = result if result else datetime.now(timezone.utc)
            
        return min_date, max_date
    
    def migrate_batch(self, start_date, end_date):
        """Migrate one batch of data."""
        logger.info(f"Migrating {start_date.date()} to {end_date.date()}...")
        
        start_time = time.time()
        
        with db_manager.get_db_session() as db:
            # Count records in batch
            count = db.execute(text("""
                SELECT COUNT(*) FROM odd 
                WHERE updated_at >= :start AND updated_at < :end
            """), {
                "start": start_date,
                "end": end_date
            }).scalar()
            
            if count == 0:
                logger.info("No records in this batch")
                return 0
            
            logger.info(f"Processing {count:,} records...")
            
            # Migrate using INSERT SELECT with manual mapping
            inserted = 0
            chunk_size = 100000
            
            for offset in range(0, count, chunk_size):
                try:
                    # Get chunk of data
                    records = db.execute(text("""
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
                        "limit": chunk_size,
                        "offset": offset
                    }).fetchall()
                    
                    # Transform and insert
                    normalized_records = []
                    for record in records:
                        try:
                            # Map values using cache
                            bookmaker_id = self.lookup_cache['bookmaker'].get(record[1])
                            source_id = self.lookup_cache['source'].get(record[2])
                            market_type_id = self.lookup_cache['market_type'].get(record[3])
                            
                            if bookmaker_id is None or source_id is None or market_type_id is None:
                                continue
                            
                            # Map outcome
                            outcome_id = {'option_1': 0, 'option_2': 1, 'option_3': 2}.get(record[4])
                            if outcome_id is None:
                                continue
                            
                            normalized_records.append({
                                'market_id': record[0],
                                'bookmaker_id': bookmaker_id,
                                'source_id': source_id,
                                'market_type_id': market_type_id,
                                'outcome_id': outcome_id,
                                'position': record[5] if record[5] is not None else 0,
                                'line_x100': int(record[6] * 100) if record[6] else None,
                                'decimal_odds_x1000': int(record[7] * 1000) if record[7] else 0,
                                'american_odds': int(record[8]) if record[8] else None,
                                'implied_x10000': int(record[9] * 10000) if record[9] else None,
                                'updated_at': int(record[10].timestamp())
                            })
                        except Exception as e:
                            # Skip bad records
                            continue
                    
                    # Batch insert
                    if normalized_records:
                        db.execute(text("""
                            INSERT OR IGNORE INTO odds_normalized
                            (market_id, bookmaker_id, source_id, market_type_id, 
                             outcome_id, position, line_x100, decimal_odds_x1000,
                             american_odds, implied_x10000, updated_at)
                            VALUES
                            (:market_id, :bookmaker_id, :source_id, :market_type_id,
                             :outcome_id, :position, :line_x100, :decimal_odds_x1000,
                             :american_odds, :implied_x10000, :updated_at)
                        """), normalized_records)
                        
                        inserted += len(normalized_records)
                    
                    if offset % 500000 == 0 and offset > 0:
                        logger.info(f"  Progress: {offset:,}/{count:,} ({offset/count*100:.1f}%)")
                        db.commit()
                        
                except Exception as e:
                    logger.error(f"Error in chunk at offset {offset}: {e}")
                    continue
            
            db.commit()
            
        elapsed = time.time() - start_time
        rate = inserted / elapsed if elapsed > 0 else 0
        logger.info(f"Migrated {inserted:,} records in {elapsed:.1f}s ({rate:.0f} records/sec)")
        
        return inserted
    
    def run_migration(self):
        """Run the full migration."""
        logger.info("Starting odds migration to normalized schema...")
        
        min_date, max_date = self.get_migration_bounds()
        logger.info(f"Migration range: {min_date} to {max_date}")
        
        total_migrated = 0
        current_date = min_date
        
        while current_date < max_date:
            batch_end = min(current_date + timedelta(days=self.batch_days), max_date)
            
            migrated = self.migrate_batch(current_date, batch_end)
            total_migrated += migrated
            
            current_date = batch_end
            
            # Show progress
            progress = (current_date - min_date) / (max_date - min_date) * 100
            logger.info(f"Overall progress: {progress:.1f}%")
            logger.info(f"Total migrated: {total_migrated:,}")
            logger.info("")
        
        logger.info(f"Migration complete! Total records: {total_migrated:,}")
        
        # Verify migration
        self.verify_migration()
    
    def verify_migration(self):
        """Verify the migration was successful."""
        logger.info("\nVerifying migration...")
        
        with db_manager.get_db_session() as db:
            # Count normalized records
            norm_count = db.execute(text(
                "SELECT COUNT(*) FROM odds_normalized"
            )).scalar()
            
            logger.info(f"Normalized records: {norm_count:,}")
            
            # Sample some data
            samples = db.execute(text("""
                SELECT 
                    o.market_id,
                    b.name as bookmaker,
                    o.decimal_odds_x1000 / 1000.0 as odds,
                    datetime(o.updated_at, 'unixepoch') as updated
                FROM odds_normalized o
                JOIN lu_bookmakers b ON o.bookmaker_id = b.id
                ORDER BY o.updated_at DESC
                LIMIT 5
            """)).fetchall()
            
            logger.info("\nSample migrated records:")
            for market_id, bookmaker, odds, updated in samples:
                logger.info(f"  {market_id[:16]}... {bookmaker}: {odds:.3f} @ {updated}")


def main():
    parser = argparse.ArgumentParser(description='Migrate odds to normalized schema')
    parser.add_argument('--batch-days', type=int, default=1,
                        help='Number of days to process per batch')
    parser.add_argument('--start-date', type=str,
                        help='Start date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    start_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    
    migrator = OddsMigrator(
        batch_days=args.batch_days,
        start_date=start_date
    )
    migrator.run_migration()


if __name__ == "__main__":
    main()