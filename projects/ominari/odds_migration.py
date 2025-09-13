#!/usr/bin/env python3
"""
Odds Migration Script
Migrates the massive odds table (515M+ records) from SQLite to PostgreSQL
"""

import sqlite3
import psycopg2
import logging
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Optional
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OddsMigrator:
    """High-performance odds data migration."""

    def __init__(self, batch_size=100000):
        self.batch_size = batch_size
        self.sqlite_path = 'sport_odds.db'

        self.pg_config = {
            'host': os.getenv('PG_HOST', 'localhost'),
            'port': os.getenv('PG_PORT', '5435'),
            'user': os.getenv('PG_USER', 'ominari_user'),
            'password': os.getenv('PG_PASSWORD', 'ominari_2025_secure'),
            'database': os.getenv('PG_DB', 'ominari_production')
        }

        # Load lookup cache
        self.lookup_cache = {}
        self._load_lookup_cache()

        logger.info(f"Odds migrator initialized: batch_size={batch_size:,}")

    @contextmanager
    def get_sqlite_connection(self):
        """Highly optimized SQLite connection for massive table reading."""
        conn = sqlite3.connect(self.sqlite_path, timeout=300.0)
        conn.row_factory = sqlite3.Row

        # Maximum optimization for very large table
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA cache_size=500000")  # 2GB cache
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=17179869184")  # 16GB memory map
        conn.execute("PRAGMA read_uncommitted=true")

        try:
            yield conn
        finally:
            conn.close()

    def _load_lookup_cache(self):
        """Load lookup tables for odds transformation."""
        logger.info("Loading lookup cache for odds migration...")

        try:
            with psycopg2.connect(**self.pg_config) as conn:
                with conn.cursor() as cursor:
                    # Load essential lookups for odds
                    cursor.execute("SELECT id, name FROM ominari.lu_bookmakers")
                    self.lookup_cache['bookmakers'] = {name: id for id, name in cursor.fetchall()}

                    cursor.execute("SELECT id, name FROM ominari.lu_outcomes")
                    self.lookup_cache['outcomes'] = {name: id for id, name in cursor.fetchall()}

                    cursor.execute("SELECT id, name FROM ominari.lu_sources")
                    self.lookup_cache['sources'] = {name: id for id, name in cursor.fetchall()}

                    # Get existing market external_ids for relationship mapping
                    cursor.execute("SELECT external_id, id FROM ominari.markets_normalized")
                    self.lookup_cache['markets'] = {ext_id: id for ext_id, id in cursor.fetchall()}

            logger.info(f"Loaded lookups: {len(self.lookup_cache['bookmakers'])} bookmakers, "
                       f"{len(self.lookup_cache['markets'])} markets")

        except Exception as e:
            logger.error(f"Failed to load lookup cache: {e}")
            raise

    def get_odds_statistics(self):
        """Analyze odds table structure and size."""
        logger.info("Analyzing odds table...")

        try:
            with self.get_sqlite_connection() as conn:
                cursor = conn.cursor()

                # Try to get count (may timeout)
                try:
                    cursor.execute("SELECT MAX(rowid) FROM odd")
                    max_rowid = cursor.fetchone()[0]
                    logger.info(f"Max odds rowid: {max_rowid:,}")
                except Exception:
                    max_rowid = 515919464  # From previous analysis

                # Get sample data to understand structure
                cursor.execute("""
                    SELECT source_id, bookmaker, outcome, decimal_odds,
                           american_odds, normalized_implied, updated_at
                    FROM odd
                    WHERE rowid <= 10
                    LIMIT 5
                """)

                samples = cursor.fetchall()
                logger.info("Sample odds records:")
                for sample in samples:
                    logger.info(f"  {sample['source_id']}: {sample['bookmaker']} - {sample['outcome']} @ {sample['decimal_odds']}")

                return {'max_rowid': max_rowid, 'estimated_count': max_rowid}

        except Exception as e:
            logger.error(f"Failed to analyze odds: {e}")
            return {'max_rowid': 515919464, 'estimated_count': 515919464}

    def migrate_odds_batch(self, start_rowid: int, limit: int) -> int:
        """Migrate a batch of odds records."""
        try:
            with self.get_sqlite_connection() as sqlite_conn:
                with psycopg2.connect(**self.pg_config) as pg_conn:
                    sqlite_cursor = sqlite_conn.cursor()
                    pg_cursor = pg_conn.cursor()

                    # Get batch of odds
                    sqlite_cursor.execute("""
                        SELECT
                            source_id,
                            bookmaker,
                            outcome,
                            decimal_odds,
                            american_odds,
                            normalized_implied,
                            updated_at,
                            rowid
                        FROM odd
                        WHERE rowid >= ? AND rowid < ?
                        ORDER BY rowid
                    """, (start_rowid, start_rowid + limit))

                    odds_to_insert = []

                    for row in sqlite_cursor:
                        odds_data = self._transform_odd(row)
                        if odds_data:
                            odds_to_insert.append(odds_data)

                    if odds_to_insert:
                        # Insert into partitioned odds table
                        pg_cursor.executemany("""
                            INSERT INTO ominari.odds_normalized
                            (market_id, outcome_id, bookmaker_id, decimal_odds_x1000,
                             american_odds, implied_prob_x10000, updated_at_ts)
                            VALUES (%(market_id)s, %(outcome_id)s, %(bookmaker_id)s,
                                   %(decimal_odds_x1000)s, %(american_odds)s,
                                   %(implied_prob_x10000)s, %(updated_at_ts)s)
                            ON CONFLICT DO NOTHING
                        """, odds_to_insert)

                        pg_conn.commit()

                    return len(odds_to_insert)

        except Exception as e:
            logger.error(f"Odds batch migration failed at rowid {start_rowid}: {e}")
            return 0

    def _transform_odd(self, row) -> Optional[Dict]:
        """Transform SQLite odds row to normalized PostgreSQL format."""
        try:
            # Map to market ID
            market_id = self.lookup_cache['markets'].get(row['source_id'])
            if not market_id:
                return None  # Skip odds for non-migrated markets

            # Map lookups with fallbacks
            bookmaker_id = self.lookup_cache['bookmakers'].get(row['bookmaker'] or 'overtime_markets', 1)
            outcome_id = self.lookup_cache['outcomes'].get(row['outcome'] or 'option_1', 1)

            # Convert to normalized format
            decimal_odds_x1000 = int(float(row['decimal_odds'] or 0) * 1000) if row['decimal_odds'] else None
            american_odds = int(row['american_odds']) if row['american_odds'] else None
            implied_prob_x10000 = int(float(row['normalized_implied'] or 0) * 10000) if row['normalized_implied'] else None

            # Parse timestamp
            updated_at_ts = self._parse_timestamp(row['updated_at'])

            return {
                'market_id': market_id,
                'outcome_id': outcome_id,
                'bookmaker_id': bookmaker_id,
                'decimal_odds_x1000': decimal_odds_x1000,
                'american_odds': american_odds,
                'implied_prob_x10000': implied_prob_x10000,
                'updated_at_ts': updated_at_ts
            }

        except Exception as e:
            logger.debug(f"Odds transform failed for rowid {row.get('rowid', 'unknown')}: {e}")
            return None

    def _parse_timestamp(self, timestamp_str: str) -> Optional[int]:
        """Parse timestamp to Unix timestamp."""
        if not timestamp_str:
            return None

        try:
            if isinstance(timestamp_str, str):
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                return int(dt.timestamp())
            return None
        except Exception:
            return None

    def migrate_all_odds(self, target_records: int = None):
        """Migrate all odds records in batches."""
        logger.info("🚀 Starting massive odds migration...")

        stats = self.get_odds_statistics()
        max_rowid = stats['max_rowid']

        if target_records is None:
            target_records = max_rowid

        logger.info(f"Target odds migration: {target_records:,} records from max_rowid: {max_rowid:,}")

        total_migrated = 0
        current_rowid = 1
        batch_num = 1
        start_time = time.time()
        consecutive_empty = 0

        while current_rowid < max_rowid and total_migrated < target_records and consecutive_empty < 5:
            batch_start = time.time()
            remaining = target_records - total_migrated
            current_batch_size = min(self.batch_size, remaining)

            logger.info(f"📦 Odds Batch {batch_num}: Processing {current_batch_size:,} (rowid {current_rowid:,})")

            batch_count = self.migrate_odds_batch(current_rowid, current_batch_size)

            if batch_count == 0:
                consecutive_empty += 1
                logger.warning(f"Empty batch {consecutive_empty}/5")
            else:
                consecutive_empty = 0
                total_migrated += batch_count

            current_rowid += current_batch_size

            # Progress reporting
            batch_time = time.time() - batch_start
            elapsed = time.time() - start_time
            rate = total_migrated / elapsed if elapsed > 0 else 0
            progress = (current_rowid / max_rowid) * 100

            logger.info(f"📊 Odds Progress: {total_migrated:,} migrated, {progress:.1f}% scanned, "
                       f"{rate:.0f} records/sec")

            batch_num += 1
            time.sleep(0.2)  # Brief pause

            # Status check every 10 batches
            if batch_num % 10 == 0:
                self._verify_odds_migration()

        total_elapsed = time.time() - start_time
        final_rate = total_migrated / total_elapsed if total_elapsed > 0 else 0

        logger.info(f"🎯 Odds migration completed: {total_migrated:,} records in {total_elapsed:.2f}s "
                   f"(avg {final_rate:.0f} records/sec)")

        return total_migrated

    def _verify_odds_migration(self):
        """Quick verification of odds migration progress."""
        try:
            with psycopg2.connect(**self.pg_config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM ominari.odds_normalized")
                    count = cursor.fetchone()[0]
                    logger.info(f"✅ Current odds in PostgreSQL: {count:,}")
        except Exception as e:
            logger.warning(f"Verification check failed: {e}")


def main():
    """Execute odds migration."""
    migrator = OddsMigrator(batch_size=150000)  # Larger batch for odds

    # Start with moderate target - can increase later
    migrated_count = migrator.migrate_all_odds(target_records=5000000)  # 5M odds first

    if migrated_count > 0:
        logger.info(f"✅ Odds migration completed: {migrated_count:,} records")
    else:
        logger.error("❌ Odds migration failed")


if __name__ == "__main__":
    main()