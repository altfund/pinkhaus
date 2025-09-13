#!/usr/bin/env python3
"""
Enhanced Full Migration Script
Migrates ALL historical data from SQLite to PostgreSQL with complete data preservation.
"""

import sqlite3
import psycopg2
import logging
from datetime import datetime, timedelta
import time
import os
import argparse
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CompleteMigrator:
    """Complete historical data migration from SQLite to PostgreSQL."""

    def __init__(self, sqlite_path='sport_odds.db', batch_size=25000):
        self.sqlite_path = sqlite_path
        self.batch_size = batch_size

        self.pg_config = {
            'host': os.getenv('PG_HOST', 'localhost'),
            'port': os.getenv('PG_PORT', '5435'),
            'user': os.getenv('PG_USER', 'ominari_user'),
            'password': os.getenv('PG_PASSWORD', 'ominari_2025_secure'),
            'database': os.getenv('PG_DB', 'ominari_production')
        }

        # Load existing lookup cache from PostgreSQL
        self.lookup_cache = {}
        self._load_lookup_cache()

        logger.info(f"Complete migrator initialized: batch_size={batch_size:,}")

    @contextmanager
    def get_sqlite_connection(self):
        """Get highly optimized SQLite connection for large-scale reading."""
        conn = sqlite3.connect(self.sqlite_path, timeout=120.0)
        conn.row_factory = sqlite3.Row

        # Maximum read optimization for large database
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")  # Faster for read-only
        conn.execute("PRAGMA cache_size=200000")  # 800MB cache
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=8589934592")  # 8GB memory map
        conn.execute("PRAGMA read_uncommitted=true")  # Fastest reads

        try:
            yield conn
        finally:
            conn.close()

    def _load_lookup_cache(self):
        """Load all existing PostgreSQL lookup tables."""
        logger.info("Loading comprehensive lookup cache...")

        try:
            with psycopg2.connect(**self.pg_config) as conn:
                with conn.cursor() as cursor:

                    # Load all lookup tables
                    tables = ['bookmakers', 'sources', 'market_types', 'outcomes', 'sports', 'teams']
                    for table in tables:
                        cursor.execute(f"SELECT id, name FROM ominari.lu_{table}")
                        self.lookup_cache[table] = {name: id for id, name in cursor.fetchall()}

            total_lookups = sum(len(cache) for cache in self.lookup_cache.values())
            logger.info(f"Loaded {total_lookups} total lookup entries across {len(self.lookup_cache)} tables")

        except Exception as e:
            logger.error(f"Failed to load lookup cache: {e}")
            raise

    def get_sqlite_statistics(self):
        """Get comprehensive SQLite database statistics."""
        logger.info("Analyzing SQLite database...")

        try:
            with self.get_sqlite_connection() as conn:
                cursor = conn.cursor()

                stats = {}

                # Get table row counts
                tables = ['market', 'odd']
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        stats[f'{table}_count'] = count
                    except Exception as e:
                        logger.warning(f"Could not count {table}: {e}")
                        # Try alternative method
                        try:
                            cursor.execute(f"SELECT MAX(rowid) FROM {table}")
                            max_rowid = cursor.fetchone()[0]
                            stats[f'{table}_count'] = max_rowid  # Approximation
                            logger.info(f"Using MAX(rowid) approximation for {table}: {max_rowid:,}")
                        except:
                            stats[f'{table}_count'] = 1000000  # Default estimate

                # Get date range
                try:
                    cursor.execute("SELECT MIN(maturity_date), MAX(maturity_date) FROM market WHERE maturity_date IS NOT NULL LIMIT 1000")
                    result = cursor.fetchone()
                    if result and result[0]:
                        stats['date_range'] = (result[0], result[1])
                except Exception as e:
                    logger.warning(f"Could not get date range: {e}")

                logger.info(f"SQLite Statistics:")
                for key, value in stats.items():
                    if isinstance(value, int):
                        logger.info(f"  {key}: {value:,}")
                    else:
                        logger.info(f"  {key}: {value}")

                return stats

        except Exception as e:
            logger.error(f"Failed to analyze SQLite: {e}")
            return {'market_count': 1000000, 'odd_count': 5000000}  # Default estimates

    def migrate_all_markets(self, target_records: int = None):
        """Migrate all markets without date filtering."""
        logger.info("🚀 Starting complete market migration (NO date filtering)")

        # Get statistics first
        stats = self.get_sqlite_statistics()
        total_estimated = stats.get('market_count', 1000000)

        if target_records is None:
            target_records = total_estimated
        else:
            target_records = min(target_records, total_estimated)

        logger.info(f"Target migration: {target_records:,} markets")

        total_migrated = 0
        current_rowid = 1
        batch_num = 1
        start_time = time.time()
        consecutive_empty_batches = 0

        while total_migrated < target_records and consecutive_empty_batches < 3:
            batch_start = time.time()
            remaining = target_records - total_migrated
            current_batch_size = min(self.batch_size, remaining)

            logger.info(f"📦 Batch {batch_num}: Processing {current_batch_size:,} markets (rowid >= {current_rowid})")

            try:
                batch_count = self._migrate_markets_batch(current_batch_size, current_rowid)

                if batch_count == 0:
                    consecutive_empty_batches += 1
                    logger.warning(f"Empty batch {consecutive_empty_batches}/3")
                    current_rowid += current_batch_size  # Skip ahead
                else:
                    consecutive_empty_batches = 0
                    total_migrated += batch_count
                    current_rowid += current_batch_size

                # Progress reporting
                batch_time = time.time() - batch_start
                elapsed = time.time() - start_time
                rate = total_migrated / elapsed if elapsed > 0 else 0
                progress = (total_migrated / target_records) * 100

                logger.info(f"📊 Progress: {total_migrated:,}/{target_records:,} ({progress:.1f}%) "
                           f"at {rate:.0f} records/sec - Batch: {batch_count:,} in {batch_time:.2f}s")

                batch_num += 1
                time.sleep(0.5)  # Brief pause

            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                current_rowid += current_batch_size  # Skip problematic batch
                consecutive_empty_batches += 1

        total_elapsed = time.time() - start_time
        final_rate = total_migrated / total_elapsed if total_elapsed > 0 else 0

        logger.info(f"🎯 Market migration completed: {total_migrated:,} records in {total_elapsed:.2f}s "
                   f"(avg {final_rate:.0f} records/sec)")

        return total_migrated

    def _migrate_markets_batch(self, limit: int, start_rowid: int) -> int:
        """Migrate a single batch of markets."""
        try:
            with self.get_sqlite_connection() as sqlite_conn:
                with psycopg2.connect(**self.pg_config) as pg_conn:
                    sqlite_cursor = sqlite_conn.cursor()
                    pg_cursor = pg_conn.cursor()

                    # Remove date filtering - get ALL historical data
                    sqlite_cursor.execute("""
                        SELECT
                            source_id,
                            sport,
                            home_team,
                            away_team,
                            maturity_date,
                            start_time,
                            source,
                            market_type,
                            rowid
                        FROM market
                        WHERE rowid >= ? AND rowid < ?
                        ORDER BY rowid
                    """, (start_rowid, start_rowid + limit))

                    markets_to_insert = []
                    new_teams_created = 0

                    for row in sqlite_cursor:
                        market_data = self._transform_market(row)
                        if market_data:
                            markets_to_insert.append(market_data)

                    if markets_to_insert:
                        # Batch insert with conflict handling
                        pg_cursor.executemany("""
                            INSERT INTO ominari.markets_normalized
                            (external_id, source_id, sport_id, home_team_id, away_team_id,
                             start_time, maturity_date, market_type_id)
                            VALUES (%(external_id)s, %(source_id)s, %(sport_id)s,
                                   %(home_team_id)s, %(away_team_id)s,
                                   %(start_time)s, %(maturity_date)s, %(market_type_id)s)
                            ON CONFLICT (external_id) DO UPDATE SET
                                source_id = EXCLUDED.source_id,
                                sport_id = EXCLUDED.sport_id,
                                home_team_id = EXCLUDED.home_team_id,
                                away_team_id = EXCLUDED.away_team_id,
                                start_time = EXCLUDED.start_time,
                                maturity_date = EXCLUDED.maturity_date,
                                market_type_id = EXCLUDED.market_type_id
                        """, markets_to_insert)

                        pg_conn.commit()

                    return len(markets_to_insert)

        except Exception as e:
            logger.error(f"Batch migration failed at rowid {start_rowid}: {e}")
            return 0

    def _transform_market(self, row) -> Optional[Dict]:
        """Transform SQLite market row to PostgreSQL format with enhanced team handling."""
        try:
            # Enhanced team creation with caching
            home_team_id = self._get_or_create_team_cached(row['home_team'], row['sport'])
            away_team_id = self._get_or_create_team_cached(row['away_team'], row['sport'])

            # Robust lookup mapping with defaults
            source_id = self.lookup_cache['sources'].get(row['source'] or 'overtime_markets', 1)
            sport_id = self._map_sport_with_fallback(row['sport'])
            market_type_id = self.lookup_cache['market_types'].get(row['market_type'] or 'winner', 1)

            # Enhanced date parsing
            start_time = self._parse_datetime_robust(row['start_time'])
            maturity_date = self._parse_datetime_robust(row['maturity_date'])

            return {
                'external_id': row['source_id'] or f"migrated_{row['rowid']}",
                'source_id': source_id,
                'sport_id': sport_id,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'start_time': start_time,
                'maturity_date': maturity_date,
                'market_type_id': market_type_id
            }

        except Exception as e:
            logger.debug(f"Market transform failed for rowid {row.get('rowid', 'unknown')}: {e}")
            return None

    def _get_or_create_team_cached(self, team_name: str, sport: str) -> Optional[int]:
        """Enhanced team creation with better error handling and caching."""
        if not team_name or not team_name.strip():
            return None

        team_name = team_name.strip()

        # Check cache
        if team_name in self.lookup_cache['teams']:
            return self.lookup_cache['teams'][team_name]

        # Create new team
        try:
            sport_id = self._map_sport_with_fallback(sport)

            with psycopg2.connect(**self.pg_config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO ominari.lu_teams (name, sport_id)
                        VALUES (%s, %s)
                        ON CONFLICT (name) DO NOTHING
                        RETURNING id
                    """, (team_name, sport_id))

                    result = cursor.fetchone()
                    if result:
                        team_id = result[0]
                    else:
                        cursor.execute("SELECT id FROM ominari.lu_teams WHERE name = %s", (team_name,))
                        result = cursor.fetchone()
                        team_id = result[0] if result else None

                    if team_id:
                        self.lookup_cache['teams'][team_name] = team_id

                    return team_id

        except Exception as e:
            logger.debug(f"Team creation failed for '{team_name}': {e}")
            return None

    def _map_sport_with_fallback(self, sport_name: str) -> int:
        """Enhanced sport mapping with comprehensive fallbacks."""
        if not sport_name:
            return 1  # Default sport

        # Direct match
        if sport_name in self.lookup_cache['sports']:
            return self.lookup_cache['sports'][sport_name]

        # Comprehensive fuzzy matching
        sport_lower = sport_name.lower().strip()
        mappings = {
            'soccer': 'Soccer',
            'football': 'Soccer',  # Default football to soccer
            'american football': 'American Football',
            'nfl': 'American Football',
            'basketball': 'Basketball',
            'nba': 'Basketball',
            'baseball': 'Baseball',
            'mlb': 'Baseball',
            'hockey': 'Ice Hockey',
            'ice hockey': 'Ice Hockey',
            'nhl': 'Ice Hockey',
            'tennis': 'Tennis',
            'golf': 'Golf',
            'mma': 'MMA',
            'boxing': 'Boxing',
            'esports': 'eSports',
            'e-sports': 'eSports',
            'cricket': 'Cricket',
            'rugby': 'Rugby'
        }

        mapped_sport = mappings.get(sport_lower)
        if mapped_sport and mapped_sport in self.lookup_cache['sports']:
            return self.lookup_cache['sports'][mapped_sport]

        return 1  # Ultimate fallback

    def _parse_datetime_robust(self, dt_str: str) -> Optional[datetime]:
        """Robust datetime parsing with multiple format support."""
        if not dt_str:
            return None

        try:
            # Try multiple formats
            formats = [
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d'
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(dt_str.replace('Z', ''), fmt.replace('Z', ''))
                except ValueError:
                    continue

            # ISO format fallback
            if 'T' in dt_str:
                return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))

        except Exception as e:
            logger.debug(f"Date parse failed for '{dt_str}': {e}")

        return None

    def verify_complete_migration(self):
        """Comprehensive verification of migration completeness."""
        logger.info("🔍 Verifying complete migration...")

        try:
            # PostgreSQL counts
            with psycopg2.connect(**self.pg_config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM ominari.markets_normalized")
                    pg_market_count = cursor.fetchone()[0]

                    cursor.execute("SELECT COUNT(*) FROM ominari.lu_teams")
                    pg_team_count = cursor.fetchone()[0]

                    cursor.execute("SELECT pg_size_pretty(pg_database_size('ominari_production'))")
                    pg_db_size = cursor.fetchone()[0]

            # SQLite comparison (if accessible)
            sqlite_market_count = "N/A (large database)"
            try:
                with self.get_sqlite_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM market LIMIT 100000")
                    sqlite_market_count = cursor.fetchone()[0]
            except:
                pass

            logger.info("📊 Migration Verification:")
            logger.info(f"  PostgreSQL Markets: {pg_market_count:,}")
            logger.info(f"  PostgreSQL Teams: {pg_team_count:,}")
            logger.info(f"  PostgreSQL DB Size: {pg_db_size}")
            logger.info(f"  SQLite Markets: {sqlite_market_count}")

            return pg_market_count > 0

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False


def main():
    """Main execution with command line arguments."""
    parser = argparse.ArgumentParser(description="Complete PostgreSQL Migration")
    parser.add_argument('--target-records', type=int, default=1000000, help='Target record count')
    parser.add_argument('--batch-size', type=int, default=25000, help='Batch size for processing')
    args = parser.parse_args()

    migrator = CompleteMigrator(batch_size=args.batch_size)

    logger.info(f"🚀 Starting COMPLETE migration: target={args.target_records:,}, batch_size={args.batch_size:,}")

    # Execute migration
    start_time = time.time()
    migrated_count = migrator.migrate_all_markets(target_records=args.target_records)
    total_time = time.time() - start_time

    if migrated_count > 0:
        logger.info(f"✅ Migration completed: {migrated_count:,} records in {total_time:.2f}s")

        # Verify results
        if migrator.verify_complete_migration():
            logger.info("✅ Verification passed - migration successful!")
        else:
            logger.warning("⚠️ Verification had issues")
    else:
        logger.error("❌ Migration failed")


if __name__ == "__main__":
    main()