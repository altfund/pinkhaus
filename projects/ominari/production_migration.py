#!/usr/bin/env python3
"""
Production PostgreSQL Migration Script
Handles large-scale data migration from SQLite to PostgreSQL with batch processing.
"""

import sqlite3
import psycopg2
import logging
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductionMigrator:
    """Production-scale data migration with batch processing and progress tracking."""

    def __init__(self, sqlite_path='sport_odds.db', batch_size=10000):
        self.sqlite_path = sqlite_path
        self.batch_size = batch_size

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

        logger.info(f"Production migrator initialized with batch size: {batch_size:,}")

    @contextmanager
    def get_sqlite_connection(self):
        """Get optimized SQLite connection."""
        conn = sqlite3.connect(self.sqlite_path, timeout=60.0)
        conn.row_factory = sqlite3.Row

        # Optimize for read performance
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=100000")  # 400MB cache
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=2147483648")  # 2GB memory map

        try:
            yield conn
        finally:
            conn.close()

    def _load_lookup_cache(self):
        """Load PostgreSQL lookup tables."""
        logger.info("Loading lookup cache...")

        try:
            with psycopg2.connect(**self.pg_config) as conn:
                with conn.cursor() as cursor:

                    # Load bookmakers
                    cursor.execute("SELECT id, name FROM ominari.lu_bookmakers")
                    self.lookup_cache['bookmakers'] = {name: id for id, name in cursor.fetchall()}

                    # Load sources
                    cursor.execute("SELECT id, name FROM ominari.lu_sources")
                    self.lookup_cache['sources'] = {name: id for id, name in cursor.fetchall()}

                    # Load market types
                    cursor.execute("SELECT id, name FROM ominari.lu_market_types")
                    self.lookup_cache['market_types'] = {name: id for id, name in cursor.fetchall()}

                    # Load outcomes
                    cursor.execute("SELECT id, name FROM ominari.lu_outcomes")
                    self.lookup_cache['outcomes'] = {name: id for id, name in cursor.fetchall()}

                    # Load sports
                    cursor.execute("SELECT id, name FROM ominari.lu_sports")
                    self.lookup_cache['sports'] = {name: id for id, name in cursor.fetchall()}

                    # Load teams
                    cursor.execute("SELECT id, name FROM ominari.lu_teams")
                    self.lookup_cache['teams'] = {name: id for id, name in cursor.fetchall()}

            logger.info(f"Loaded lookup cache: {len(self.lookup_cache['teams'])} teams, "
                       f"{len(self.lookup_cache['sports'])} sports")
        except Exception as e:
            logger.error(f"Failed to load lookup cache: {e}")
            raise

    def get_total_market_count(self):
        """Get approximate total market count for progress tracking."""
        try:
            with self.get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM market WHERE rowid <= 100000")
                sample_count = cursor.fetchone()[0]

                # Estimate total based on sample
                cursor.execute("SELECT MAX(rowid) FROM market")
                max_rowid = cursor.fetchone()[0]

                estimated_total = (sample_count / 100000) * max_rowid
                logger.info(f"Estimated total markets: {estimated_total:,.0f}")
                return int(estimated_total)

        except Exception as e:
            logger.warning(f"Could not estimate market count: {e}")
            return 1000000  # Default estimate

    def migrate_markets_batch(self, limit=50000, start_rowid=1):
        """Migrate markets in batches with progress tracking."""

        start_time = time.time()
        logger.info(f"Starting batch migration: limit={limit:,}, starting from rowid {start_rowid}")

        try:
            with self.get_sqlite_connection() as sqlite_conn:
                with psycopg2.connect(**self.pg_config) as pg_conn:
                    sqlite_cursor = sqlite_conn.cursor()
                    pg_cursor = pg_conn.cursor()

                    # Use the working query pattern from fixed_sqlite_extractor
                    sqlite_cursor.execute("""
                        SELECT
                            source_id,
                            sport,
                            home_team,
                            away_team,
                            maturity_date,
                            start_time,
                            source,
                            market_type
                        FROM market
                        WHERE rowid >= ?
                        ORDER BY rowid
                        LIMIT ?
                    """, (start_rowid, limit))

                    markets_to_insert = []
                    teams_created = 0

                    for row in sqlite_cursor:
                        market_data = self._transform_market(row)
                        if market_data:
                            markets_to_insert.append(market_data)

                            # Track team creation
                            if market_data.get('home_team_id') and market_data['home_team_id'] not in [t[1] for t in markets_to_insert[:-1] if 'home_team_id' in t]:
                                teams_created += 1
                            if market_data.get('away_team_id') and market_data['away_team_id'] not in [t[1] for t in markets_to_insert[:-1] if 'away_team_id' in t]:
                                teams_created += 1

                    # Batch insert
                    if markets_to_insert:
                        pg_cursor.executemany("""
                            INSERT INTO ominari.markets_normalized
                            (external_id, source_id, sport_id, home_team_id, away_team_id,
                             start_time, maturity_date, market_type_id)
                            VALUES (%(external_id)s, %(source_id)s, %(sport_id)s,
                                   %(home_team_id)s, %(away_team_id)s,
                                   %(start_time)s, %(maturity_date)s, %(market_type_id)s)
                            ON CONFLICT (external_id) DO NOTHING
                        """, markets_to_insert)

                        pg_conn.commit()

                    elapsed = time.time() - start_time
                    rate = len(markets_to_insert) / elapsed if elapsed > 0 else 0

                    logger.info(f"✅ Batch completed: {len(markets_to_insert)} markets, "
                               f"{teams_created} teams created, "
                               f"{elapsed:.2f}s, {rate:.0f} records/sec")

                    return len(markets_to_insert), start_rowid + len(markets_to_insert)

        except Exception as e:
            logger.error(f"Batch migration failed at rowid {start_rowid}: {e}")
            return 0, start_rowid

    def _transform_market(self, row) -> Optional[Dict]:
        """Transform SQLite market row to PostgreSQL format."""
        try:
            # Get or create team IDs
            home_team_id = self._get_or_create_team(row['home_team'], row['sport'])
            away_team_id = self._get_or_create_team(row['away_team'], row['sport'])

            # Map lookups
            source_id = self.lookup_cache['sources'].get(row['source'] or 'overtime_markets', 1)
            sport_id = self._map_sport(row['sport'])
            market_type_id = self.lookup_cache['market_types'].get(row['market_type'] or 'winner', 1)

            # Parse dates
            start_time = self._parse_datetime(row['start_time'])
            maturity_date = self._parse_datetime(row['maturity_date'])

            return {
                'external_id': row['source_id'],
                'source_id': source_id,
                'sport_id': sport_id,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'start_time': start_time,
                'maturity_date': maturity_date,
                'market_type_id': market_type_id
            }

        except Exception as e:
            logger.debug(f"Market transform failed: {e}")
            return None

    def _get_or_create_team(self, team_name: str, sport: str) -> Optional[int]:
        """Get or create team ID."""
        if not team_name or team_name.strip() == '':
            return None

        team_name = team_name.strip()

        # Check cache first
        if team_name in self.lookup_cache['teams']:
            return self.lookup_cache['teams'][team_name]

        # Create new team
        try:
            sport_id = self._map_sport(sport)

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
                        # Team already exists, get its ID
                        cursor.execute("SELECT id FROM ominari.lu_teams WHERE name = %s", (team_name,))
                        result = cursor.fetchone()
                        team_id = result[0] if result else None

                    if team_id:
                        # Cache it
                        self.lookup_cache['teams'][team_name] = team_id

                    return team_id

        except Exception as e:
            logger.debug(f"Team creation failed for '{team_name}': {e}")
            return None

    def _map_sport(self, sport_name: str) -> Optional[int]:
        """Map sport name to ID."""
        if not sport_name:
            return 1  # Default to first sport

        # Direct match
        if sport_name in self.lookup_cache['sports']:
            return self.lookup_cache['sports'][sport_name]

        # Fuzzy matching
        sport_lower = sport_name.lower()
        mappings = {
            'soccer': 'Soccer',
            'football': 'American Football',
            'basketball': 'Basketball',
            'baseball': 'Baseball',
            'hockey': 'Ice Hockey',
            'tennis': 'Tennis',
            'golf': 'Golf',
            'mma': 'MMA',
            'boxing': 'Boxing'
        }

        if sport_lower in mappings:
            mapped_sport = mappings[sport_lower]
            return self.lookup_cache['sports'].get(mapped_sport, 1)

        return 1  # Default sport

    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse datetime string safely."""
        if not dt_str:
            return None

        try:
            # Handle different formats
            if 'T' in dt_str and 'Z' in dt_str:
                return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            elif 'T' in dt_str:
                return datetime.fromisoformat(dt_str)
            else:
                return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logger.debug(f"Date parse failed for '{dt_str}': {e}")
            return None

    def run_full_migration(self, target_records=500000):
        """Run full production migration with progress tracking."""
        logger.info(f"🚀 Starting production migration targeting {target_records:,} records")

        total_migrated = 0
        current_rowid = 1
        batch_num = 1
        start_time = time.time()

        while total_migrated < target_records:
            batch_start = time.time()

            # Calculate remaining records needed
            remaining = target_records - total_migrated
            current_batch_size = min(self.batch_size, remaining)

            logger.info(f"📦 Batch {batch_num}: Processing {current_batch_size:,} records (starting from rowid: {current_rowid})")

            batch_count, next_rowid = self.migrate_markets_batch(limit=current_batch_size, start_rowid=current_rowid)

            if batch_count == 0:
                logger.warning("No more records to process")
                break

            total_migrated += batch_count
            current_rowid = next_rowid
            batch_num += 1

            # Progress reporting
            elapsed = time.time() - start_time
            rate = total_migrated / elapsed if elapsed > 0 else 0
            progress = (total_migrated / target_records) * 100

            logger.info(f"📊 Progress: {total_migrated:,}/{target_records:,} ({progress:.1f}%) "
                       f"at {rate:.0f} records/sec")

            # Brief pause between batches
            time.sleep(0.5)

        total_elapsed = time.time() - start_time
        final_rate = total_migrated / total_elapsed if total_elapsed > 0 else 0

        logger.info(f"🎯 Migration completed: {total_migrated:,} records in {total_elapsed:.2f}s "
                   f"(avg {final_rate:.0f} records/sec)")

        return total_migrated


def main():
    """Main production migration execution."""
    migrator = ProductionMigrator(batch_size=10000)

    # Run migration targeting 100K records for initial production test
    migrated_count = migrator.run_full_migration(target_records=100000)

    if migrated_count > 0:
        logger.info("✅ Production migration test completed successfully!")

        # Verify the data
        with psycopg2.connect(**migrator.pg_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM ominari.markets_normalized")
                total_markets = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM ominari.lu_teams")
                total_teams = cursor.fetchone()[0]

                logger.info(f"📊 Final counts: {total_markets:,} markets, {total_teams:,} teams")
    else:
        logger.error("❌ Production migration failed")


if __name__ == "__main__":
    main()