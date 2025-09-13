#!/usr/bin/env python3
"""
Smart SQLite Data Extractor
Efficiently extracts data from the large SQLite database using targeted queries 
and transforms it for the normalized PostgreSQL schema.
"""

import sqlite3
import psycopg2
import logging
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Optional, Tuple
import json
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SmartSQLiteExtractor:
    """Extract data from SQLite efficiently and load into normalized PostgreSQL."""
    
    def __init__(self, sqlite_path='sport_odds.db', months_back=6):
        self.sqlite_path = sqlite_path
        self.months_back = months_back
        self.cutoff_date = datetime.now() - timedelta(days=30 * months_back)
        
        self.pg_config = {
            'host': os.getenv('PG_HOST', 'localhost'),
            'port': os.getenv('PG_PORT', '5435'),
            'user': os.getenv('PG_USER', 'ominari_user'),
            'password': os.getenv('PG_PASSWORD', 'ominari_2025_secure'),
            'database': os.getenv('PG_DB', 'ominari_production')
        }
        
        # Cache lookup tables for fast mapping
        self.lookup_cache = {}
        self._load_lookup_cache()
        
        logger.info(f"Extractor initialized: extracting data since {self.cutoff_date.date()}")
    
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
        conn.execute("PRAGMA mmap_size=1073741824")  # 1GB memory map
        
        try:
            yield conn
        finally:
            conn.close()
    
    def _load_lookup_cache(self):
        """Load PostgreSQL lookup tables into memory for fast mapping."""
        logger.info("Loading lookup cache...")
        
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
        
        logger.info(f"Loaded {len(self.lookup_cache['bookmakers'])} bookmakers, "
                   f"{len(self.lookup_cache['sources'])} sources, "
                   f"{len(self.lookup_cache['teams'])} teams")
    
    def analyze_sqlite_data(self):
        """Analyze SQLite data to understand what we're working with."""
        logger.info("Analyzing SQLite data structure...")
        
        try:
            with self.get_sqlite_connection() as conn:
                cursor = conn.cursor()
                
                # Get table structure
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                stats = {}
                
                for table in ['market', 'odd']:
                    if table in tables:
                        # Get sample of recent data
                        if table == 'market':
                            cursor.execute(f"""
                                SELECT COUNT(*) as total,
                                       MIN(maturity_date) as earliest,
                                       MAX(maturity_date) as latest
                                FROM {table}
                                WHERE rowid <= 50000
                            """)
                        else:  # odd table
                            cursor.execute(f"""
                                SELECT COUNT(*) as total,
                                       MIN(updated_at) as earliest,
                                       MAX(updated_at) as latest
                                FROM {table}
                                WHERE rowid <= 50000
                            """)
                        
                        result = cursor.fetchone()
                        stats[table] = {
                            'sample_count': result[0],
                            'earliest': result[1],
                            'latest': result[2]
                        }
                        
                        # Get unique values for mapping
                        if table == 'odd':
                            cursor.execute(f"""
                                SELECT DISTINCT bookmaker
                                FROM {table}
                                WHERE bookmaker IS NOT NULL
                                AND rowid <= 100000
                                LIMIT 50
                            """)
                            stats[table]['bookmakers'] = [row[0] for row in cursor.fetchall()]
                            
                            cursor.execute(f"""
                                SELECT DISTINCT source
                                FROM {table}
                                WHERE source IS NOT NULL
                                AND rowid <= 100000
                                LIMIT 20
                            """)
                            stats[table]['sources'] = [row[0] for row in cursor.fetchall()]
                
                logger.info("📊 SQLite analysis results:")
                for table, data in stats.items():
                    logger.info(f"   {table}: {data['sample_count']:,} sample records")
                    logger.info(f"      Date range: {data['earliest']} to {data['latest']}")
                    if 'bookmakers' in data:
                        logger.info(f"      Sample bookmakers: {data['bookmakers'][:5]}")
                    if 'sources' in data:
                        logger.info(f"      Sample sources: {data['sources']}")
                
                return stats
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {}
    
    def extract_markets(self, batch_size=10000):
        """Extract markets from SQLite to PostgreSQL."""
        logger.info("Extracting markets...")
        
        total_extracted = 0
        
        try:
            with self.get_sqlite_connection() as sqlite_conn:
                with psycopg2.connect(**self.pg_config) as pg_conn:
                    sqlite_cursor = sqlite_conn.cursor()
                    pg_cursor = pg_conn.cursor()
                    
                    # Get total count estimate
                    sqlite_cursor.execute("""
                        SELECT COUNT(*) FROM market 
                        WHERE datetime(substr(maturity_date, 1, 19)) >= ? 
                        AND rowid <= 500000
                    """, (self.cutoff_date.isoformat(),))
                    
                    estimated_count = sqlite_cursor.fetchone()[0]
                    logger.info(f"Estimated {estimated_count:,} recent markets to extract")
                    
                    # Extract in batches
                    offset = 0
                    while True:
                        sqlite_cursor.execute("""
                            SELECT 
                                market_id,
                                sport,
                                home_team,
                                away_team,
                                starts_at,
                                maturity_date,
                                source
                            FROM market
                            WHERE datetime(substr(maturity_date, 1, 19)) >= ?
                            ORDER BY rowid
                            LIMIT ? OFFSET ?
                        """, (self.cutoff_date.isoformat(), batch_size, offset))
                        
                        batch = sqlite_cursor.fetchall()
                        if not batch:
                            break
                        
                        # Transform and insert batch
                        markets_to_insert = []
                        for row in batch:
                            market_data = self._transform_market(row)
                            if market_data:
                                markets_to_insert.append(market_data)
                        
                        if markets_to_insert:
                            # Bulk insert
                            pg_cursor.executemany("""
                                INSERT INTO ominari.markets_normalized 
                                (external_id, source_id, sport_id, home_team_id, away_team_id, 
                                 start_time, maturity_date)
                                VALUES (%(external_id)s, %(source_id)s, %(sport_id)s, 
                                       %(home_team_id)s, %(away_team_id)s, 
                                       %(start_time)s, %(maturity_date)s)
                                ON CONFLICT (external_id) DO NOTHING
                            """, markets_to_insert)
                            
                            pg_conn.commit()
                            total_extracted += len(markets_to_insert)
                        
                        offset += batch_size
                        
                        if total_extracted % 50000 == 0 and total_extracted > 0:
                            logger.info(f"   Extracted {total_extracted:,} markets...")
                    
                    logger.info(f"✅ Extracted {total_extracted:,} markets")
                    return total_extracted
                    
        except Exception as e:
            logger.error(f"Market extraction failed: {e}")
            return 0
    
    def _transform_market(self, row) -> Optional[Dict]:
        """Transform SQLite market row to PostgreSQL format."""
        try:
            # Get or create team IDs
            home_team_id = self._get_or_create_team(row['home_team'], row['sport'])
            away_team_id = self._get_or_create_team(row['away_team'], row['sport'])
            
            # Map lookups
            source_id = self.lookup_cache['sources'].get(row['source'] or 'overtime_markets', 0)
            sport_id = self._map_sport(row['sport'])
            
            # Parse dates
            start_time = self._parse_datetime(row['starts_at'])
            maturity_date = self._parse_datetime(row['maturity_date'])
            
            return {
                'external_id': row['market_id'],
                'source_id': source_id,
                'sport_id': sport_id,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'start_time': start_time,
                'maturity_date': maturity_date
            }
            
        except Exception as e:
            logger.debug(f"Market transform failed: {e}")
            return None
    
    def _get_or_create_team(self, team_name: str, sport: str) -> Optional[int]:
        """Get or create team ID with caching."""
        if not team_name:
            return None
        
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
                        team_id = cursor.fetchone()[0]
                    
                    # Cache it
                    self.lookup_cache['teams'][team_name] = team_id
                    return team_id
                    
        except Exception as e:
            logger.debug(f"Team creation failed: {e}")
            return None
    
    def _map_sport(self, sport_name: str) -> Optional[int]:
        """Map sport name to ID with fuzzy matching."""
        if not sport_name:
            return None
        
        # Direct match
        if sport_name in self.lookup_cache['sports']:
            return self.lookup_cache['sports'][sport_name]
        
        # Fuzzy matching
        sport_lower = sport_name.lower()
        for name, id in self.lookup_cache['sports'].items():
            if name.lower() in sport_lower or sport_lower in name.lower():
                return id
        
        return None  # Unknown sport
    
    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse datetime string safely."""
        if not dt_str:
            return None
        
        try:
            # Handle different formats
            if 'T' in dt_str:
                return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            else:
                return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        except:
            return None
    
    def extract_recent_odds_sample(self, max_records=1000000):
        """Extract a sample of recent odds for testing."""
        logger.info(f"Extracting sample of {max_records:,} recent odds...")
        
        total_extracted = 0
        
        try:
            with self.get_sqlite_connection() as sqlite_conn:
                with psycopg2.connect(**self.pg_config) as pg_conn:
                    sqlite_cursor = sqlite_conn.cursor()
                    pg_cursor = pg_conn.cursor()
                    
                    # Extract recent odds sample
                    sqlite_cursor.execute("""
                        SELECT 
                            source_id,
                            bookmaker,
                            outcome,
                            decimal_odds,
                            american_odds,
                            normalized_implied,
                            updated_at
                        FROM odd
                        WHERE datetime(substr(updated_at, 1, 19)) >= ?
                        AND decimal_odds IS NOT NULL
                        ORDER BY rowid DESC
                        LIMIT ?
                    """, (self.cutoff_date.isoformat(), max_records))
                    
                    batch_size = 10000
                    batch = []
                    
                    for row in sqlite_cursor:
                        odds_data = self._transform_odds(row)
                        if odds_data:
                            batch.append(odds_data)
                        
                        if len(batch) >= batch_size:
                            self._insert_odds_batch(pg_cursor, batch)
                            pg_conn.commit()
                            total_extracted += len(batch)
                            batch = []
                            
                            if total_extracted % 50000 == 0:
                                logger.info(f"   Extracted {total_extracted:,} odds...")
                    
                    # Insert remaining batch
                    if batch:
                        self._insert_odds_batch(pg_cursor, batch)
                        pg_conn.commit()
                        total_extracted += len(batch)
                    
                    logger.info(f"✅ Extracted {total_extracted:,} odds")
                    return total_extracted
                    
        except Exception as e:
            logger.error(f"Odds extraction failed: {e}")
            return 0
    
    def _transform_odds(self, row) -> Optional[Dict]:
        """Transform SQLite odds row to PostgreSQL format."""
        try:
            # Map lookups
            bookmaker_id = self.lookup_cache['bookmakers'].get(row['bookmaker'])
            outcome_id = self.lookup_cache['outcomes'].get(row['outcome'])
            
            if bookmaker_id is None or outcome_id is None:
                return None
            
            # Get market ID from PostgreSQL
            market_id = self._get_market_id(row['source_id'])
            if not market_id:
                return None
            
            # Convert values to optimized storage
            decimal_odds_x1000 = int(float(row['decimal_odds']) * 1000) if row['decimal_odds'] else 0
            implied_prob_x10000 = int(float(row['normalized_implied']) * 10000) if row['normalized_implied'] else None
            updated_at_ts = int(self._parse_datetime(row['updated_at']).timestamp()) if row['updated_at'] else int(time.time())
            
            return {
                'market_id': market_id,
                'bookmaker_id': bookmaker_id,
                'outcome_id': outcome_id,
                'decimal_odds_x1000': decimal_odds_x1000,
                'american_odds': row['american_odds'],
                'implied_prob_x10000': implied_prob_x10000,
                'updated_at_ts': updated_at_ts
            }
            
        except Exception as e:
            logger.debug(f"Odds transform failed: {e}")
            return None
    
    def _get_market_id(self, external_id: str) -> Optional[int]:
        """Get PostgreSQL market ID from external ID."""
        try:
            with psycopg2.connect(**self.pg_config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT id FROM ominari.markets_normalized 
                        WHERE external_id = %s
                    """, (external_id,))
                    
                    result = cursor.fetchone()
                    return result[0] if result else None
                    
        except Exception as e:
            return None
    
    def _insert_odds_batch(self, cursor, batch):
        """Insert batch of odds efficiently."""
        if not batch:
            return
        
        cursor.executemany("""
            INSERT INTO ominari.odds_normalized 
            (market_id, bookmaker_id, outcome_id, decimal_odds_x1000, 
             american_odds, implied_prob_x10000, updated_at_ts)
            VALUES (%(market_id)s, %(bookmaker_id)s, %(outcome_id)s, 
                   %(decimal_odds_x1000)s, %(american_odds)s, 
                   %(implied_prob_x10000)s, %(updated_at_ts)s)
            ON CONFLICT (market_id, bookmaker_id, outcome_id, updated_at_ts) DO NOTHING
        """, batch)
    
    def run_extraction(self):
        """Run complete extraction process."""
        logger.info("🚀 Starting smart SQLite extraction...")
        
        start_time = time.time()
        
        try:
            # Analyze source data
            stats = self.analyze_sqlite_data()
            
            # Extract markets first
            markets_extracted = self.extract_markets()
            
            # Extract sample of recent odds
            odds_extracted = self.extract_recent_odds_sample()
            
            elapsed = time.time() - start_time
            
            logger.info(f"✅ Extraction completed in {elapsed:.1f}s")
            logger.info(f"   Markets: {markets_extracted:,}")
            logger.info(f"   Odds: {odds_extracted:,}")
            
            # Verify extraction
            self._verify_extraction()
            
            return {
                'markets': markets_extracted,
                'odds': odds_extracted,
                'duration': elapsed
            }
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            raise
    
    def _verify_extraction(self):
        """Verify extracted data integrity."""
        logger.info("Verifying extracted data...")
        
        with psycopg2.connect(**self.pg_config) as conn:
            with conn.cursor() as cursor:
                
                # Count extracted records
                cursor.execute("SELECT COUNT(*) FROM ominari.markets_normalized")
                market_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM ominari.odds_normalized")
                odds_count = cursor.fetchone()[0]
                
                # Sample some data
                cursor.execute("""
                    SELECT m.external_id, s.name as sport, ht.name as home_team, at.name as away_team
                    FROM ominari.markets_normalized m
                    LEFT JOIN ominari.lu_sports s ON m.sport_id = s.id
                    LEFT JOIN ominari.lu_teams ht ON m.home_team_id = ht.id
                    LEFT JOIN ominari.lu_teams at ON m.away_team_id = at.id
                    ORDER BY m.id DESC
                    LIMIT 3
                """)
                
                samples = cursor.fetchall()
                
                logger.info(f"📊 Verification results:")
                logger.info(f"   Markets: {market_count:,}")
                logger.info(f"   Odds: {odds_count:,}")
                logger.info(f"   Sample markets:")
                for external_id, sport, home, away in samples:
                    logger.info(f"     {external_id[:16]}... {sport}: {home} vs {away}")


def main():
    """Main extraction execution."""
    extractor = SmartSQLiteExtractor(months_back=6)  # Extract last 6 months
    extractor.run_extraction()


if __name__ == "__main__":
    main()