#!/usr/bin/env python3
"""
Fixed SQLite Data Extractor
Updated to work with the actual SQLite schema discovered in the database.
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


class FixedSQLiteExtractor:
    """Extract data from SQLite using correct schema and load into PostgreSQL."""
    
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
        
        # Cache lookup tables
        self.lookup_cache = {}
        self._load_lookup_cache()
        
        logger.info(f"Extractor initialized: extracting data since {self.cutoff_date.date()}")
    
    @contextmanager
    def get_sqlite_connection(self):
        """Get optimized SQLite connection."""
        conn = sqlite3.connect(self.sqlite_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        
        # Optimize for read performance  
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=50000")  # 200MB cache
        conn.execute("PRAGMA temp_store=MEMORY")
        
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
            
            logger.info(f"Loaded {len(self.lookup_cache['bookmakers'])} bookmakers, "
                       f"{len(self.lookup_cache['sources'])} sources")
        except Exception as e:
            logger.error(f"Failed to load lookup cache: {e}")
            self.lookup_cache = {'bookmakers': {}, 'sources': {}, 'market_types': {}, 
                               'outcomes': {}, 'sports': {}, 'teams': {}}
    
    def test_small_extraction(self):
        """Test extraction with a small sample."""
        logger.info("Testing small extraction...")
        
        try:
            with self.get_sqlite_connection() as sqlite_conn:
                cursor = sqlite_conn.cursor()
                
                # Test market extraction
                cursor.execute("""
                    SELECT source_id, sport, home_team, away_team, maturity_date, 
                           start_time, source, market_type
                    FROM market
                    WHERE rowid <= 10
                    LIMIT 5
                """)
                
                markets = cursor.fetchall()
                logger.info(f"Found {len(markets)} test markets:")
                
                for market in markets:
                    logger.info(f"  {market['source_id']}: {market['home_team']} vs {market['away_team']} ({market['sport']})")
                
                # Test odds extraction
                cursor.execute("""
                    SELECT source_id, bookmaker, outcome, decimal_odds, 
                           american_odds, normalized_implied, updated_at
                    FROM odd
                    WHERE rowid <= 10
                    LIMIT 5
                """)
                
                odds = cursor.fetchall()
                logger.info(f"Found {len(odds)} test odds:")
                
                for odd in odds:
                    logger.info(f"  {odd['source_id']}: {odd['bookmaker']} - {odd['outcome']} @ {odd['decimal_odds']}")
                
                logger.info("✅ Small extraction test successful!")
                return True
                
        except Exception as e:
            logger.error(f"❌ Test extraction failed: {e}")
            return False
    
    def extract_sample_markets(self, limit=1000):
        """Extract a sample of markets to PostgreSQL."""
        logger.info(f"Extracting {limit} sample markets...")
        
        extracted_count = 0
        
        try:
            with self.get_sqlite_connection() as sqlite_conn:
                with psycopg2.connect(**self.pg_config) as pg_conn:
                    sqlite_cursor = sqlite_conn.cursor()
                    pg_cursor = pg_conn.cursor()
                    
                    # Get recent markets with correct column names
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
                        WHERE datetime(substr(maturity_date, 1, 19)) >= ?
                        ORDER BY rowid
                        LIMIT ?
                    """, (self.cutoff_date.isoformat(), limit))
                    
                    markets_to_insert = []
                    
                    for row in sqlite_cursor:
                        market_data = self._transform_market(row)
                        if market_data:
                            markets_to_insert.append(market_data)
                    
                    if markets_to_insert:
                        # Insert markets
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
                        extracted_count = len(markets_to_insert)
                    
                    logger.info(f"✅ Extracted {extracted_count} markets")
                    return extracted_count
                    
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
            market_type_id = self.lookup_cache['market_types'].get(row['market_type'] or 'winner', 0)
            
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
            return None
        
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
            return self.lookup_cache['sports'].get(mapped_sport)
        
        return None
    
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
    
    def verify_postgresql_data(self):
        """Verify extracted data in PostgreSQL."""
        logger.info("Verifying PostgreSQL data...")
        
        try:
            with psycopg2.connect(**self.pg_config) as conn:
                with conn.cursor() as cursor:
                    
                    # Count records
                    cursor.execute("SELECT COUNT(*) FROM ominari.markets_normalized")
                    market_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM ominari.lu_teams")
                    team_count = cursor.fetchone()[0]
                    
                    # Sample some data
                    cursor.execute("""
                        SELECT m.external_id, s.name as sport, ht.name as home_team, 
                               at.name as away_team, m.start_time
                        FROM ominari.markets_normalized m
                        LEFT JOIN ominari.lu_sports s ON m.sport_id = s.id
                        LEFT JOIN ominari.lu_teams ht ON m.home_team_id = ht.id
                        LEFT JOIN ominari.lu_teams at ON m.away_team_id = at.id
                        ORDER BY m.id DESC
                        LIMIT 5
                    """)
                    
                    samples = cursor.fetchall()
                    
                    logger.info(f"📊 PostgreSQL verification:")
                    logger.info(f"   Markets: {market_count:,}")
                    logger.info(f"   Teams: {team_count:,}")
                    logger.info(f"   Sample markets:")
                    for external_id, sport, home, away, start_time in samples:
                        logger.info(f"     {external_id}: {sport} - {home} vs {away} @ {start_time}")
                    
                    return market_count > 0
                    
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False


def main():
    """Main test execution."""
    extractor = FixedSQLiteExtractor(months_back=8)  # Get more data
    
    # Test small extraction first
    if extractor.test_small_extraction():
        # Extract sample markets
        count = extractor.extract_sample_markets(limit=5000)
        
        if count > 0:
            # Verify the data
            extractor.verify_postgresql_data()
            logger.info("🎯 Sample extraction completed successfully!")
        else:
            logger.error("❌ No markets extracted")
    else:
        logger.error("❌ Small extraction test failed")


if __name__ == "__main__":
    main()