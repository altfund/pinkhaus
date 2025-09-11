#!/usr/bin/env python3
"""
Hybrid Database Access Layer

Provides unified access to both SQLite (historical) and PostgreSQL (blockchain) data.
Routes queries automatically based on data age and source type.
"""

import sqlite3
import psycopg2
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from contextlib import contextmanager
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Unified market data structure."""
    market_id: str
    sport: str
    league: str
    home_team: str
    away_team: str
    start_time: datetime
    is_finished: bool
    source: str  # 'sqlite' or 'postgres'
    

@dataclass
class OddsData:
    """Unified odds data structure."""
    market_id: str
    outcome: str
    decimal_odds: float
    bookmaker: str
    timestamp: datetime
    source: str


class HybridDatabaseAccess:
    """Unified access layer for hybrid SQLite/PostgreSQL architecture."""
    
    def __init__(self):
        # SQLite connection (historical data)
        self.sqlite_path = 'sport_odds.db'
        
        # PostgreSQL connection details
        self.pg_config = {
            'host': os.getenv('PG_HOST', 'localhost'),
            'port': os.getenv('PG_PORT', '5435'),
            'user': os.getenv('PG_USER', 'ominari_user'),
            'password': os.getenv('PG_PASSWORD', 'ominari_2025_secure'),
            'database': os.getenv('PG_DB', 'ominari_production')
        }
        
        # Data cutoff - queries newer than this check PostgreSQL first
        self.data_cutoff = datetime.now() - timedelta(days=180)
        
        logger.info(f"Hybrid access initialized with cutoff: {self.data_cutoff.date()}")
    
    @contextmanager
    def get_sqlite_connection(self):
        """Get optimized SQLite connection."""
        conn = sqlite3.connect(self.sqlite_path, timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        
        # Apply read optimizations
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        
        try:
            yield conn
        finally:
            conn.close()
    
    @contextmanager
    def get_postgres_connection(self):
        """Get PostgreSQL connection."""
        try:
            conn = psycopg2.connect(**self.pg_config)
            conn.autocommit = True
            yield conn
        except psycopg2.Error as e:
            logger.warning(f"PostgreSQL connection failed: {e}")
            yield None
        finally:
            if 'conn' in locals() and conn:
                conn.close()
    
    def get_markets(self, 
                   sport: Optional[str] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   limit: int = 100) -> List[MarketData]:
        """Get markets from appropriate source based on date range."""
        
        markets = []
        
        # Determine which sources to query
        use_postgres = (start_date is None or start_date >= self.data_cutoff)
        use_sqlite = (end_date is None or end_date < datetime.now() - timedelta(days=30))
        
        # Query PostgreSQL for recent data
        if use_postgres:
            markets.extend(self._get_markets_postgres(sport, start_date, end_date, limit))
        
        # Query SQLite for historical data
        if use_sqlite and len(markets) < limit:
            remaining_limit = limit - len(markets)
            markets.extend(self._get_markets_sqlite(sport, start_date, end_date, remaining_limit))
        
        return markets[:limit]
    
    def _get_markets_postgres(self, 
                             sport: Optional[str],
                             start_date: Optional[datetime],
                             end_date: Optional[datetime],
                             limit: int) -> List[MarketData]:
        """Get markets from PostgreSQL."""
        markets = []
        
        with self.get_postgres_connection() as conn:
            if not conn:
                return markets
            
            cursor = conn.cursor()
            
            # Build query
            query = """
                SELECT market_id, sport, league, home_team, away_team, 
                       start_time, resolved, 'postgres' as source
                FROM blockchain.markets 
                WHERE 1=1
            """
            params = []
            
            if sport:
                query += " AND sport ILIKE %s"
                params.append(f"%{sport}%")
            
            if start_date:
                query += " AND start_time >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND start_time <= %s"
                params.append(end_date)
            
            query += " ORDER BY start_time DESC LIMIT %s"
            params.append(limit)
            
            try:
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                for row in rows:
                    markets.append(MarketData(
                        market_id=row[0],
                        sport=row[1] or 'Unknown',
                        league=row[2] or 'Unknown',
                        home_team=row[3] or 'Home',
                        away_team=row[4] or 'Away',
                        start_time=row[5] or datetime.now(),
                        is_finished=bool(row[6]),
                        source='postgres'
                    ))
                
                logger.debug(f"Retrieved {len(markets)} markets from PostgreSQL")
                
            except Exception as e:
                logger.error(f"PostgreSQL market query error: {e}")
        
        return markets
    
    def _get_markets_sqlite(self,
                           sport: Optional[str],
                           start_date: Optional[datetime],
                           end_date: Optional[datetime],
                           limit: int) -> List[MarketData]:
        """Get markets from SQLite."""
        markets = []
        
        with self.get_sqlite_connection() as conn:
            cursor = conn.cursor()
            
            # Build query
            query = """
                SELECT source_id, sport, league_name, home_team, away_team,
                       maturity_date, is_finished, 'sqlite' as source
                FROM market 
                WHERE 1=1
            """
            params = []
            
            if sport:
                query += " AND sport LIKE ?"
                params.append(f"%{sport}%")
            
            if start_date:
                query += " AND datetime(substr(maturity_date, 1, 19)) >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND datetime(substr(maturity_date, 1, 19)) <= ?"
                params.append(end_date.isoformat())
            
            # Limit to recent data for performance
            query += " AND datetime(substr(maturity_date, 1, 19)) >= date('now', '-365 days')"
            query += " ORDER BY maturity_date DESC LIMIT ?"
            params.append(limit)
            
            try:
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                for row in rows:
                    # Parse datetime safely
                    try:
                        start_time = datetime.fromisoformat(row[5].replace('Z', '+00:00')) if row[5] else datetime.now()
                    except:
                        start_time = datetime.now()
                    
                    markets.append(MarketData(
                        market_id=row[0],
                        sport=row[1] or 'Unknown',
                        league=row[2] or 'Unknown',
                        home_team=row[3] or 'Home',
                        away_team=row[4] or 'Away',
                        start_time=start_time,
                        is_finished=bool(row[6]),
                        source='sqlite'
                    ))
                
                logger.debug(f"Retrieved {len(markets)} markets from SQLite")
                
            except Exception as e:
                logger.error(f"SQLite market query error: {e}")
        
        return markets
    
    def get_latest_odds(self, market_id: str) -> List[OddsData]:
        """Get latest odds for a market from appropriate source."""
        
        odds = []
        
        # Try PostgreSQL first (for recent data)
        odds.extend(self._get_odds_postgres(market_id))
        
        # If no results, try SQLite
        if not odds:
            odds.extend(self._get_odds_sqlite(market_id))
        
        return odds
    
    def _get_odds_postgres(self, market_id: str) -> List[OddsData]:
        """Get odds from PostgreSQL."""
        odds = []
        
        with self.get_postgres_connection() as conn:
            if not conn:
                return odds
            
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    SELECT DISTINCT ON (outcome)
                        market_id, outcome, decimal_odds, 'blockchain' as bookmaker,
                        timestamp, 'postgres' as source
                    FROM blockchain.odds 
                    WHERE market_id = %s
                    ORDER BY outcome, timestamp DESC
                """, (market_id,))
                
                rows = cursor.fetchall()
                
                for row in rows:
                    odds.append(OddsData(
                        market_id=row[0],
                        outcome=row[1],
                        decimal_odds=float(row[2]),
                        bookmaker=row[3],
                        timestamp=row[4],
                        source='postgres'
                    ))
                
            except Exception as e:
                logger.error(f"PostgreSQL odds query error: {e}")
        
        return odds
    
    def _get_odds_sqlite(self, market_id: str) -> List[OddsData]:
        """Get odds from SQLite."""
        odds = []
        
        with self.get_sqlite_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    SELECT source_id, outcome, decimal_odds, bookmaker,
                           updated_at, 'sqlite' as source
                    FROM odd 
                    WHERE source_id = ?
                    ORDER BY updated_at DESC
                    LIMIT 20
                """, (market_id,))
                
                rows = cursor.fetchall()
                
                for row in rows:
                    # Parse datetime safely
                    try:
                        timestamp = datetime.fromisoformat(row[4].replace('Z', '+00:00')) if row[4] else datetime.now()
                    except:
                        timestamp = datetime.now()
                    
                    odds.append(OddsData(
                        market_id=row[0],
                        outcome=row[1],
                        decimal_odds=float(row[2]) if row[2] else 1.0,
                        bookmaker=row[3] or 'Unknown',
                        timestamp=timestamp,
                        source='sqlite'
                    ))
                
            except Exception as e:
                logger.error(f"SQLite odds query error: {e}")
        
        return odds
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics from both databases."""
        stats = {
            'sqlite': {},
            'postgres': {},
            'hybrid_cutoff': self.data_cutoff.isoformat()
        }
        
        # SQLite stats
        with self.get_sqlite_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM market WHERE rowid < 100000")
                stats['sqlite']['markets_sample'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM odd WHERE rowid < 100000")
                stats['sqlite']['odds_sample'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
                stats['sqlite']['size_bytes'] = cursor.fetchone()[0]
                
            except Exception as e:
                logger.error(f"SQLite stats error: {e}")
        
        # PostgreSQL stats
        with self.get_postgres_connection() as conn:
            if conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT COUNT(*) FROM blockchain.markets")
                    stats['postgres']['markets'] = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM blockchain.odds")
                    stats['postgres']['odds'] = cursor.fetchone()[0]
                    
                    cursor.execute("""
                        SELECT pg_size_pretty(pg_database_size(current_database()))
                    """)
                    stats['postgres']['size'] = cursor.fetchone()[0]
                    
                except Exception as e:
                    logger.error(f"PostgreSQL stats error: {e}")
        
        return stats
    
    def test_hybrid_access(self):
        """Test the hybrid access system."""
        logger.info("🧪 Testing hybrid database access...")
        
        # Test 1: Get recent markets
        logger.info("Test 1: Recent markets")
        recent_markets = self.get_markets(
            start_date=datetime.now() - timedelta(days=30),
            limit=5
        )
        logger.info(f"  Found {len(recent_markets)} recent markets")
        for market in recent_markets[:2]:
            logger.info(f"    {market.market_id}: {market.home_team} vs {market.away_team} ({market.source})")
        
        # Test 2: Get historical markets
        logger.info("Test 2: Historical markets")
        historical_markets = self.get_markets(
            end_date=datetime.now() - timedelta(days=200),
            limit=5
        )
        logger.info(f"  Found {len(historical_markets)} historical markets")
        for market in historical_markets[:2]:
            logger.info(f"    {market.market_id}: {market.home_team} vs {market.away_team} ({market.source})")
        
        # Test 3: Get odds for a market
        if recent_markets:
            logger.info("Test 3: Market odds")
            market_id = recent_markets[0].market_id
            odds = self.get_latest_odds(market_id)
            logger.info(f"  Found {len(odds)} odds for {market_id}")
            for odd in odds[:3]:
                logger.info(f"    {odd.outcome}: {odd.decimal_odds} ({odd.source})")
        
        # Test 4: Database stats
        logger.info("Test 4: Database statistics")
        stats = self.get_database_stats()
        logger.info(f"  SQLite markets (sample): {stats['sqlite'].get('markets_sample', 'N/A'):,}")
        logger.info(f"  PostgreSQL markets: {stats['postgres'].get('markets', 'N/A'):,}")
        logger.info(f"  PostgreSQL odds: {stats['postgres'].get('odds', 'N/A'):,}")
        
        logger.info("✅ Hybrid access testing complete")


if __name__ == "__main__":
    # Initialize and test hybrid access
    hybrid = HybridDatabaseAccess()
    hybrid.test_hybrid_access()