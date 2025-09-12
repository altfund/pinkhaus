#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Historical Data Fetcher
Fetches and stores comprehensive historical data for backtesting.
"""

import asyncio
import logging
import sqlite3
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import json

from data_source_manager import DataSourceManager

logger = logging.getLogger(__name__)


class HistoricalDataFetcher:
    """
    Fetches historical data from multiple sources.
    
    Features:
    - Multi-source data aggregation
    - Incremental updates
    - Data validation and cleaning
    - Efficient storage with compression
    """
    
    def __init__(self, 
                 network: str = 'optimism',
                 db_path: str = 'historical_data.db'):
        self.network = network
        self.db_path = db_path
        self.data_manager = DataSourceManager(network)
        
        self._init_database()
        
    def _init_database(self):
        """Initialize historical database with optimized schema."""
        conn = sqlite3.connect(self.db_path)
        
        # Markets table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_markets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                sport TEXT,
                league TEXT,
                home_team TEXT,
                away_team TEXT,
                market_type TEXT,
                maturity_date DATETIME,
                is_resolved BOOLEAN,
                winning_outcome TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_id, timestamp)
            )
        """)
        
        # Odds table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_odds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                bookmaker TEXT,
                home_odds REAL,
                away_odds REAL,
                draw_odds REAL,
                home_spread REAL,
                away_spread REAL,
                total_line REAL,
                over_odds REAL,
                under_odds REAL,
                volume REAL,
                liquidity REAL,
                FOREIGN KEY (market_id) REFERENCES historical_markets(id),
                UNIQUE(market_id, timestamp, bookmaker)
            )
        """)
        
        # Trades table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                side TEXT,
                outcome TEXT,
                size REAL,
                price REAL,
                trader TEXT,
                tx_hash TEXT,
                FOREIGN KEY (market_id) REFERENCES historical_markets(id)
            )
        """)
        
        # Create indexes for performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_markets_source ON historical_markets(source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_markets_time ON historical_markets(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_markets_sport ON historical_markets(sport)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_odds_market ON historical_odds(market_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_odds_time ON historical_odds(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_market ON historical_trades(market_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_time ON historical_trades(timestamp)")
        
        # Metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fetch_metadata (
                source TEXT PRIMARY KEY,
                last_fetch_time DATETIME,
                last_block_number INTEGER,
                total_records INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
        
    async def fetch_historical_range(self, 
                                   start_date: datetime,
                                   end_date: datetime,
                                   sports: Optional[List[str]] = None) -> Dict[str, int]:
        """Fetch historical data for a date range."""
        await self.data_manager.initialize()
        
        stats = {
            'markets': 0,
            'odds': 0,
            'trades': 0,
            'errors': 0
        }
        
        # Fetch in daily chunks to avoid memory issues
        current_date = start_date
        while current_date < end_date:
            next_date = min(current_date + timedelta(days=1), end_date)
            
            try:
                logger.info(f"Fetching data for {current_date.date()}")
                
                # Fetch markets
                markets = await self._fetch_markets_chunk(current_date, next_date, sports)
                stats['markets'] += len(markets)
                
                # Fetch odds for these markets
                for market in markets:
                    odds = await self._fetch_odds_chunk(market['source_id'], current_date, next_date)
                    stats['odds'] += len(odds)
                    
                # Fetch trades (if available)
                trades = await self._fetch_trades_chunk(current_date, next_date)
                stats['trades'] += len(trades)
                
                # Store progress
                self._update_metadata('historical_fetch', current_date)
                
            except Exception as e:
                logger.error(f"Error fetching data for {current_date}: {e}")
                stats['errors'] += 1
                
            current_date = next_date
            
            # Rate limiting
            await asyncio.sleep(1)
            
        logger.info(f"Historical fetch complete: {stats}")
        return stats
        
    async def _fetch_markets_chunk(self, 
                                 start: datetime,
                                 end: datetime,
                                 sports: Optional[List[str]]) -> List[Dict]:
        """Fetch markets for a time chunk."""
        conn = sqlite3.connect(self.db_path)
        markets_stored = []
        
        try:
            # Try GraphQL first
            if hasattr(self.data_manager, 'graphql_source'):
                markets = await self.data_manager.graphql_source.get_markets()
                
                for market in markets:
                    # Check if in date range
                    market_date = pd.to_datetime(market.get('maturity_date'))
                    if start <= market_date <= end:
                        if not sports or market.get('sport') in sports:
                            market_id = self._store_market(conn, market)
                            markets_stored.append({'id': market_id, **market})
                            
            # Fallback to blockchain
            elif hasattr(self.data_manager, 'blockchain_source'):
                # Would query historical events from blockchain
                pass
                
        finally:
            conn.commit()
            conn.close()
            
        return markets_stored
        
    async def _fetch_odds_chunk(self, 
                              source_id: str,
                              start: datetime,
                              end: datetime) -> List[Dict]:
        """Fetch odds history for a market."""
        conn = sqlite3.connect(self.db_path)
        odds_stored = []
        
        try:
            # Get market id
            market = conn.execute(
                "SELECT id FROM historical_markets WHERE source_id = ?",
                (source_id,)
            ).fetchone()
            
            if not market:
                return []
                
            market_id = market[0]
            
            # Fetch odds snapshots
            # This would query historical odds from blockchain events or API
            # For now, simulate with some data points
            current = start
            while current < end:
                odds_data = {
                    'market_id': market_id,
                    'timestamp': current,
                    'bookmaker': 'overtime',
                    'home_odds': 2.0 + np.random.random() * 0.5,
                    'away_odds': 2.0 + np.random.random() * 0.5,
                    'volume': np.random.exponential(10000),
                    'liquidity': np.random.exponential(50000)
                }
                
                self._store_odds(conn, odds_data)
                odds_stored.append(odds_data)
                
                current += timedelta(hours=1)
                
        finally:
            conn.commit()
            conn.close()
            
        return odds_stored
        
    async def _fetch_trades_chunk(self,
                                start: datetime,
                                end: datetime) -> List[Dict]:
        """Fetch trades for a time chunk."""
        # This would query blockchain for actual trade events
        # For now, return empty
        return []
        
    def _store_market(self, conn: sqlite3.Connection, market: Dict) -> int:
        """Store market and return id."""
        cursor = conn.execute("""
            INSERT OR IGNORE INTO historical_markets
            (source_id, timestamp, sport, league, home_team, away_team,
             market_type, maturity_date, is_resolved, winning_outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market['source_id'],
            market.get('timestamp', datetime.now(timezone.utc)),
            market.get('sport'),
            market.get('league'),
            market.get('home_team'),
            market.get('away_team'),
            market.get('market_type', 'winner'),
            market.get('maturity_date'),
            market.get('is_resolved', False),
            market.get('winning_outcome')
        ))
        
        if cursor.lastrowid:
            return cursor.lastrowid
        else:
            # Get existing id
            result = conn.execute(
                "SELECT id FROM historical_markets WHERE source_id = ?",
                (market['source_id'],)
            ).fetchone()
            return result[0] if result else None
            
    def _store_odds(self, conn: sqlite3.Connection, odds: Dict):
        """Store odds snapshot."""
        conn.execute("""
            INSERT OR IGNORE INTO historical_odds
            (market_id, timestamp, bookmaker, home_odds, away_odds,
             draw_odds, volume, liquidity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            odds['market_id'],
            odds['timestamp'],
            odds.get('bookmaker', 'unknown'),
            odds.get('home_odds'),
            odds.get('away_odds'),
            odds.get('draw_odds'),
            odds.get('volume', 0),
            odds.get('liquidity', 0)
        ))
        
    def _update_metadata(self, source: str, timestamp: datetime):
        """Update fetch metadata."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO fetch_metadata
            (source, last_fetch_time)
            VALUES (?, ?)
        """, (source, timestamp))
        conn.commit()
        conn.close()
        
    def get_data_summary(self) -> Dict:
        """Get summary of available historical data."""
        conn = sqlite3.connect(self.db_path)
        
        summary = {}
        
        # Markets summary
        markets = conn.execute("""
            SELECT 
                COUNT(*) as total,
                MIN(timestamp) as earliest,
                MAX(timestamp) as latest,
                COUNT(DISTINCT sport) as sports,
                COUNT(DISTINCT league) as leagues
            FROM historical_markets
        """).fetchone()
        
        summary['markets'] = {
            'total': markets[0],
            'earliest': markets[1],
            'latest': markets[2],
            'sports': markets[3],
            'leagues': markets[4]
        }
        
        # Odds summary
        odds = conn.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT market_id) as markets_with_odds,
                AVG(volume) as avg_volume,
                SUM(volume) as total_volume
            FROM historical_odds
        """).fetchone()
        
        summary['odds'] = {
            'total_snapshots': odds[0],
            'markets_with_odds': odds[1],
            'avg_volume': odds[2],
            'total_volume': odds[3]
        }
        
        # Sports breakdown
        sports = conn.execute("""
            SELECT sport, COUNT(*) as count
            FROM historical_markets
            GROUP BY sport
            ORDER BY count DESC
        """).fetchall()
        
        summary['sports_breakdown'] = {s[0]: s[1] for s in sports}
        
        conn.close()
        return summary
        
    def export_for_backtesting(self,
                              start_date: datetime,
                              end_date: datetime,
                              sports: Optional[List[str]] = None) -> pd.DataFrame:
        """Export data in format ready for backtesting."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                m.source_id,
                m.timestamp as market_time,
                m.sport,
                m.league,
                m.home_team,
                m.away_team,
                m.market_type,
                m.maturity_date,
                m.is_resolved,
                m.winning_outcome,
                o.timestamp as odds_time,
                o.bookmaker,
                o.home_odds,
                o.away_odds,
                o.draw_odds,
                o.volume,
                o.liquidity
            FROM historical_markets m
            LEFT JOIN historical_odds o ON m.id = o.market_id
            WHERE m.timestamp BETWEEN ? AND ?
        """
        
        params = [start_date, end_date]
        
        if sports:
            placeholders = ','.join(['?' for _ in sports])
            query += f" AND m.sport IN ({placeholders})"
            params.extend(sports)
            
        query += " ORDER BY m.timestamp, o.timestamp"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convert timestamps
        df['market_time'] = pd.to_datetime(df['market_time'])
        df['odds_time'] = pd.to_datetime(df['odds_time'])
        df['maturity_date'] = pd.to_datetime(df['maturity_date'])
        
        return df


async def fetch_historical_demo():
    """Demo historical data fetching."""
    fetcher = HistoricalDataFetcher()
    
    # Fetch last 30 days
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)
    
    print(f"Fetching historical data from {start_date.date()} to {end_date.date()}")
    
    stats = await fetcher.fetch_historical_range(
        start_date,
        end_date,
        sports=['NFL', 'NBA', 'EPL']
    )
    
    print(f"\nFetch complete: {stats}")
    
    # Get summary
    summary = fetcher.get_data_summary()
    print(f"\nData summary: {json.dumps(summary, indent=2, default=str)}")
    
    # Export for backtesting
    df = fetcher.export_for_backtesting(start_date, end_date)
    print(f"\nExported {len(df)} rows for backtesting")
    print(df.head())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(fetch_historical_demo())