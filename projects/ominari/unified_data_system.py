#!/usr/bin/env python3
"""
Unified Data Access System

Provides a single interface for accessing data from:
- Historical SQLite (pre-2025 data)
- Live PostgreSQL (post-2025 data)
- Real-time blockchain
- External APIs
- Redis cache
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import redis
import asyncio
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources."""
    SQLITE_HISTORICAL = "sqlite_historical"
    POSTGRESQL_LIVE = "postgresql_live"
    BLOCKCHAIN_REALTIME = "blockchain_realtime"
    API_EXTERNAL = "api_external"
    REDIS_CACHE = "redis_cache"


@dataclass
class MarketData:
    """Unified market data structure."""
    market_id: str
    sport: str
    league: str
    home_team: str
    away_team: str
    starts_at: datetime
    is_finished: bool
    source: str
    odds: Optional[Dict[str, float]] = None
    liquidity: Optional[Dict[str, float]] = None
    last_updated: Optional[datetime] = None
    metadata: Optional[Dict] = None


class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    async def get_market(self, market_id: str) -> Optional[MarketData]:
        pass
    
    @abstractmethod
    async def get_markets_by_date(self, start_date: datetime, end_date: datetime) -> List[MarketData]:
        pass
    
    @abstractmethod
    async def get_odds(self, market_id: str) -> Dict:
        pass


class SQLiteProvider(DataProvider):
    """Provider for historical SQLite data."""
    
    def __init__(self, db_path: str = 'sport_odds.db'):
        self.db_path = db_path
        
    async def get_market(self, market_id: str) -> Optional[MarketData]:
        """Get market from SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, sport, league, home_team, away_team, 
                       starts_at, is_finished
                FROM market
                WHERE id = ?
            """, (market_id,))
            
            row = cursor.fetchone()
            if row:
                return MarketData(
                    market_id=row[0],
                    sport=row[1],
                    league=row[2],
                    home_team=row[3],
                    away_team=row[4],
                    starts_at=datetime.fromisoformat(row[5]),
                    is_finished=bool(row[6]),
                    source=DataSource.SQLITE_HISTORICAL.value
                )
        finally:
            conn.close()
        
        return None
    
    async def get_markets_by_date(self, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get markets within date range from SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        markets = []
        try:
            cursor.execute("""
                SELECT id, sport, league, home_team, away_team, 
                       starts_at, is_finished
                FROM market
                WHERE starts_at BETWEEN ? AND ?
                ORDER BY starts_at
                LIMIT 1000
            """, (start_date.isoformat(), end_date.isoformat()))
            
            for row in cursor.fetchall():
                markets.append(MarketData(
                    market_id=row[0],
                    sport=row[1],
                    league=row[2],
                    home_team=row[3],
                    away_team=row[4],
                    starts_at=datetime.fromisoformat(row[5]),
                    is_finished=bool(row[6]),
                    source=DataSource.SQLITE_HISTORICAL.value
                ))
        finally:
            conn.close()
        
        return markets
    
    async def get_odds(self, market_id: str) -> Dict:
        """Get odds history from SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        odds_data = {
            'market_id': market_id,
            'odds': [],
            'source': DataSource.SQLITE_HISTORICAL.value
        }
        
        try:
            cursor.execute("""
                SELECT bookmaker, outcome, odds, timestamp
                FROM odd
                WHERE market_id = ?
                ORDER BY timestamp DESC
                LIMIT 100
            """, (market_id,))
            
            for row in cursor.fetchall():
                odds_data['odds'].append({
                    'bookmaker': row[0],
                    'outcome': row[1],
                    'odds': row[2],
                    'timestamp': row[3]
                })
        finally:
            conn.close()
        
        return odds_data


class PostgreSQLProvider(DataProvider):
    """Provider for live PostgreSQL data."""
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
    
    async def get_market(self, market_id: str) -> Optional[MarketData]:
        """Get market from PostgreSQL."""
        session = self.Session()
        
        try:
            result = session.execute(text("""
                SELECT market_id, sport, league, home_team, away_team,
                       starts_at, is_finished, updated_at
                FROM blockchain_markets
                WHERE market_id = :market_id
            """), {"market_id": market_id})
            
            row = result.fetchone()
            if row:
                return MarketData(
                    market_id=row[0],
                    sport=row[1],
                    league=row[2],
                    home_team=row[3],
                    away_team=row[4],
                    starts_at=row[5],
                    is_finished=row[6],
                    source=DataSource.POSTGRESQL_LIVE.value,
                    last_updated=row[7]
                )
        finally:
            session.close()
        
        return None
    
    async def get_markets_by_date(self, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get markets within date range from PostgreSQL."""
        session = self.Session()
        markets = []
        
        try:
            result = session.execute(text("""
                SELECT market_id, sport, league, home_team, away_team,
                       starts_at, is_finished, updated_at
                FROM blockchain_markets
                WHERE starts_at BETWEEN :start_date AND :end_date
                ORDER BY starts_at
                LIMIT 1000
            """), {"start_date": start_date, "end_date": end_date})
            
            for row in result:
                markets.append(MarketData(
                    market_id=row[0],
                    sport=row[1],
                    league=row[2],
                    home_team=row[3],
                    away_team=row[4],
                    starts_at=row[5],
                    is_finished=row[6],
                    source=DataSource.POSTGRESQL_LIVE.value,
                    last_updated=row[7]
                ))
        finally:
            session.close()
        
        return markets
    
    async def get_odds(self, market_id: str) -> Dict:
        """Get current odds from PostgreSQL."""
        session = self.Session()
        
        odds_data = {
            'market_id': market_id,
            'odds': [],
            'source': DataSource.POSTGRESQL_LIVE.value
        }
        
        try:
            result = session.execute(text("""
                SELECT position, outcome, decimal_odds, buy_odds, 
                       sell_odds, liquidity, updated_at
                FROM blockchain_odds
                WHERE market_id = :market_id
                ORDER BY updated_at DESC
                LIMIT 10
            """), {"market_id": market_id})
            
            for row in result:
                odds_data['odds'].append({
                    'position': row[0],
                    'outcome': row[1],
                    'decimal_odds': row[2],
                    'buy_odds': row[3],
                    'sell_odds': row[4],
                    'liquidity': row[5],
                    'updated_at': row[6].isoformat()
                })
        finally:
            session.close()
        
        return odds_data


class BlockchainProvider(DataProvider):
    """Provider for real-time blockchain data."""
    
    def __init__(self, blockchain_reader):
        self.blockchain_reader = blockchain_reader
    
    async def get_market(self, market_id: str) -> Optional[MarketData]:
        """Get market from blockchain."""
        # This would interface with blockchain_reader.py
        market = await self.blockchain_reader.get_market_by_id(market_id)
        
        if market:
            return MarketData(
                market_id=market.market_id,
                sport=market.sport,
                league=market.league,
                home_team=market.home_team,
                away_team=market.away_team,
                starts_at=market.starts_at,
                is_finished=market.is_finished,
                source=DataSource.BLOCKCHAIN_REALTIME.value,
                odds={str(k): v for k, v in market.odds.items()},
                liquidity={str(k): v for k, v in market.liquidity.items()},
                metadata={'network': market.network}
            )
        return None
    
    async def get_markets_by_date(self, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get active markets from blockchain."""
        markets = await self.blockchain_reader.get_active_markets()
        
        # Filter by date range
        filtered = []
        for market in markets:
            if start_date <= market.starts_at <= end_date:
                filtered.append(MarketData(
                    market_id=market.market_id,
                    sport=market.sport,
                    league=market.league,
                    home_team=market.home_team,
                    away_team=market.away_team,
                    starts_at=market.starts_at,
                    is_finished=market.is_finished,
                    source=DataSource.BLOCKCHAIN_REALTIME.value,
                    odds={str(k): v for k, v in market.odds.items()},
                    liquidity={str(k): v for k, v in market.liquidity.items()}
                ))
        
        return filtered
    
    async def get_odds(self, market_id: str) -> Dict:
        """Get real-time odds from blockchain."""
        market = await self.blockchain_reader.get_market_by_id(market_id)
        
        if market:
            return {
                'market_id': market_id,
                'odds': market.odds,
                'liquidity': market.liquidity,
                'source': DataSource.BLOCKCHAIN_REALTIME.value,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        return {'market_id': market_id, 'odds': [], 'source': DataSource.BLOCKCHAIN_REALTIME.value}


class UnifiedDataSystem:
    """Unified interface for all data sources."""
    
    def __init__(self):
        # Initialize providers
        self.sqlite_provider = SQLiteProvider()
        
        # PostgreSQL (configure with environment variables)
        pg_url = "postgresql://ominari_user:ominari_2025_secure@localhost:5432/ominari_blockchain"
        self.pg_provider = PostgreSQLProvider(pg_url)
        
        # Redis cache
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Blockchain provider would be initialized with actual blockchain reader
        self.blockchain_provider = None  # BlockchainProvider(blockchain_reader)
        
        # Data routing configuration
        self.cutoff_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    
    async def get_market(self, market_id: str, use_cache: bool = True) -> Optional[MarketData]:
        """Get market data from appropriate source."""
        
        # 1. Check cache first
        if use_cache:
            cached = self._get_from_cache(f"market:{market_id}")
            if cached:
                return MarketData(**cached)
        
        # 2. Try blockchain for real-time data
        if self.blockchain_provider:
            market = await self.blockchain_provider.get_market(market_id)
            if market:
                self._save_to_cache(f"market:{market_id}", asdict(market), ttl=60)
                return market
        
        # 3. Try PostgreSQL for recent data
        market = await self.pg_provider.get_market(market_id)
        if market:
            self._save_to_cache(f"market:{market_id}", asdict(market), ttl=300)
            return market
        
        # 4. Fall back to SQLite for historical
        market = await self.sqlite_provider.get_market(market_id)
        if market:
            self._save_to_cache(f"market:{market_id}", asdict(market), ttl=3600)
            return market
        
        return None
    
    async def get_markets_by_date(self, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get markets from appropriate source based on date."""
        
        markets = []
        
        # Route based on date
        if end_date >= self.cutoff_date:
            # Recent data - use PostgreSQL and blockchain
            markets.extend(await self.pg_provider.get_markets_by_date(start_date, end_date))
            
            if self.blockchain_provider:
                blockchain_markets = await self.blockchain_provider.get_markets_by_date(start_date, end_date)
                # Deduplicate
                existing_ids = {m.market_id for m in markets}
                markets.extend([m for m in blockchain_markets if m.market_id not in existing_ids])
        
        if start_date < self.cutoff_date:
            # Historical data - use SQLite
            historical_markets = await self.sqlite_provider.get_markets_by_date(start_date, end_date)
            markets.extend(historical_markets)
        
        # Sort by date
        markets.sort(key=lambda m: m.starts_at)
        
        return markets
    
    async def get_odds(self, market_id: str) -> Dict:
        """Get odds data with automatic source selection."""
        
        # Check if market is recent
        market = await self.get_market(market_id)
        
        if market:
            if market.source == DataSource.BLOCKCHAIN_REALTIME.value:
                return await self.blockchain_provider.get_odds(market_id)
            elif market.source == DataSource.POSTGRESQL_LIVE.value:
                return await self.pg_provider.get_odds(market_id)
            else:
                return await self.sqlite_provider.get_odds(market_id)
        
        # Try all sources
        for provider in [self.blockchain_provider, self.pg_provider, self.sqlite_provider]:
            if provider:
                odds = await provider.get_odds(market_id)
                if odds and odds.get('odds'):
                    return odds
        
        return {'market_id': market_id, 'odds': [], 'source': 'not_found'}
    
    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Get data from Redis cache."""
        try:
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    def _save_to_cache(self, key: str, data: Dict, ttl: int = 300):
        """Save data to Redis cache."""
        try:
            self.redis_client.setex(key, ttl, json.dumps(data))
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def get_unified_market_view(self, market_id: str) -> Dict:
        """Get complete market view from all sources."""
        
        # Get base market data
        market = await self.get_market(market_id)
        if not market:
            return {'error': 'Market not found'}
        
        # Get odds from all sources
        odds_data = await self.get_odds(market_id)
        
        # Combine into unified view
        unified = {
            'market': asdict(market),
            'odds': odds_data,
            'sources_checked': [
                DataSource.REDIS_CACHE.value,
                DataSource.BLOCKCHAIN_REALTIME.value,
                DataSource.POSTGRESQL_LIVE.value,
                DataSource.SQLITE_HISTORICAL.value
            ],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return unified


async def demo_unified_system():
    """Demonstrate unified data system."""
    logger.info("🔄 Unified Data System Demo")
    logger.info("=" * 60)
    
    system = UnifiedDataSystem()
    
    # Test 1: Get a market (will route to appropriate source)
    logger.info("\n1. Getting market data...")
    market = await system.get_market("test_market_001")
    if market:
        logger.info(f"   Found: {market.home_team} vs {market.away_team}")
        logger.info(f"   Source: {market.source}")
    
    # Test 2: Get markets by date range
    logger.info("\n2. Getting markets for date range...")
    start = datetime(2024, 12, 1, tzinfo=timezone.utc)
    end = datetime(2025, 2, 1, tzinfo=timezone.utc)
    
    markets = await system.get_markets_by_date(start, end)
    logger.info(f"   Found {len(markets)} markets")
    
    # Group by source
    by_source = {}
    for market in markets:
        by_source[market.source] = by_source.get(market.source, 0) + 1
    
    for source, count in by_source.items():
        logger.info(f"   - {source}: {count} markets")
    
    # Test 3: Get unified view
    logger.info("\n3. Getting unified market view...")
    if markets:
        unified = await system.get_unified_market_view(markets[0].market_id)
        logger.info(f"   Market: {markets[0].market_id}")
        logger.info(f"   Sources checked: {len(unified['sources_checked'])}")
        logger.info(f"   Odds records: {len(unified['odds'].get('odds', []))}")


if __name__ == "__main__":
    asyncio.run(demo_unified_system())