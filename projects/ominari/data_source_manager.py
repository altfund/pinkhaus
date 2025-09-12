#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Source Manager for Ominari
Provides a unified interface for data access with priority:
1. Local Graph Node (GraphQL)
2. Public Graph (GraphQL) 
3. Direct blockchain access
4. Overtime API (backup only)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod
import aiohttp
from enum import Enum

from graphql_client import OvertimeGraphQLClient, MarketData
from blockchain_reader import BlockchainReader
import os

logger = logging.getLogger(__name__)


class DataSourcePriority(Enum):
    """Priority levels for data sources."""
    LOCAL_GRAPH = 1
    PUBLIC_GRAPH = 2
    BLOCKCHAIN = 3
    API = 4


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    async def get_markets(self, sport: Optional[str] = None, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch market data."""
        pass
    
    @abstractmethod
    async def get_odds(self, market_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch odds for specific markets."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the data source is available."""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> DataSourcePriority:
        """Return the priority level of this source."""
        pass


class GraphQLDataSource(DataSource):
    """GraphQL data source (local or public)."""
    
    def __init__(self, client: OvertimeGraphQLClient, is_local: bool = False):
        self.client = client
        self.is_local = is_local
        
    async def get_markets(self, sport: Optional[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch markets from GraphQL."""
        try:
            # Use the client's existing methods
            if start_date and end_date:
                markets = await self.client.fetch_markets_paginated(
                    start_date=start_date,
                    end_date=end_date,
                    sport=sport
                )
            else:
                markets = await self.client.fetch_active_markets(sport=sport)
            
            return [self._market_to_dict(m) for m in markets]
        except Exception as e:
            logger.error(f"GraphQL fetch error: {e}")
            raise
    
    async def get_odds(self, market_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch odds for specific markets."""
        try:
            odds_data = []
            for market_id in market_ids:
                market = await self.client.fetch_market_by_id(market_id)
                if market and hasattr(market, 'positions'):
                    for pos in market.positions:
                        odds_data.append({
                            'market_id': market_id,
                            'position': pos.get('side'),
                            'odds': pos.get('odds', 0),
                            'liquidity': pos.get('liquidity', 0)
                        })
            return odds_data
        except Exception as e:
            logger.error(f"GraphQL odds fetch error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check GraphQL endpoint availability."""
        try:
            # Simple query to test connection
            await self.client.fetch_active_markets(limit=1)
            return True
        except Exception:
            return False
    
    @property
    def priority(self) -> DataSourcePriority:
        return DataSourcePriority.LOCAL_GRAPH if self.is_local else DataSourcePriority.PUBLIC_GRAPH
    
    def _market_to_dict(self, market: MarketData) -> Dict[str, Any]:
        """Convert MarketData to dictionary."""
        return {
            'source': 'graphql',
            'source_id': market.game_id,
            'address': market.address,
            'sport': market.sport,
            'league': market.league,
            'home_team': market.home_team,
            'away_team': market.away_team,
            'market_type': market.market_type,
            'maturity_date': market.maturity_date,
            'liquidity': market.liquidity,
            'total_volume': market.total_volume,
            'positions': market.positions
        }


class BlockchainDataSource(DataSource):
    """Direct blockchain data source."""
    
    def __init__(self, reader: BlockchainReader):
        self.reader = reader
        
    async def get_markets(self, sport: Optional[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch markets from blockchain."""
        try:
            # Convert async blockchain reader methods if needed
            markets = await asyncio.to_thread(
                self.reader.fetch_recent_markets,
                hours_back=24 if not start_date else None
            )
            
            # Filter by date range if provided
            if start_date or end_date:
                filtered = []
                for market in markets:
                    market_date = market.get('maturity_date')
                    if isinstance(market_date, str):
                        market_date = datetime.fromisoformat(market_date)
                    
                    if start_date and market_date < start_date:
                        continue
                    if end_date and market_date > end_date:
                        continue
                    filtered.append(market)
                markets = filtered
            
            # Filter by sport if provided
            if sport:
                markets = [m for m in markets if m.get('sport', '').lower() == sport.lower()]
            
            return markets
        except Exception as e:
            logger.error(f"Blockchain fetch error: {e}")
            raise
    
    async def get_odds(self, market_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch odds from blockchain - not implemented yet."""
        # TODO: Implement direct odds reading from blockchain
        return []
    
    async def health_check(self) -> bool:
        """Check blockchain connection."""
        try:
            return await asyncio.to_thread(self.reader.check_connection)
        except Exception:
            return False
    
    @property
    def priority(self) -> DataSourcePriority:
        return DataSourcePriority.BLOCKCHAIN


class APIDataSource(DataSource):
    """Overtime API data source (backup only)."""
    
    def __init__(self, api_key: str, base_url: str, network_id: str):
        self.api_key = api_key
        self.base_url = base_url
        self.network_id = network_id
        self.api_url = f"{base_url}/networks/{network_id}"
        
    async def get_markets(self, sport: Optional[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch markets from API."""
        try:
            headers = {"x-api-key": self.api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/markets",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
            
            # Convert API format to standard format
            markets = []
            for sport_key, leagues in data.items():
                if sport and sport_key.lower() != sport.lower():
                    continue
                    
                for league_id, league_markets in leagues.items():
                    for market in league_markets:
                        if isinstance(market, dict):
                            markets.append({
                                'source': 'api',
                                'source_id': market.get('gameId'),
                                'sport': sport_key,
                                'league': league_id,
                                'home_team': market.get('homeTeam'),
                                'away_team': market.get('awayTeam'),
                                'market_type': market.get('type', 'moneyline'),
                                'maturity_date': datetime.fromtimestamp(
                                    market.get('maturityDate', 0)
                                ),
                                'odds': market.get('odds', [])
                            })
            
            return markets
        except Exception as e:
            logger.error(f"API fetch error: {e}")
            raise
    
    async def get_odds(self, market_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch odds from API."""
        # API returns odds with markets, so this would need a separate implementation
        return []
    
    async def health_check(self) -> bool:
        """Check API availability."""
        try:
            headers = {"x-api-key": self.api_key}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
        except Exception:
            return False
    
    @property
    def priority(self) -> DataSourcePriority:
        return DataSourcePriority.API


class DataSourceManager:
    """Manages multiple data sources with fallback support."""
    
    def __init__(self):
        self.sources: List[DataSource] = []
        self._initialized = False
        
    async def initialize(self, network: str = 'optimism'):
        """Initialize all data sources."""
        logger.info(f"Initializing data sources for network: {network}")
        
        # 1. Local GraphQL (if available)
        try:
            local_client = OvertimeGraphQLClient(network=network, use_local=True)
            local_source = GraphQLDataSource(local_client, is_local=True)
            if await local_source.health_check():
                self.sources.append(local_source)
                logger.info("✅ Local GraphQL source available")
        except Exception as e:
            logger.warning(f"Local GraphQL not available: {e}")
        
        # 2. Public GraphQL
        try:
            public_client = OvertimeGraphQLClient(network=network, use_local=False)
            public_source = GraphQLDataSource(public_client, is_local=False)
            if await public_source.health_check():
                self.sources.append(public_source)
                logger.info("✅ Public GraphQL source available")
        except Exception as e:
            logger.warning(f"Public GraphQL not available: {e}")
        
        # 3. Direct blockchain
        try:
            blockchain_reader = BlockchainReader(network=network)
            blockchain_source = BlockchainDataSource(blockchain_reader)
            blockchain_source.is_available = await blockchain_source.health_check()
            self.sources.append(blockchain_source)
            if blockchain_source.is_available:
                logger.info("✅ Blockchain source available")
            else:
                logger.warning("Blockchain source initialized but not fully available")
        except Exception as e:
            logger.warning(f"Blockchain source not available: {e}")
        
        # 4. API (backup only)
        api_key = os.getenv("OVERTIME_API_KEY")
        if api_key:
            try:
                api_source = APIDataSource(
                    api_key=api_key,
                    base_url="https://api.overtime.io/overtime-v2",
                    network_id=os.getenv("OVERTIME_NETWORK_ID", "10")
                )
                if await api_source.health_check():
                    self.sources.append(api_source)
                    logger.info("✅ API source available (backup)")
            except Exception as e:
                logger.warning(f"API source not available: {e}")
        
        # Sort by priority
        self.sources.sort(key=lambda s: s.priority.value)
        self._initialized = True
        
        if not self.sources:
            raise Exception("No data sources available!")
        
        logger.info(f"Initialized {len(self.sources)} data sources")
    
    async def get_markets(self, sport: Optional[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         use_backup: bool = False) -> List[Dict[str, Any]]:
        """
        Fetch markets with automatic fallback.
        
        Args:
            sport: Filter by sport
            start_date: Start date filter
            end_date: End date filter
            use_backup: If True, will use API even if other sources work
        """
        if not self._initialized:
            await self.initialize()
        
        errors = []
        
        for source in self.sources:
            # Skip non-backup sources if use_backup is True
            if use_backup and source.priority != DataSourcePriority.API:
                continue
                
            try:
                logger.info(f"Fetching markets from {source.__class__.__name__}")
                markets = await source.get_markets(sport, start_date, end_date)
                logger.info(f"Successfully fetched {len(markets)} markets")
                return markets
            except Exception as e:
                error_msg = f"{source.__class__.__name__} failed: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)
                continue
        
        # All sources failed
        raise Exception(f"All data sources failed: {'; '.join(errors)}")
    
    async def get_odds(self, market_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch odds with automatic fallback."""
        if not self._initialized:
            await self.initialize()
        
        for source in self.sources:
            try:
                return await source.get_odds(market_ids)
            except Exception as e:
                logger.warning(f"{source.__class__.__name__} odds fetch failed: {e}")
                continue
        
        return []
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of all data sources."""
        sources_status = []
        for source in self.sources:
            try:
                available = await source.health_check()
            except Exception:
                available = False
            
            sources_status.append({
                'name': source.__class__.__name__,
                'priority': source.priority.name,
                'available': available
            })
        
        return {
            'initialized': self._initialized,
            'sources': sources_status
        }


# Example usage
async def main():
    """Example of using the data source manager."""
    manager = DataSourceManager()
    await manager.initialize()
    
    # Get status
    print("Data source status:")
    status = await manager.get_status()
    for source in status['sources']:
        print(f"  - {source['name']}: {'✅' if source['available'] else '❌'}")
    
    # Fetch markets
    try:
        markets = await manager.get_markets(sport='football')
        print(f"\nFetched {len(markets)} football markets")
    except Exception as e:
        print(f"Error fetching markets: {e}")


if __name__ == "__main__":
    asyncio.run(main())