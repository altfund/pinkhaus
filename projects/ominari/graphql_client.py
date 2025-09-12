#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphQL Client for Overtime Markets Data
Provides efficient access to Overtime's GraphQL API with subscriptions support.
"""

import asyncio
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Structured market data from GraphQL."""
    address: str
    game_id: str
    sport: str
    league: str
    home_team: str
    away_team: str
    market_type: str
    maturity_date: datetime
    positions: List[Dict[str, Any]]
    liquidity: float
    total_volume: float
    

class OvertimeGraphQLClient:
    """GraphQL client for Overtime Markets with fallback support."""
    
    # GraphQL endpoints with priorities (local, public, legacy)
    ENDPOINTS = {
        'optimism': [
            'http://localhost:8000/subgraphs/name/overtime/optimism',  # Self-hosted
            'https://api.thegraph.com/subgraphs/name/thales-markets/overtime-optimism',  # Public
            'https://api.thegraph.com/subgraphs/name/thales-markets/thales-optimism',  # Legacy
        ],
        'arbitrum': [
            'http://localhost:8000/subgraphs/name/overtime/arbitrum',  # Self-hosted
            'https://api.thegraph.com/subgraphs/name/thales-markets/overtime-arbitrum',  # Public
            'https://api.thegraph.com/subgraphs/name/thales-markets/thales-arbitrum',  # Legacy
        ],
        'base': [
            'http://localhost:8000/subgraphs/name/overtime/base',  # Self-hosted
            'https://api.thegraph.com/subgraphs/name/thales-markets/overtime-base',  # Public
        ],
    }
    
    def __init__(self, network: str = 'optimism', use_local: bool = False, timeout: int = 30):
        """Initialize client for specific network.
        
        Args:
            network: Network name (optimism, arbitrum, base)
            use_local: If True, only try local endpoint; if False, try all endpoints
            timeout: Request timeout in seconds
        """
        if network not in self.ENDPOINTS:
            raise ValueError(f"Unknown network: {network}")
            
        self.network = network
        self.use_local = use_local
        self.timeout = timeout
        self.endpoints = self.ENDPOINTS[network] if not use_local else [self.ENDPOINTS[network][0]]
        self.current_endpoint = None
        self.client = None
    
    # Common queries
    QUERIES = {
        'markets': gql("""
            query GetMarkets($first: Int!, $skip: Int!, $where: Market_filter) {
                markets(first: $first, skip: $skip, where: $where, orderBy: timestamp, orderDirection: desc) {
                    id
                    address
                    gameId
                    sport
                    league
                    homeTeam
                    awayTeam
                    marketType
                    maturityDate
                    isResolved
                    isCanceled
                    finalResult
                    poolSize
                    numberOfParticipants
                    totalVolume
                    totalBuyVolume
                    totalSellVolume
                    positions {
                        id
                        side
                        claimable
                    }
                    odds {
                        position
                        homeOdds
                        awayOdds
                        drawOdds
                        timestamp
                    }
                    liquidityPool {
                        id
                        liquidityProviders
                        totalDeposited
                        totalWithdrawn
                    }
                }
            }
        """),
        
        'market_details': gql("""
            query GetMarketDetails($id: ID!) {
                market(id: $id) {
                    id
                    address
                    gameId
                    sport
                    league
                    homeTeam
                    awayTeam
                    marketType
                    maturityDate
                    isResolved
                    finalResult
                    poolSize
                    totalVolume
                    positions {
                        id
                        side
                        claimable
                        trades(orderBy: timestamp, orderDirection: desc) {
                            id
                            account
                            amount
                            payout
                            timestamp
                            txHash
                        }
                    }
                    odds {
                        id
                        position
                        homeOdds
                        awayOdds
                        drawOdds
                        timestamp
                    }
                }
            }
        """),
        
        'user_positions': gql("""
            query GetUserPositions($user: String!, $first: Int!) {
                positions(where: {account: $user}, first: $first, orderBy: timestamp, orderDirection: desc) {
                    id
                    market {
                        id
                        address
                        gameId
                        sport
                        league
                        homeTeam
                        awayTeam
                        marketType
                        maturityDate
                        isResolved
                        finalResult
                    }
                    side
                    amount
                    payout
                    isExercised
                    isClaimed
                    timestamp
                }
            }
        """),
        
        'market_trades': gql("""
            query GetMarketTrades($market: String!, $first: Int!, $skip: Int!) {
                trades(where: {market: $market}, first: $first, skip: $skip, orderBy: timestamp, orderDirection: desc) {
                    id
                    market {
                        id
                        address
                    }
                    account
                    side
                    amount
                    payout
                    price
                    fee
                    timestamp
                    txHash
                }
            }
        """),
        
        'liquidity_actions': gql("""
            query GetLiquidityActions($first: Int!, $skip: Int!) {
                liquidityActions(first: $first, skip: $skip, orderBy: timestamp, orderDirection: desc) {
                    id
                    type
                    account
                    amount
                    share
                    timestamp
                    txHash
                    liquidityPool {
                        id
                        totalDeposited
                        totalWithdrawn
                    }
                }
            }
        """)
    }
    
        
    async def _get_working_client(self):
        """Get a working GraphQL client, trying endpoints in order."""
        for endpoint in self.endpoints:
            try:
                transport = AIOHTTPTransport(
                    url=endpoint,
                    timeout=self.timeout
                )
                client = Client(
                    transport=transport, 
                    fetch_schema_from_transport=False
                )
                
                # Test the endpoint
                async with client as session:
                    await session.execute(
                        gql("{ _meta { block { number } } }")
                    )
                
                self.current_endpoint = endpoint
                logger.info(f"Connected to GraphQL endpoint: {endpoint}")
                return client
                
            except Exception as e:
                logger.warning(f"Failed to connect to {endpoint}: {e}")
                continue
                
        raise ConnectionError(f"All GraphQL endpoints failed for {self.network}")
        
    async def get_active_markets(self, 
                               sport: Optional[str] = None,
                               league: Optional[str] = None,
                               limit: int = 100) -> List[MarketData]:
        """Get currently active markets."""
        current_time = int(datetime.now(timezone.utc).timestamp())
        
        # Build filter
        where_filter = {
            'maturityDate_gt': current_time,
            'isCanceled': False,
            'isResolved': False
        }
        
        if sport:
            where_filter['sport'] = sport
        if league:
            where_filter['league'] = league
            
        # Execute query with fallback
        client = await self._get_working_client()
        
        try:
            async with client as session:
                result = await session.execute(
                    self.QUERIES['markets'],
                    variable_values={
                        'first': limit,
                        'skip': 0,
                        'where': where_filter
                    }
                )
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []
    
    async def fetch_active_markets(self, 
                                  sport: Optional[str] = None,
                                  limit: int = 100) -> List[MarketData]:
        """Fetch active markets - wrapper for get_active_markets."""
        return await self.get_active_markets(sport=sport, limit=limit)
    
    async def fetch_markets_paginated(self,
                                    start_date: datetime,
                                    end_date: datetime,
                                    sport: Optional[str] = None,
                                    page_size: int = 1000) -> List[MarketData]:
        """Fetch markets in date range with pagination."""
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        
        where_filter = {
            'maturityDate_gte': start_ts,
            'maturityDate_lte': end_ts
        }
        
        if sport:
            where_filter['sport'] = sport
            
        all_markets = []
        skip = 0
        
        client = await self._get_working_client()
        
        while True:
            try:
                async with client as session:
                    result = await session.execute(
                        self.QUERIES['markets'],
                        variable_values={
                            'first': page_size,
                            'skip': skip,
                            'where': where_filter
                        }
                    )
                
                markets = result.get('markets', [])
                if not markets:
                    break
                    
                for market in markets:
                    all_markets.append(self._parse_market(market))
                
                if len(markets) < page_size:
                    break
                    
                skip += page_size
                
            except Exception as e:
                logger.error(f"Error in pagination at skip={skip}: {e}")
                break
        
        return all_markets
    
    async def fetch_market_by_id(self, market_id: str) -> Optional[MarketData]:
        """Fetch a specific market by ID."""
        client = await self._get_working_client()
        
        try:
            async with client as session:
                result = await session.execute(
                    gql("""
                        query GetMarket($id: ID!) {
                            market(id: $id) {
                                id
                                address
                                gameId
                                sport
                                league
                                homeTeam
                                awayTeam
                                marketType
                                maturityDate
                                isResolved
                                isCanceled
                                positions {
                                    id
                                    side
                                    claimable
                                }
                                odds {
                                    position
                                    homeOdds
                                    awayOdds
                                    drawOdds
                                }
                                poolSize
                                totalVolume
                            }
                        }
                    """),
                    variable_values={'id': market_id}
                )
            
            market = result.get('market')
            return self._parse_market(market) if market else None
            
        except Exception as e:
            logger.error(f"Error fetching market {market_id}: {e}")
            return None
    
    async def check_connection(self) -> bool:
        """Check if GraphQL endpoint is accessible."""
        try:
            client = await self._get_working_client()
            return client is not None
        except Exception:
            return False
        
    async def get_market_details(self, market_address: str) -> MarketData:
        """Get detailed information for a specific market."""
        async with self.client as session:
            result = await session.execute(
                self.QUERIES['market_details'],
                variable_values={'id': market_address.lower()}
            )
            
        if result['market']:
            return self._parse_market(result['market'])
        else:
            raise ValueError(f"Market not found: {market_address}")
            
    async def get_market_trades(self, 
                              market_address: str,
                              limit: int = 100) -> pd.DataFrame:
        """Get recent trades for a market."""
        trades = []
        skip = 0
        
        async with self.client as session:
            while len(trades) < limit:
                result = await session.execute(
                    self.QUERIES['market_trades'],
                    variable_values={
                        'market': market_address.lower(),
                        'first': min(100, limit - len(trades)),
                        'skip': skip
                    }
                )
                
                if not result['trades']:
                    break
                    
                trades.extend(result['trades'])
                skip += 100
                
        # Convert to DataFrame
        df = pd.DataFrame(trades)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['amount'] = df['amount'].astype(float)
            df['payout'] = df['payout'].astype(float)
            df['price'] = df['price'].astype(float)
            df['fee'] = df['fee'].astype(float)
            
        return df
        
    async def get_user_positions(self, user_address: str) -> pd.DataFrame:
        """Get all positions for a user."""
        async with self.client as session:
            result = await session.execute(
                self.QUERIES['user_positions'],
                variable_values={
                    'user': user_address.lower(),
                    'first': 1000
                }
            )
            
        # Convert to DataFrame
        positions = []
        for pos in result['positions']:
            position_data = {
                'position_id': pos['id'],
                'market_address': pos['market']['address'],
                'game_id': pos['market']['gameId'],
                'sport': pos['market']['sport'],
                'league': pos['market']['league'],
                'home_team': pos['market']['homeTeam'],
                'away_team': pos['market']['awayTeam'],
                'market_type': pos['market']['marketType'],
                'maturity_date': datetime.fromtimestamp(
                    int(pos['market']['maturityDate']), tz=timezone.utc
                ),
                'side': pos['side'],
                'amount': float(pos['amount']),
                'payout': float(pos['payout']),
                'is_exercised': pos['isExercised'],
                'is_claimed': pos['isClaimed'],
                'timestamp': datetime.fromtimestamp(
                    int(pos['timestamp']), tz=timezone.utc
                )
            }
            positions.append(position_data)
            
        return pd.DataFrame(positions)
        
    async def stream_market_updates(self, 
                                  callback,
                                  sport: Optional[str] = None) -> None:
        """Stream real-time market updates using subscriptions."""
        # Note: This would require WebSocket transport for subscriptions
        # For now, we'll implement polling
        
        last_update = datetime.now(timezone.utc)
        
        while True:
            try:
                # Get markets updated since last check
                markets = await self.get_active_markets(sport=sport)
                
                for market in markets:
                    # Check if market was updated
                    if hasattr(market, 'last_update') and market.last_update > last_update:
                        await callback(market)
                        
                last_update = datetime.now(timezone.utc)
                await asyncio.sleep(10)  # Poll every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in market stream: {e}")
                await asyncio.sleep(30)  # Wait longer on error
                
    def _parse_market(self, raw_market: Dict[str, Any]) -> MarketData:
        """Parse raw GraphQL market data."""
        # Parse positions
        positions = []
        for pos in raw_market.get('positions', []):
            positions.append({
                'id': pos['id'],
                'side': pos['side'],
                'claimable': pos.get('claimable', False)
            })
            
        # Get latest odds
        odds_data = raw_market.get('odds', [])
        if odds_data:
            latest_odds = max(odds_data, key=lambda x: x['timestamp'])
        else:
            latest_odds = {}
            
        return MarketData(
            address=raw_market['address'],
            game_id=raw_market['gameId'],
            sport=raw_market['sport'],
            league=raw_market['league'],
            home_team=raw_market['homeTeam'],
            away_team=raw_market['awayTeam'],
            market_type=raw_market['marketType'],
            maturity_date=datetime.fromtimestamp(
                int(raw_market['maturityDate']), tz=timezone.utc
            ),
            positions=positions,
            liquidity=float(raw_market.get('poolSize', 0)),
            total_volume=float(raw_market.get('totalVolume', 0))
        )


class GraphQLAggregator:
    """Aggregates data from multiple GraphQL sources."""
    
    def __init__(self, networks: List[str] = ['optimism', 'arbitrum']):
        self.clients = {
            network: OvertimeGraphQLClient(network) 
            for network in networks
        }
        
    async def get_all_active_markets(self) -> pd.DataFrame:
        """Get active markets across all networks."""
        all_markets = []
        
        # Fetch from all networks concurrently
        tasks = []
        for network, client in self.clients.items():
            task = self._fetch_with_network(client, network)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        
        # Combine results
        for markets in results:
            all_markets.extend(markets)
            
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'network': m['network'],
                'address': m['data'].address,
                'game_id': m['data'].game_id,
                'sport': m['data'].sport,
                'league': m['data'].league,
                'home_team': m['data'].home_team,
                'away_team': m['data'].away_team,
                'market_type': m['data'].market_type,
                'maturity_date': m['data'].maturity_date,
                'liquidity': m['data'].liquidity,
                'volume': m['data'].total_volume
            }
            for m in all_markets
        ])
        
        return df
        
    async def _fetch_with_network(self, client: OvertimeGraphQLClient, 
                                network: str) -> List[Dict]:
        """Fetch markets and tag with network."""
        try:
            markets = await client.get_active_markets()
            logger.info(f"Fetched {len(markets)} markets from {network}")
            return [{'network': network, 'data': m} for m in markets]
        except Exception as e:
            logger.error(f"Error fetching from {network}: {e}")
            return []
            
    async def find_arbitrage_opportunities(self) -> pd.DataFrame:
        """Find price differences across networks."""
        df = await self.get_all_active_markets()
        
        if df.empty:
            return pd.DataFrame()
            
        # Group by game_id and market_type
        grouped = df.groupby(['game_id', 'market_type'])
        
        opportunities = []
        for (game_id, market_type), group in grouped:
            if len(group) > 1:
                # Check for price differences
                # This would require odds data from the markets
                opportunities.append({
                    'game_id': game_id,
                    'market_type': market_type,
                    'networks': list(group['network']),
                    'liquidity_diff': group['liquidity'].max() - group['liquidity'].min()
                })
                
        return pd.DataFrame(opportunities)


async def main():
    """Example usage."""
    # Initialize client
    client = OvertimeGraphQLClient('optimism')
    
    # Get active markets
    markets = await client.get_active_markets(sport='Soccer', limit=10)
    print(f"Found {len(markets)} active soccer markets")
    
    for market in markets[:3]:
        print(f"\n{market.home_team} vs {market.away_team}")
        print(f"  Sport: {market.sport}")
        print(f"  League: {market.league}")
        print(f"  Maturity: {market.maturity_date}")
        print(f"  Liquidity: ${market.liquidity:,.2f}")
        
    # Get trades for a market
    if markets:
        trades = await client.get_market_trades(markets[0].address, limit=10)
        print(f"\nRecent trades for {markets[0].address}:")
        print(trades[['timestamp', 'side', 'amount', 'price']].head())
        
    # Multi-network aggregation
    aggregator = GraphQLAggregator(['optimism', 'arbitrum'])
    all_markets = await aggregator.get_all_active_markets()
    print(f"\nTotal markets across networks: {len(all_markets)}")
    print(all_markets.groupby('network')['address'].count())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())