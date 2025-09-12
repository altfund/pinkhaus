#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Aggregator for Multi-Source Collection
Combines data from REST API, GraphQL, and Blockchain sources.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import json
import sqlite3

# Import our data sources
from graphql_client import OvertimeGraphQLClient
from blockchain_reader import BlockchainReader
from database import SessionLocal
from models import Market, Odd
from sqlalchemy import func

logger = logging.getLogger(__name__)


@dataclass
class UnifiedMarket:
    """Unified market data from all sources."""
    # Core identifiers
    market_id: str  # Primary key in our DB
    game_id: str
    contract_address: Optional[str]
    
    # Market details
    sport: str
    league: str
    home_team: str
    away_team: str
    market_type: str
    maturity_date: datetime
    
    # Status
    is_open: bool
    is_resolved: bool
    is_canceled: bool
    final_result: Optional[int]
    
    # Odds data
    home_odds: Optional[float]
    away_odds: Optional[float]
    draw_odds: Optional[float]
    
    # Volume and liquidity
    total_volume: Optional[float]
    liquidity: Optional[float]
    
    # Source tracking
    sources: List[str]
    last_updated: datetime
    
    # Additional metadata
    tags: Optional[List[int]]
    metadata: Dict[str, Any]


class DataAggregator:
    """Aggregates market data from multiple sources."""
    
    def __init__(self, 
                 use_api: bool = True,
                 use_graphql: bool = True,
                 use_blockchain: bool = True,
                 networks: List[str] = ['optimism', 'arbitrum']):
        """Initialize data aggregator with specified sources."""
        self.use_api = use_api
        self.use_graphql = use_graphql
        self.use_blockchain = use_blockchain
        self.networks = networks
        
        # Initialize sources
        # API uses direct database access
            
        if use_graphql:
            self.graphql_clients = {
                network: OvertimeGraphQLClient(network)
                for network in networks
            }
            
        if use_blockchain:
            self.blockchain_readers = {
                network: BlockchainReader(network)
                for network in networks
            }
            
        # Cache for deduplication
        self.market_cache: Dict[str, UnifiedMarket] = {}
        
    async def get_active_markets(self, 
                               sport: Optional[str] = None,
                               league: Optional[str] = None,
                               hours_ahead: int = 48) -> List[UnifiedMarket]:
        """Get all active markets from all sources."""
        logger.info("Fetching active markets from all sources...")
        
        # Collect from all sources in parallel
        tasks = []
        
        if self.use_api:
            tasks.append(self._get_api_markets(sport, league, hours_ahead))
            
        if self.use_graphql:
            for network in self.networks:
                tasks.append(self._get_graphql_markets(network, sport, league))
                
        if self.use_blockchain:
            for network in self.networks:
                tasks.append(self._get_blockchain_markets(network))
                
        # Wait for all sources
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_markets = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Source {i} failed: {result}")
            elif isinstance(result, list):
                all_markets.extend(result)
                
        # Deduplicate and merge
        unified_markets = self._deduplicate_markets(all_markets)
        
        logger.info(f"Found {len(unified_markets)} unique markets from {len(all_markets)} total")
        return list(unified_markets.values())
        
    async def _get_api_markets(self, sport: Optional[str], 
                             league: Optional[str],
                             hours_ahead: int) -> List[UnifiedMarket]:
        """Get markets from REST API."""
        try:
            logger.info("Fetching from REST API...")
            
            # Get data from API
            cutoff_time = datetime.now(timezone.utc) + timedelta(hours=hours_ahead)
            
            # Use existing database session
            session = SessionLocal()
            
            # Query markets
            query = session.query(Market).filter(
                Market.is_open == True,
                Market.maturity_date > datetime.now(timezone.utc),
                Market.maturity_date <= cutoff_time
            )
            
            if sport:
                query = query.filter(Market.sport == sport)
            if league:
                query = query.filter(Market.league == league)
                
            markets = query.all()
            
            # Convert to unified format
            unified = []
            for market in markets:
                # Get latest odds
                latest_odd = session.query(Odd).filter(
                    Odd.market_id == market.market_id
                ).order_by(Odd.timestamp.desc()).first()
                
                unified.append(UnifiedMarket(
                    market_id=market.market_id,
                    game_id=market.game_id or market.market_id,
                    contract_address=None,  # API doesn't provide this
                    sport=market.sport,
                    league=market.league,
                    home_team=market.home_team,
                    away_team=market.away_team,
                    market_type=market.type or 'moneyline',
                    maturity_date=market.maturity_date,
                    is_open=market.is_open,
                    is_resolved=market.resolved,
                    is_canceled=market.cancelled,
                    final_result=market.final_result,
                    home_odds=latest_odd.home_odds if latest_odd else None,
                    away_odds=latest_odd.away_odds if latest_odd else None,
                    draw_odds=latest_odd.draw_odds if latest_odd else None,
                    total_volume=None,
                    liquidity=None,
                    sources=['api'],
                    last_updated=datetime.now(timezone.utc),
                    tags=None,
                    metadata={'api_market_id': market.market_id}
                ))
                
            session.close()
            logger.info(f"Got {len(unified)} markets from API")
            return unified
            
        except Exception as e:
            logger.error(f"API fetch failed: {e}")
            return []
            
    async def _get_graphql_markets(self, network: str,
                                 sport: Optional[str],
                                 league: Optional[str]) -> List[UnifiedMarket]:
        """Get markets from GraphQL."""
        try:
            logger.info(f"Fetching from GraphQL {network}...")
            
            client = self.graphql_clients[network]
            markets = await client.get_active_markets(sport=sport, league=league)
            
            # Convert to unified format
            unified = []
            for market in markets:
                # Parse odds from positions or odds data
                home_odds = None
                away_odds = None
                draw_odds = None
                
                # GraphQL market data structure varies
                if hasattr(market, 'positions') and market.positions:
                    # Extract odds from positions
                    pass
                    
                unified.append(UnifiedMarket(
                    market_id=f"{network}_{market.address}",
                    game_id=market.game_id,
                    contract_address=market.address,
                    sport=market.sport,
                    league=market.league,
                    home_team=market.home_team,
                    away_team=market.away_team,
                    market_type=market.market_type,
                    maturity_date=market.maturity_date,
                    is_open=True,  # Active markets query
                    is_resolved=False,
                    is_canceled=False,
                    final_result=None,
                    home_odds=home_odds,
                    away_odds=away_odds,
                    draw_odds=draw_odds,
                    total_volume=market.total_volume,
                    liquidity=market.liquidity,
                    sources=[f'graphql_{network}'],
                    last_updated=datetime.now(timezone.utc),
                    tags=None,
                    metadata={
                        'network': network,
                        'contract_address': market.address
                    }
                ))
                
            logger.info(f"Got {len(unified)} markets from GraphQL {network}")
            return unified
            
        except Exception as e:
            logger.error(f"GraphQL {network} fetch failed: {e}")
            return []
            
    async def _get_blockchain_markets(self, network: str) -> List[UnifiedMarket]:
        """Get markets from blockchain."""
        try:
            logger.info(f"Fetching from blockchain {network}...")
            
            reader = self.blockchain_readers[network]
            
            # Get markets from local blockchain database
            conn = sqlite3.connect(reader.db_path)
            
            # Query recent markets
            cutoff_timestamp = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp())
            
            query = """
                SELECT 
                    market_address, game_id, game_label, 
                    maturity_date, tags, normalized_odds
                FROM blockchain_markets
                WHERE network = ? 
                AND maturity_date > ?
                AND maturity_date < ?
                ORDER BY creation_block DESC
                LIMIT 100
            """
            
            current_time = int(datetime.now(timezone.utc).timestamp())
            future_time = current_time + (48 * 3600)  # 48 hours
            
            cursor = conn.execute(query, (network, current_time, future_time))
            rows = cursor.fetchall()
            
            unified = []
            for row in rows:
                # Parse game label for teams
                game_label = row[2] or ""
                teams = game_label.split(" vs ")
                home_team = teams[0] if len(teams) > 0 else "Unknown"
                away_team = teams[1] if len(teams) > 1 else "Unknown"
                
                # Parse tags
                tags = json.loads(row[4]) if row[4] else []
                
                unified.append(UnifiedMarket(
                    market_id=f"{network}_{row[0]}",
                    game_id=row[1],
                    contract_address=row[0],
                    sport="Unknown",  # Would need to decode from tags
                    league="Unknown",
                    home_team=home_team,
                    away_team=away_team,
                    market_type="moneyline",
                    maturity_date=datetime.fromtimestamp(row[3], tz=timezone.utc),
                    is_open=True,
                    is_resolved=False,
                    is_canceled=False,
                    final_result=None,
                    home_odds=None,  # Would need to fetch from contract
                    away_odds=None,
                    draw_odds=None,
                    total_volume=None,
                    liquidity=None,
                    sources=[f'blockchain_{network}'],
                    last_updated=datetime.now(timezone.utc),
                    tags=tags,
                    metadata={
                        'network': network,
                        'contract_address': row[0],
                        'normalized_odds': json.loads(row[5]) if row[5] else []
                    }
                ))
                
            conn.close()
            
            logger.info(f"Got {len(unified)} markets from blockchain {network}")
            return unified
            
        except Exception as e:
            logger.error(f"Blockchain {network} fetch failed: {e}")
            return []
            
    def _deduplicate_markets(self, markets: List[UnifiedMarket]) -> Dict[str, UnifiedMarket]:
        """Deduplicate and merge market data from multiple sources."""
        deduped = {}
        
        for market in markets:
            # Try to find matching market
            key = None
            
            # First, try contract address
            if market.contract_address:
                key = market.contract_address.lower()
                
            # Then try game_id
            elif market.game_id:
                # Check if we already have this game_id
                for existing_key, existing in deduped.items():
                    if existing.game_id == market.game_id:
                        key = existing_key
                        break
                        
                if not key:
                    key = market.game_id
                    
            # Finally, use market_id
            else:
                key = market.market_id
                
            # Merge or add
            if key in deduped:
                # Merge data from multiple sources
                existing = deduped[key]
                
                # Update sources
                for source in market.sources:
                    if source not in existing.sources:
                        existing.sources.append(source)
                        
                # Update odds if better data available
                if market.home_odds and not existing.home_odds:
                    existing.home_odds = market.home_odds
                if market.away_odds and not existing.away_odds:
                    existing.away_odds = market.away_odds
                if market.draw_odds and not existing.draw_odds:
                    existing.draw_odds = market.draw_odds
                    
                # Update volume/liquidity
                if market.total_volume:
                    existing.total_volume = market.total_volume
                if market.liquidity:
                    existing.liquidity = market.liquidity
                    
                # Keep latest update time
                if market.last_updated > existing.last_updated:
                    existing.last_updated = market.last_updated
                    
            else:
                deduped[key] = market
                
        return deduped
        
    async def sync_to_database(self, markets: List[UnifiedMarket]):
        """Sync unified markets to database."""
        logger.info(f"Syncing {len(markets)} markets to database...")
        
        session = SessionLocal()
        
        try:
            for market in markets:
                # Check if market exists
                existing = session.query(Market).filter(
                    Market.market_id == market.market_id
                ).first()
                
                if existing:
                    # Update existing market
                    existing.is_open = market.is_open
                    existing.resolved = market.is_resolved
                    existing.cancelled = market.is_canceled
                    existing.final_result = market.final_result
                    existing.updated_at = datetime.now(timezone.utc)
                    
                else:
                    # Create new market
                    new_market = Market(
                        market_id=market.market_id,
                        sport=market.sport,
                        league=market.league,
                        home_team=market.home_team,
                        away_team=market.away_team,
                        maturity_date=market.maturity_date,
                        is_open=market.is_open,
                        type=market.market_type,
                        game_id=market.game_id,
                        resolved=market.is_resolved,
                        cancelled=market.is_canceled,
                        final_result=market.final_result,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc)
                    )
                    session.add(new_market)
                    
                # Add odds if available
                if any([market.home_odds, market.away_odds, market.draw_odds]):
                    odd = Odd(
                        market_id=market.market_id,
                        home_odds=market.home_odds,
                        away_odds=market.away_odds,
                        draw_odds=market.draw_odds,
                        timestamp=market.last_updated
                    )
                    session.add(odd)
                    
            session.commit()
            logger.info("Database sync completed")
            
        except Exception as e:
            logger.error(f"Database sync failed: {e}")
            session.rollback()
            raise
        finally:
            session.close()
            
    def get_source_status(self) -> Dict[str, bool]:
        """Get status of each data source."""
        status = {}
        
        if self.use_api:
            try:
                session = SessionLocal()
                count = session.query(func.count(Market.market_id)).scalar()
                session.close()
                status['api'] = count > 0
            except:
                status['api'] = False
                
        if self.use_graphql:
            for network in self.networks:
                status[f'graphql_{network}'] = True  # Would need actual health check
                
        if self.use_blockchain:
            for network in self.networks:
                try:
                    reader = self.blockchain_readers[network]
                    status[f'blockchain_{network}'] = reader.w3.is_connected()
                except:
                    status[f'blockchain_{network}'] = False
                    
        return status


async def test_aggregator():
    """Test the data aggregator."""
    logger.info("Testing Data Aggregator...")
    
    # Initialize aggregator
    aggregator = DataAggregator(
        use_api=True,
        use_graphql=False,  # Skip for now unless GraphQL is set up
        use_blockchain=True,
        networks=['optimism']
    )
    
    # Get source status
    status = aggregator.get_source_status()
    logger.info(f"Source status: {status}")
    
    # Get active markets
    markets = await aggregator.get_active_markets(hours_ahead=24)
    
    if markets:
        logger.info(f"\nFound {len(markets)} active markets:")
        for market in markets[:5]:
            logger.info(f"\n{market.home_team} vs {market.away_team}")
            logger.info(f"  Sport: {market.sport}")
            logger.info(f"  League: {market.league}")
            logger.info(f"  Maturity: {market.maturity_date}")
            logger.info(f"  Sources: {', '.join(market.sources)}")
            logger.info(f"  Odds: H:{market.home_odds} A:{market.away_odds} D:{market.draw_odds}")
    else:
        logger.info("No active markets found")
        
    # Sync to database
    if markets:
        await aggregator.sync_to_database(markets[:10])  # Sync first 10
        logger.info("Synced markets to database")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_aggregator())