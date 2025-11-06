#!/usr/bin/env python3
"""Unified data fetcher that combines blockchain and API sources for complete market coverage"""

import os
import requests
import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Set up database environment
os.environ['USE_POSTGRESQL'] = '1'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import and_, or_, func
from market_id_mapper import MarketIDMapper
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedDataFetcher:
    """Fetches and merges data from blockchain and API sources with blockchain-first priority"""
    
    def __init__(self, blockchain_first: bool = True):
        self.api_url = "https://api.overtime.io/overtime-v2/games-info"
        self.id_mapper = MarketIDMapper()
        self.blockchain_first = blockchain_first  # Prioritize blockchain data
        
        # Blockchain sources (highest priority)
        self.blockchain_sources = [
            'blockchain_optimism_v2', 
            'blockchain_arbitrum_v2', 
            'blockchain_v2_optimism',
            'blockchain_optimism_v1',
            'blockchain_live',
            'blockchain_v2'
        ]
        
        # API sources (lower priority, only used to supplement)
        self.api_sources = [
            'overtime_v2',
            'overtime_soccer', 
            'overtime_v2_public',
            'api_live',
            'api_import'
        ]
        
        # Data source priority configuration
        self.data_source_priority = {
            'blockchain': 1,    # Highest priority - real on-chain data
            'database': 2,      # Medium priority - cached blockchain + API data
            'api': 3           # Lowest priority - only supplement missing data
        }
        
        # Market filtering configuration
        self.supported_sports = os.getenv('SUPPORTED_SPORTS', 'soccer,football').split(',')
        self.supported_outcomes = os.getenv('SUPPORTED_OUTCOMES', 'home,away,draw').split(',')
        self.min_odds = float(os.getenv('MIN_ODDS', '1.01'))
        self.max_odds = float(os.getenv('MAX_ODDS', '100.0'))
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '50'))
        
        # Load blockchain connections if available
        self.blockchain_connections = {}
        try:
            with open('blockchain_connections.json', 'r') as f:
                self.blockchain_connections = json.load(f)
                logger.info(f"Loaded {len(self.blockchain_connections)} blockchain connections")
        except FileNotFoundError:
            logger.warning("No blockchain_connections.json found, will rely on ID matching")
    
    def is_valid_soccer_market(self, market: Dict) -> bool:
        """Check if market is a valid soccer win/loss/draw market"""
        # Check sport
        sport = market.get('sport', '').lower()
        league = market.get('league', '').lower()
        tournament = market.get('tournamentName', '').lower()
        
        is_soccer = any(
            sport_term in sport or sport_term in league or sport_term in tournament
            for sport_term in self.supported_sports
        )
        
        if not is_soccer:
            return False
        
        # Check we have valid teams
        home_team = market.get('home_team', '').strip()
        away_team = market.get('away_team', '').strip()
        
        if not home_team or not away_team:
            return False
        
        # Check odds are valid
        odds = market.get('odds', {})
        
        for outcome in self.supported_outcomes:
            if outcome in odds:
                odds_value = float(odds[outcome]) if odds[outcome] else 0
                if self.min_odds <= odds_value <= self.max_odds:
                    continue
                else:
                    return False
        
        # Must have at least home and away odds
        has_home = 'home' in odds and self.min_odds <= float(odds.get('home', 0)) <= self.max_odds
        has_away = 'away' in odds and self.min_odds <= float(odds.get('away', 0)) <= self.max_odds
        
        return has_home and has_away
    
    async def fetch_api_markets(self) -> List[Dict]:
        """Fetch current markets from Overtime API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, headers={'accept': 'application/json'}) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Handle Overtime V2 format (dict with hex market IDs as keys)
                        soccer_markets = []
                        
                        if isinstance(data, dict):
                            # New format: {hex_id: market_data, ...}
                            for market_id, market_data in data.items():
                                if not isinstance(market_data, dict):
                                    continue
                                
                                # Add market_id to the data
                                market_data['market_id'] = market_id
                                
                                # Check sport - look in tournamentName or tags
                                tournament = market_data.get('tournamentName', '').lower()
                                sport_type = market_data.get('sport', '').lower()
                                tags = market_data.get('tags', [])
                                
                                # Check if it's soccer
                                is_soccer = (
                                    'soccer' in tournament or 
                                    'football' in tournament or
                                    'soccer' in sport_type or
                                    'football' in sport_type or
                                    any('soccer' in str(tag).lower() or 'football' in str(tag).lower() for tag in tags)
                                )
                                
                                if is_soccer and not market_data.get('isGameFinished', True):
                                    # Extract teams from positionNames
                                    positions = market_data.get('positionNames', [])
                                    if len(positions) >= 2:
                                        market_data['home_team'] = positions[0]
                                        market_data['away_team'] = positions[1]
                                        
                                        # Extract odds if available
                                        market_data['odds'] = {}
                                        odds_values = market_data.get('odds', [])
                                        if len(odds_values) >= 2:
                                            market_data['odds']['home'] = odds_values[0]
                                            market_data['odds']['away'] = odds_values[1]
                                            if len(odds_values) >= 3:
                                                market_data['odds']['draw'] = odds_values[2]
                                        
                                        soccer_markets.append(market_data)
                        
                        elif isinstance(data, list):
                            # Old format support
                            for market in data:
                                if not isinstance(market, dict):
                                    continue
                                    
                                sport = market.get('sport', '').lower()
                                if 'soccer' in sport or 'football' in sport:
                                    soccer_markets.append(market)
                        
                        logger.info(f"Fetched {len(soccer_markets)} soccer markets from API")
                        return soccer_markets
                    else:
                        logger.error(f"API request failed with status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching API data: {e}")
            return []
    
    def fetch_database_markets(self) -> List[Dict]:
        """Fetch current markets from database (both blockchain and API sources)"""
        database_markets = []
        
        with db_manager.get_db_session() as db:
            # Get unfinished soccer markets from ALL sources in database
            # Note: Some markets have incorrect future dates (2025/2026), but we'll include them
            now = datetime.now(timezone.utc)
            
            # Get markets from all sources, not just blockchain
            all_sources = self.blockchain_sources + self.api_sources
            
            markets = db.query(Market).filter(
                and_(
                    Market.sport.ilike('%soccer%'),
                    Market.is_finished == False,
                    Market.maturity_date > now - timedelta(days=1)  # Include recently started
                )
            ).order_by(Market.maturity_date.asc()).all()
            
            for market in markets:
                # Get odds for this market
                odds = db.query(Odd).filter(
                    Odd.source_id == market.source_id
                ).order_by(Odd.updated_at.desc()).all()
                
                market_dict = {
                    'market_id': market.source_id,
                    'source': market.source,
                    'sport': market.sport,
                    'league': market.league_name,
                    'home_team': market.home_team,
                    'away_team': market.away_team,
                    'maturity_date': market.maturity_date.isoformat(),
                    'odds': {}
                }
                
                # Add odds
                for odd in odds:
                    outcome = odd.outcome.lower()
                    if outcome in ['home', 'away', 'draw']:
                        market_dict['odds'][outcome] = odd.decimal_odds or odd.american_odds
                
                if market_dict['odds']:  # Only add if we have odds
                    database_markets.append(market_dict)
        
        logger.info(f"Found {len(database_markets)} database markets with odds")
        return database_markets
    
    def merge_market_data(self, api_markets: List[Dict], database_markets: List[Dict]) -> List[Dict]:
        """Merge API and database markets using ID mapping"""
        merged_markets = {}
        id_to_key = {}  # Map normalized IDs to market keys
        
        # First, process all markets and create ID mappings
        all_markets = []
        for market in database_markets:
            market['original_source'] = 'database'
            all_markets.append(market)
        for market in api_markets:
            market['original_source'] = 'api'
            all_markets.append(market)
        
        # Group markets by normalized ID
        for market in all_markets:
            market_id = market.get('market_id', '') or market.get('source_id', '')
            if not market_id:
                continue
                
            # Get normalized ID
            core_id = self.id_mapper.extract_core_id(market_id)
            if not core_id:
                continue
            
            # Create a unique key for this market
            if core_id not in id_to_key:
                # First time seeing this core ID
                key = f"id_{core_id}"
                id_to_key[core_id] = key
                merged_markets[key] = {
                    'core_id': core_id,
                    'data_sources': [],
                    'all_ids': {},
                    'odds': {},
                    'blockchain_id': None,
                    'api_id': None
                }
            else:
                key = id_to_key[core_id]
            
            # Add this market's data to the merged entry
            merged = merged_markets[key]
            source = market['original_source']
            
            # Track data sources
            if source not in merged['data_sources']:
                merged['data_sources'].append(source)
            
            # Store the original ID
            merged['all_ids'][market.get('source', source)] = market_id
            
            # Set blockchain/API IDs
            if source == 'database' and market.get('source') in self.blockchain_sources:
                merged['blockchain_id'] = market_id
                merged['data_sources'] = ['blockchain' if s == 'database' else s for s in merged['data_sources']]
            elif source == 'api' or (source == 'database' and market.get('source') in self.api_sources):
                merged['api_id'] = market_id
                if 'database' in merged['data_sources'] and 'api' not in merged['data_sources']:
                    merged['data_sources'].append('api')
                
                # Check if we have a blockchain connection for this API market
                if market_id in self.blockchain_connections:
                    connection = self.blockchain_connections[market_id]
                    merged['blockchain_id'] = connection['blockchain_address']
                    merged['blockchain_address'] = connection['blockchain_address']
                    if 'blockchain' not in merged['data_sources']:
                        merged['data_sources'].append('blockchain')
                    logger.debug(f"Connected API market to blockchain: {connection['blockchain_address']}")
            
            # Merge market data (prefer first seen for basic fields)
            for field in ['sport', 'league', 'home_team', 'away_team', 'maturity_date']:
                if field in market and field not in merged:
                    merged[field] = market[field]
            
            # Merge odds (prefer blockchain if available)
            market_odds = market.get('odds', {})
            if market_odds and (not merged['odds'] or source == 'database'):
                merged['odds'].update(market_odds)
        
        # Convert back to list
        result = list(merged_markets.values())
        logger.info(f"Merged to {len(result)} total markets")
        
        # Log data source breakdown  
        blockchain_only = sum(1 for m in result if 'blockchain' in m['data_sources'] and len(m['data_sources']) == 1)
        api_only = sum(1 for m in result if 'api' in m['data_sources'] and len(m['data_sources']) == 1)
        both = sum(1 for m in result if len(m['data_sources']) > 1)
        
        logger.info(f"Data sources: {blockchain_only} blockchain-only, {api_only} api-only, {both} both")
        
        # Log sample connected markets
        connected = [m for m in result if len(m['data_sources']) > 1]
        if connected:
            logger.info(f"Sample connected markets:")
            for market in connected[:3]:
                logger.info(f"  {market.get('home_team')} vs {market.get('away_team')} - Sources: {market['data_sources']}")
        
        return result
    
    def _normalize_market_id(self, market_id: str) -> str:
        """Normalize market ID to match across sources"""
        # Remove common prefixes
        id_normalized = market_id
        prefixes = ['overtime_real_', 'blockchain_v2_', 'v2_', 'api_', 'live_api_', 'blockchain_']
        
        for prefix in prefixes:
            if id_normalized.startswith(prefix):
                id_normalized = id_normalized[len(prefix):]
                break
        
        # If it's a hex ID, normalize it
        if id_normalized.startswith('0x'):
            # Remove trailing zeros (padding)
            id_normalized = id_normalized.rstrip('0')
            # Ensure minimum length
            if len(id_normalized) < 10:
                id_normalized = market_id  # Keep original if too short
        
        return id_normalized
    
    def _get_market_key(self, market: Dict) -> str:
        """Generate unique key for market matching"""
        # First try to match by normalized ID
        market_id = market.get('market_id', '') or market.get('source_id', '')
        if market_id:
            normalized_id = self._normalize_market_id(market_id)
            if len(normalized_id) > 10:  # Valid normalized ID
                return f"id_{normalized_id}"
        
        # Fallback to team+date matching
        home = market.get('home_team', '').lower().strip()
        away = market.get('away_team', '').lower().strip()
        date = market.get('maturity_date', '').split('T')[0]  # Just date part
        return f"teams_{home}_{away}_{date}"
    
    async def fetch_all_markets(self) -> List[Dict]:
        """Fetch and merge market data with blockchain-first priority"""
        
        if self.blockchain_first:
            logger.info("🔗 Blockchain-first mode: Prioritizing on-chain data")
            
            # Step 1: Get all database markets (blockchain + cached API)
            database_markets = self.fetch_database_markets()
            
            # Step 2: Only fetch API data if needed to supplement blockchain data
            api_markets = []
            blockchain_market_count = sum(1 for m in database_markets if m.get('source') in self.blockchain_sources)
            
            if blockchain_market_count < 100:  # If we have few blockchain markets, supplement with API
                logger.info(f"Only {blockchain_market_count} blockchain markets found, supplementing with API data")
                api_markets = await self.fetch_api_markets()
            else:
                logger.info(f"Found {blockchain_market_count} blockchain markets, skipping API fetch")
        else:
            # Original behavior: fetch both sources
            api_markets = await self.fetch_api_markets()
            database_markets = self.fetch_database_markets()
        
        # Merge data with blockchain priority
        merged_markets = self.merge_market_data(api_markets, database_markets)
        
        # Filter for valid soccer trading markets with blockchain preference
        valid_markets = []
        blockchain_preferred = []
        api_fallback = []
        
        logger.info(f"Filtering {len(merged_markets)} markets for valid soccer markets...")
        
        for market in merged_markets:
            # Must be a valid soccer market
            if not self.is_valid_soccer_market(market):
                continue
                
            # Prioritize markets with blockchain data
            if 'blockchain' in market.get('data_sources', []):
                blockchain_preferred.append(market)
            else:
                api_fallback.append(market)
        
        # Combine with blockchain-first priority
        if self.blockchain_first:
            # Use blockchain markets primarily, only add API if we need more
            valid_markets = blockchain_preferred
            
            if len(valid_markets) < 50:  # If we need more markets, add API ones
                logger.info(f"Adding {len(api_fallback)} API markets to supplement {len(blockchain_preferred)} blockchain markets")
                valid_markets.extend(api_fallback)
            else:
                logger.info(f"Using {len(blockchain_preferred)} blockchain markets only")
        else:
            valid_markets = blockchain_preferred + api_fallback
        
        logger.info(f"Returning {len(valid_markets)} valid markets for trading")
        logger.info(f"  Blockchain markets: {len(blockchain_preferred)}")
        logger.info(f"  API markets: {len(api_fallback)}")
        
        return valid_markets
    
    def format_for_trading(self, markets: List[Dict]) -> List[Dict]:
        """Format markets for portfolio trading engine"""
        formatted_markets = []
        
        for market in markets:
            try:
                # Base market data with blockchain connectivity info
                base = {
                    'market_id': market.get('blockchain_id') or market.get('api_id') or market.get('market_id'),
                    'match_id': market.get('market_id', ''),
                    'source': 'unified',
                    'sport': 'Soccer',
                    'league': market.get('league', 'Unknown'),
                    'home_team': market['home_team'],
                    'away_team': market['away_team'],
                    'maturity_date': datetime.fromisoformat(market['maturity_date'].replace('Z', '+00:00')),
                    'data_sources': market.get('data_sources', []),
                    'blockchain_id': market.get('blockchain_id'),
                    'blockchain_address': market.get('blockchain_address'),
                    'api_id': market.get('api_id'),
                    'has_blockchain_data': 'blockchain' in market.get('data_sources', []),
                    'blockchain_priority': 'blockchain' in market.get('data_sources', [])
                }
                
                # Extract odds
                odds = market.get('odds', {})
                home_odds = float(odds.get('home', 0))
                away_odds = float(odds.get('away', 0))
                draw_odds = float(odds.get('draw', 0))  # Only use draw if available
                
                # Create separate entries for each supported outcome
                for outcome in self.supported_outcomes:
                    outcome_odds = 0
                    if outcome == 'home':
                        outcome_odds = home_odds
                    elif outcome == 'away':
                        outcome_odds = away_odds
                    elif outcome == 'draw':
                        outcome_odds = draw_odds
                    
                    # Only create entry if odds are valid and within range
                    if self.min_odds <= outcome_odds <= self.max_odds:
                        outcome_market = {**base}
                        outcome_market.update({
                            'position': outcome,
                            'normalized_outcome': outcome,
                            'odds': outcome_odds,
                            'home_odds': home_odds,
                            'draw_odds': draw_odds,
                            'away_odds': away_odds
                        })
                        formatted_markets.append(outcome_market)
                    
            except Exception as e:
                logger.warning(f"Failed to format market {market.get('home_team')} vs {market.get('away_team')}: {e}")
                continue
        
        return formatted_markets


async def main():
    """Test the unified data fetcher"""
    fetcher = UnifiedDataFetcher()
    
    print("🔄 Fetching unified market data...\n")
    
    # Fetch all markets
    markets = await fetcher.fetch_all_markets()
    
    print(f"📊 Found {len(markets)} total markets\n")
    
    # Show sample markets
    print("📋 Sample markets:")
    for i, market in enumerate(markets[:5]):
        print(f"\n{i+1}. {market['home_team']} vs {market['away_team']}")
        print(f"   League: {market.get('league', 'Unknown')}")
        print(f"   Time: {market['maturity_date']}")
        print(f"   Sources: {', '.join(market.get('data_sources', []))}")
        
        odds = market.get('odds', {})
        if odds:
            odds_str = ' / '.join([f"{k.capitalize()}: {v:.2f}" for k, v in odds.items() if v])
            print(f"   Odds: {odds_str}")
    
    # Format for trading
    print("\n\n🎯 Formatting for trading...")
    trading_markets = fetcher.format_for_trading(markets)
    print(f"✅ {len(trading_markets)} markets ready for trading")
    
    return trading_markets


if __name__ == "__main__":
    asyncio.run(main())