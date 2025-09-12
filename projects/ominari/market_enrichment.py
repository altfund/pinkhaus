#!/usr/bin/env python3
"""
Market Enrichment Service

Combines blockchain data with metadata to create fully enriched market data.
This replaces the need for external APIs by providing all necessary information
from on-chain sources and local enrichment.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

from enhanced_tag_mappings import tag_mapper, SportMetadata, LeagueMetadata
from team_metadata_service import TeamMetadataService, TeamMetadata
from blockchain_reader import BlockchainReader
from database_v2 import db_manager
from models import Market, Odd

logger = logging.getLogger(__name__)


@dataclass
class EnrichedMarket:
    """Fully enriched market data."""
    # Core identifiers
    market_id: str
    blockchain_address: str
    game_id: str
    network: str
    
    # Sport/League info
    sport: SportMetadata
    league: LeagueMetadata
    
    # Teams
    home_team: TeamMetadata
    away_team: TeamMetadata
    
    # Market details
    market_type: str
    positions: List[str]
    has_draw: bool
    
    # Timing
    starts_at: datetime
    created_at: datetime
    expires_at: datetime
    
    # Odds
    current_odds: Dict[str, float]
    opening_odds: Dict[str, float]
    
    # Additional data
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'market_id': self.market_id,
            'blockchain_address': self.blockchain_address,
            'game_id': self.game_id,
            'network': self.network,
            'sport': asdict(self.sport),
            'league': asdict(self.league),
            'home_team': asdict(self.home_team),
            'away_team': asdict(self.away_team),
            'market_type': self.market_type,
            'positions': self.positions,
            'has_draw': self.has_draw,
            'starts_at': self.starts_at.isoformat(),
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'current_odds': self.current_odds,
            'opening_odds': self.opening_odds,
            'metadata': self.metadata
        }


class MarketEnrichmentService:
    """Service for enriching blockchain market data."""
    
    def __init__(self):
        self.metadata_service = TeamMetadataService()
        self.readers = {}
        self._init_readers()
    
    def _init_readers(self):
        """Initialize blockchain readers."""
        for network in ['optimism', 'arbitrum']:
            try:
                self.readers[network] = BlockchainReader(network)
                logger.info(f"Initialized {network} reader")
            except Exception as e:
                logger.warning(f"Could not initialize {network} reader: {e}")
    
    def enrich_market(self, market_address: str, 
                     network: str = 'optimism') -> Optional[EnrichedMarket]:
        """
        Fully enrich a market from blockchain data.
        
        Args:
            market_address: Blockchain address of the market
            network: Network the market is on
            
        Returns:
            Fully enriched market data or None
        """
        reader = self.readers.get(network)
        if not reader:
            logger.error(f"No reader for network {network}")
            return None
        
        try:
            # Get market from blockchain database
            import sqlite3
            conn = sqlite3.connect(reader.db_path)
            market_data = conn.execute("""
                SELECT * FROM blockchain_markets
                WHERE market_address = ? AND network = ?
            """, (market_address, network)).fetchone()
            conn.close()
            
            if not market_data:
                logger.warning(f"Market {market_address} not found in database")
                return None
            
            # Parse basic data
            tags = json.loads(market_data[4]) if isinstance(market_data[4], str) else market_data[4]
            sport, league = tag_mapper.decode_tags(tags)
            
            # Get metadata
            sport_meta = tag_mapper.get_sport_metadata(tags[0] if tags else 0)
            league_meta = tag_mapper.get_league_metadata(tags[1] if len(tags) > 1 else 0)
            
            if not sport_meta or not league_meta:
                logger.warning(f"Could not get metadata for tags {tags}")
                return None
            
            # Parse teams
            game_label = market_data[2]  # game_label column
            parsed_label = tag_mapper.parse_game_label(game_label, sport_meta.sport_id)
            
            # Find team metadata
            home_team = self._get_or_create_team(
                parsed_label.get('home', ''),
                sport_meta.name,
                league_meta.name
            )
            
            away_team = self._get_or_create_team(
                parsed_label.get('away', ''),
                sport_meta.name,
                league_meta.name
            )
            
            # Get current odds
            current_odds = reader.get_current_odds(market_address)
            
            # Format odds
            formatted_odds = {
                'home': current_odds.get(0, {}).get('buy', 0),
                'away': current_odds.get(1, {}).get('buy', 0)
            }
            
            if sport_meta.has_draw:
                formatted_odds['draw'] = current_odds.get(2, {}).get('buy', 0)
            
            # Parse opening odds
            opening_odds_data = json.loads(market_data[5]) if market_data[5] else []
            opening_odds = {
                'home': opening_odds_data[0] if len(opening_odds_data) > 0 else 0,
                'away': opening_odds_data[1] if len(opening_odds_data) > 1 else 0
            }
            if sport_meta.has_draw and len(opening_odds_data) > 2:
                opening_odds['draw'] = opening_odds_data[2]
            
            # Create enriched market
            enriched = EnrichedMarket(
                market_id=market_data[0],  # market_address
                blockchain_address=market_data[0],
                game_id=market_data[1],  # game_id
                network=network,
                sport=sport_meta,
                league=league_meta,
                home_team=home_team,
                away_team=away_team,
                market_type='moneyline',
                positions=sport_meta.positions,
                has_draw=sport_meta.has_draw,
                starts_at=datetime.fromtimestamp(market_data[3], tz=timezone.utc),  # maturity_date
                created_at=datetime.fromisoformat(market_data[9]) if market_data[9] else datetime.now(timezone.utc),
                expires_at=datetime.fromtimestamp(market_data[3], tz=timezone.utc),
                current_odds=formatted_odds,
                opening_odds=opening_odds,
                metadata={
                    'creation_block': market_data[6],
                    'creation_tx': market_data[7],
                    'tags': tags,
                    'original_label': game_label
                }
            )
            
            return enriched
            
        except Exception as e:
            logger.error(f"Error enriching market {market_address}: {e}")
            return None
    
    def _get_or_create_team(self, team_name: str, sport: str, 
                           league: str) -> TeamMetadata:
        """Get team metadata or create placeholder."""
        # Try to find existing team
        team = self.metadata_service.find_team(team_name, sport, league)
        
        if team:
            return team
        
        # Create placeholder
        team_id = f"{league.lower()}_{team_name.lower().replace(' ', '_')}"
        
        placeholder = TeamMetadata(
            team_id=team_id,
            sport=sport,
            league=league,
            full_name=team_name,
            short_name=team_name,
            abbreviation=team_name[:3].upper(),
            aliases=[team_name]
        )
        
        # Store for future use
        self.metadata_service.add_team(placeholder)
        
        return placeholder
    
    def get_active_markets(self, sport: Optional[str] = None,
                          league: Optional[str] = None,
                          limit: int = 100) -> List[EnrichedMarket]:
        """
        Get active enriched markets.
        
        Args:
            sport: Optional sport filter
            league: Optional league filter
            limit: Maximum number of markets to return
            
        Returns:
            List of enriched markets
        """
        enriched_markets = []
        
        # Get markets from database
        with db_manager.get_db_session() as db:
            query = db.query(Market).filter(
                Market.is_finished == False,
                Market.starts_at > datetime.now(timezone.utc)
            )
            
            if sport:
                query = query.filter(Market.sport == sport)
            if league:
                query = query.filter(Market.league == league)
            
            markets = query.order_by(Market.starts_at).limit(limit).all()
            
            for market in markets:
                # Extract network from source
                if market.source.startswith('blockchain_'):
                    network = market.source.replace('blockchain_', '')
                    
                    # Enrich market
                    enriched = self.enrich_market(market.market_id, network)
                    
                    if enriched:
                        enriched_markets.append(enriched)
        
        return enriched_markets
    
    def search_markets(self, query: str, limit: int = 50) -> List[EnrichedMarket]:
        """
        Search for markets by team name or other criteria.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching enriched markets
        """
        results = []
        query_lower = query.lower()
        
        with db_manager.get_db_session() as db:
            # Search in team names
            from sqlalchemy import or_
            markets = db.query(Market).filter(
                or_(
                    Market.home_team.ilike(f'%{query}%'),
                    Market.away_team.ilike(f'%{query}%'),
                    Market.sport.ilike(f'%{query}%'),
                    Market.league.ilike(f'%{query}%')
                ),
                Market.is_finished == False
            ).limit(limit).all()
            
            for market in markets:
                if market.source.startswith('blockchain_'):
                    network = market.source.replace('blockchain_', '')
                    enriched = self.enrich_market(market.market_id, network)
                    if enriched:
                        results.append(enriched)
        
        return results


def test_enrichment():
    """Test the enrichment service."""
    service = MarketEnrichmentService()
    
    # Get some active markets
    markets = service.get_active_markets(limit=5)
    
    print("Active Markets:")
    print("-" * 80)
    
    for market in markets:
        print(f"\n{market.home_team.full_name} vs {market.away_team.full_name}")
        print(f"  Sport: {market.sport.name} ({market.league.name})")
        print(f"  Starts: {market.starts_at}")
        print(f"  Odds: Home {market.current_odds.get('home', 0):.2f} "
              f"Away {market.current_odds.get('away', 0):.2f}")
        if market.has_draw:
            print(f"        Draw {market.current_odds.get('draw', 0):.2f}")
        print(f"  Venue: {market.home_team.venue}")


if __name__ == "__main__":
    test_enrichment()