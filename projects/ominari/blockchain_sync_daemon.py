#!/usr/bin/env python3
"""
Blockchain Sync Daemon

Continuously syncs sports market data from blockchain to local database.
Replaces the need for external APIs by directly reading from chain.
"""

import asyncio
import logging
import signal
import sys
import time
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set
import sqlite3
from pathlib import Path

from blockchain_reader import BlockchainReader, ChainSyncService
from enhanced_tag_mappings import tag_mapper
from team_metadata_service import TeamMetadataService
from database_v2 import db_manager
from models import Market, Odd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BlockchainSyncDaemon:
    """Daemon service for continuous blockchain synchronization."""
    
    def __init__(self, networks: List[str] = None, 
                 sync_interval: int = 60,
                 odds_update_interval: int = 300):
        """
        Initialize the sync daemon.
        
        Args:
            networks: List of networks to sync (default: ['optimism', 'arbitrum'])
            sync_interval: Seconds between checking for new markets
            odds_update_interval: Seconds between updating odds for active markets
        """
        self.networks = networks or ['optimism', 'arbitrum']
        self.sync_interval = sync_interval
        self.odds_update_interval = odds_update_interval
        self.running = False
        
        # Initialize services
        self.readers = {}
        self.sync_services = {}
        self.metadata_service = TeamMetadataService()
        
        # Track active markets
        self.active_markets: Set[str] = set()
        self.last_odds_update = {}
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Received shutdown signal, stopping daemon...")
        self.running = False
    
    async def initialize(self):
        """Initialize blockchain readers and sync services."""
        for network in self.networks:
            try:
                logger.info(f"Initializing {network} reader...")
                reader = BlockchainReader(network)
                self.readers[network] = reader
                
                # Initialize sync service
                sync_service = ChainSyncService(reader)
                self.sync_services[network] = sync_service
                
                logger.info(f"✅ {network} initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize {network}: {e}")
    
    async def sync_new_markets(self, network: str):
        """Sync new markets from blockchain."""
        reader = self.readers.get(network)
        sync_service = self.sync_services.get(network)
        
        if not reader or not sync_service:
            return
        
        try:
            current_block = reader.w3.eth.block_number
            
            # Scan for new markets
            markets = reader.scan_market_creations(
                from_block=sync_service.last_synced_block,
                to_block=current_block
            )
            
            logger.info(f"{network}: Found {len(markets)} new markets")
            
            # Process each market
            for market in markets:
                try:
                    # Decode tags
                    tags = json.loads(market['tags']) if isinstance(market['tags'], str) else market['tags']
                    sport, league = tag_mapper.decode_tags(tags)
                    
                    # Parse game label
                    parsed_label = tag_mapper.parse_game_label(
                        market['game_label'],
                        tags[0] if tags else 0
                    )
                    
                    # Enrich with metadata
                    enriched_market = {
                        'market_address': market['market_address'],
                        'game_id': market['game_id'],
                        'game_label': market['game_label'],
                        'sport': sport,
                        'league': league,
                        'home_team': parsed_label.get('home', ''),
                        'away_team': parsed_label.get('away', ''),
                        'maturity_date': datetime.fromtimestamp(market['maturity_date']),
                        'network': network,
                        'tags': tags
                    }
                    
                    # Further enrich with team metadata
                    enriched_market = self.metadata_service.enrich_market_data(enriched_market)
                    
                    # Store in our database
                    await self._store_market(enriched_market)
                    
                    # Add to active markets if not expired
                    if enriched_market['maturity_date'] > datetime.now(timezone.utc):
                        self.active_markets.add(market['market_address'])
                    
                except Exception as e:
                    logger.error(f"Error processing market {market.get('market_address')}: {e}")
            
            # Update last synced block
            sync_service.last_synced_block = current_block
            
        except Exception as e:
            logger.error(f"Error syncing markets on {network}: {e}")
    
    async def update_odds_for_active_markets(self):
        """Update odds for all active markets."""
        logger.info(f"Updating odds for {len(self.active_markets)} active markets")
        
        # Remove expired markets
        now = datetime.now(timezone.utc)
        expired = set()
        
        with db_manager.get_db_session() as db:
            for market_address in self.active_markets:
                market = db.query(Market).filter(
                    Market.market_id == market_address
                ).first()
                
                if market and market.starts_at < now:
                    expired.add(market_address)
        
        self.active_markets -= expired
        logger.info(f"Removed {len(expired)} expired markets")
        
        # Update odds for remaining markets
        for network, reader in self.readers.items():
            markets_on_network = []
            
            # Get markets for this network
            with db_manager.get_db_session() as db:
                markets = db.query(Market).filter(
                    Market.source == f'blockchain_{network}',
                    Market.market_id.in_(self.active_markets)
                ).all()
                markets_on_network = [m.market_id for m in markets]
            
            # Update odds for each market
            for market_address in markets_on_network:
                try:
                    odds_data = reader.get_current_odds(market_address)
                    
                    if odds_data:
                        await self._store_odds(market_address, odds_data, network)
                        
                except Exception as e:
                    logger.error(f"Error updating odds for {market_address}: {e}")
                
                # Rate limiting
                await asyncio.sleep(0.1)
    
    async def _store_market(self, market_data: Dict):
        """Store market in database."""
        try:
            with db_manager.get_db_session() as db:
                # Check if market exists
                existing = db.query(Market).filter(
                    Market.market_id == market_data['market_address']
                ).first()
                
                if not existing:
                    market = Market(
                        market_id=market_data['market_address'],
                        source=f"blockchain_{market_data['network']}",
                        source_id=market_data['game_id'],
                        sport=market_data['sport'],
                        league=market_data['league'],
                        home_team=market_data['home_team'],
                        away_team=market_data['away_team'],
                        market_type='moneyline',
                        starts_at=market_data['maturity_date'],
                        is_finished=False,
                        metadata=json.dumps({
                            'game_label': market_data['game_label'],
                            'tags': market_data['tags'],
                            'network': market_data['network'],
                            'home_team_metadata': market_data.get('home_team'),
                            'away_team_metadata': market_data.get('away_team')
                        })
                    )
                    db.add(market)
                    db.commit()
                    
                    logger.info(f"Stored new market: {market_data['game_label']}")
                    
        except Exception as e:
            logger.error(f"Error storing market: {e}")
    
    async def _store_odds(self, market_address: str, odds_data: Dict, network: str):
        """Store odds update in database."""
        try:
            with db_manager.get_db_session() as db:
                # Get market
                market = db.query(Market).filter(
                    Market.market_id == market_address
                ).first()
                
                if not market:
                    return
                
                # Map position odds
                # Position 0 = home, 1 = away, 2 = draw
                home_odds = odds_data.get(0, {}).get('buy', 0)
                away_odds = odds_data.get(1, {}).get('buy', 0)
                draw_odds = odds_data.get(2, {}).get('buy', 0) if len(odds_data) > 2 else None
                
                if home_odds > 0 and away_odds > 0:
                    odd = Odd(
                        market_id=market.market_id,
                        bookmaker_id=f"blockchain_{network}",
                        home_odds=home_odds,
                        away_odds=away_odds,
                        draw_odds=draw_odds,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(odd)
                    db.commit()
                    
                    logger.debug(f"Updated odds for {market_address}: {home_odds:.2f} / {away_odds:.2f}")
                    
        except Exception as e:
            logger.error(f"Error storing odds: {e}")
    
    async def run(self):
        """Main daemon loop."""
        logger.info("Starting Blockchain Sync Daemon...")
        
        # Initialize
        await self.initialize()
        
        if not self.readers:
            logger.error("No blockchain readers initialized, exiting...")
            return
        
        self.running = True
        logger.info("✅ Daemon started successfully")
        logger.info(f"Networks: {', '.join(self.networks)}")
        logger.info(f"Sync interval: {self.sync_interval}s")
        logger.info(f"Odds update interval: {self.odds_update_interval}s")
        
        # Main loop
        last_sync = {}
        last_odds_update = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Sync new markets
                for network in self.networks:
                    if network not in last_sync or \
                       current_time - last_sync[network] >= self.sync_interval:
                        logger.info(f"Syncing new markets on {network}...")
                        await self.sync_new_markets(network)
                        last_sync[network] = current_time
                
                # Update odds
                if current_time - last_odds_update >= self.odds_update_interval:
                    await self.update_odds_for_active_markets()
                    last_odds_update = current_time
                
                # Sleep for a bit
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error
        
        logger.info("Daemon stopped")
    
    def get_stats(self) -> Dict:
        """Get daemon statistics."""
        stats = {
            'running': self.running,
            'networks': self.networks,
            'active_markets': len(self.active_markets),
            'readers': {
                network: reader.check_connection() 
                for network, reader in self.readers.items()
            }
        }
        
        # Get market counts
        with db_manager.get_db_session() as db:
            for network in self.networks:
                count = db.query(Market).filter(
                    Market.source == f'blockchain_{network}'
                ).count()
                stats[f'{network}_markets'] = count
        
        return stats


async def main():
    """Run the daemon."""
    # Configuration from environment or defaults
    networks = os.getenv('BLOCKCHAIN_NETWORKS', 'optimism,arbitrum').split(',')
    sync_interval = int(os.getenv('SYNC_INTERVAL', '60'))
    odds_interval = int(os.getenv('ODDS_UPDATE_INTERVAL', '300'))
    
    # Create and run daemon
    daemon = BlockchainSyncDaemon(
        networks=networks,
        sync_interval=sync_interval,
        odds_update_interval=odds_interval
    )
    
    await daemon.run()


if __name__ == "__main__":
    import os
    asyncio.run(main())