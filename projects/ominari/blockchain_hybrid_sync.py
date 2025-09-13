#!/usr/bin/env python3
"""
Blockchain Hybrid Sync Service
Integrates blockchain data reading with PostgreSQL storage in the hybrid system.
"""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import json
from blockchain_reader import BlockchainReader
from blockchain_postgres_writer import BlockchainPostgresWriter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BlockchainHybridSync:
    """Syncs blockchain data into the PostgreSQL hybrid system."""
    
    def __init__(self, networks: List[str] = None):
        self.networks = networks or ['optimism', 'arbitrum']
        self.readers = {}
        self.postgres_writer = BlockchainPostgresWriter()
        
        # Initialize readers for each network
        for network in self.networks:
            try:
                reader = BlockchainReader(network=network, db_path=f"blockchain_{network}.db")
                self.readers[network] = reader
                logger.info(f"Initialized {network} blockchain reader")
            except Exception as e:
                logger.error(f"Failed to initialize {network} reader: {e}")
    
    def sync_recent_markets(self, hours_back: int = 24) -> Dict[str, int]:
        """Sync recent markets from all networks to PostgreSQL."""
        total_synced = {}
        
        for network, reader in self.readers.items():
            try:
                logger.info(f"Syncing recent markets from {network} (last {hours_back} hours)")
                
                # Fetch recent markets using the reader's method
                markets = reader.fetch_recent_markets(hours_back)
                
                # Store each market in PostgreSQL
                synced_count = 0
                for market in markets:
                    # Convert to format expected by PostgreSQL writer
                    market_data = {
                        'market_address': market.get('address', ''),
                        'game_id': market.get('source_id', ''),
                        'game_label': f"{market.get('home_team', 'Home')} vs {market.get('away_team', 'Away')}",
                        'maturity_date': int(market.get('maturity_date', datetime.now()).timestamp()),
                        'tags': [0, 0],  # Default tags - would need proper mapping
                        'normalized_odds': market.get('normalized_odds', []),
                        'creation_block': 0,  # Would need to get from blockchain
                        'creation_tx': '',
                        'network': network
                    }
                    
                    if self.postgres_writer.store_market(market_data):
                        synced_count += 1
                
                total_synced[network] = synced_count
                logger.info(f"Synced {synced_count} markets from {network}")
                
            except Exception as e:
                logger.error(f"Error syncing {network}: {e}")
                total_synced[network] = 0
        
        return total_synced
    
    def sync_market_by_address(self, network: str, market_address: str) -> bool:
        """Sync a specific market by address."""
        if network not in self.readers:
            logger.error(f"No reader for network {network}")
            return False
        
        try:
            reader = self.readers[network]
            
            # This would require implementing get_market_by_address in BlockchainReader
            # For now, we'll use a placeholder
            logger.warning(f"Market-specific sync not yet implemented for {market_address}")
            return False
            
        except Exception as e:
            logger.error(f"Error syncing market {market_address} from {network}: {e}")
            return False
    
    def monitor_live_odds(self, market_addresses: List[str], 
                         network: str = 'optimism', 
                         interval_minutes: int = 5) -> None:
        """Monitor live odds for specific markets."""
        if network not in self.readers:
            logger.error(f"No reader for network {network}")
            return
        
        reader = self.readers[network]
        logger.info(f"Starting live odds monitoring for {len(market_addresses)} markets on {network}")
        
        async def monitor_loop():
            while True:
                for market_address in market_addresses:
                    try:
                        # Get current odds
                        odds = reader.get_current_odds(market_address)
                        
                        # Store each position's odds
                        for position, odds_data in odds.items():
                            success = self.postgres_writer.store_odds_snapshot(
                                market_address, position, odds_data, network
                            )
                            if success:
                                logger.debug(f"Stored odds snapshot for {market_address} position {position}")
                    
                    except Exception as e:
                        logger.error(f"Error monitoring {market_address}: {e}")
                
                # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)
        
        # Run the monitoring loop
        asyncio.run(monitor_loop())
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status across all networks."""
        status = {
            'networks': {},
            'postgres_stats': self.postgres_writer.get_statistics(),
            'last_sync': datetime.now(tz=timezone.utc).isoformat()
        }
        
        # Get stats from each blockchain reader
        for network, reader in self.readers.items():
            try:
                # Check connection
                is_connected = reader.check_connection()
                
                # Get basic stats (would need to implement in BlockchainReader)
                network_status = {
                    'connected': is_connected,
                    'network': network,
                    'chain_id': reader.config.get('chain_id', 'unknown'),
                    'latest_block': reader.w3.eth.block_number if is_connected else 0
                }
                
                status['networks'][network] = network_status
                
            except Exception as e:
                logger.error(f"Error getting status for {network}: {e}")
                status['networks'][network] = {
                    'connected': False,
                    'error': str(e)
                }
        
        return status
    
    def run_continuous_sync(self, sync_interval_minutes: int = 10):
        """Run continuous synchronization."""
        logger.info(f"Starting continuous sync every {sync_interval_minutes} minutes")
        
        async def sync_loop():
            while True:
                try:
                    # Sync recent markets
                    results = self.sync_recent_markets(hours_back=2)  # Check last 2 hours
                    
                    total_synced = sum(results.values())
                    if total_synced > 0:
                        logger.info(f"Continuous sync: {total_synced} markets synced across all networks")
                    
                    # Get and log status
                    status = self.get_sync_status()
                    logger.debug(f"Sync status: {json.dumps(status, indent=2, default=str)}")
                    
                except Exception as e:
                    logger.error(f"Error in continuous sync: {e}")
                
                # Wait for next sync
                await asyncio.sleep(sync_interval_minutes * 60)
        
        asyncio.run(sync_loop())
    
    def backfill_historical_data(self, days_back: int = 7):
        """Backfill historical blockchain data."""
        logger.info(f"Starting historical backfill for last {days_back} days")
        
        for network, reader in self.readers.items():
            try:
                # Calculate block range for backfill
                block_time = reader.config.get('block_time', 2)
                blocks_per_day = int(24 * 3600 / block_time)
                current_block = reader.w3.eth.block_number
                from_block = max(0, current_block - (days_back * blocks_per_day))
                
                logger.info(f"Backfilling {network} from block {from_block} to {current_block}")
                
                # Scan for markets in chunks
                chunk_size = 1000
                total_markets = 0
                
                for start_block in range(from_block, current_block, chunk_size):
                    end_block = min(start_block + chunk_size, current_block)
                    
                    # Scan for market creations
                    markets = reader.scan_market_creations(start_block, end_block)
                    
                    # Store in PostgreSQL
                    for market in markets:
                        if self.postgres_writer.store_market(market):
                            total_markets += 1
                    
                    logger.info(f"Backfilled blocks {start_block}-{end_block}: {len(markets)} markets")
                
                logger.info(f"Completed {network} backfill: {total_markets} total markets")
                
            except Exception as e:
                logger.error(f"Error backfilling {network}: {e}")


def main():
    """Example usage of the hybrid sync service."""
    # Initialize sync service
    sync_service = BlockchainHybridSync(networks=['optimism', 'arbitrum'])
    
    # Get current sync status
    status = sync_service.get_sync_status()
    print("Current sync status:")
    print(json.dumps(status, indent=2, default=str))
    
    # Sync recent markets
    print("\nSyncing recent markets...")
    results = sync_service.sync_recent_markets(hours_back=24)
    print(f"Sync results: {results}")
    
    # Optional: Start continuous sync (uncomment to run)
    # sync_service.run_continuous_sync(sync_interval_minutes=5)


if __name__ == "__main__":
    main()