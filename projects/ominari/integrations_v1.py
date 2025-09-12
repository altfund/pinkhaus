#!/usr/bin/env python3
"""
V1 System Integrator - Simplified for v1 architecture
API for market data/quotes, blockchain for trade monitoring only.
"""

import asyncio
import logging
from typing import Dict, Optional

from data_aggregator import DataAggregator
from blockchain_reader import BlockchainReader
from signal_registry import SignalRegistry
from config_v1 import v1_config
from free_data_pull import get_all_overtime_markets, get_odds_api_markets

logger = logging.getLogger(__name__)

class V1SystemIntegrator:
    """Simplified v1 system integrator."""
    
    def __init__(self):
        self.config = v1_config
        self.data_aggregator = DataAggregator()
        self.blockchain_reader = BlockchainReader()
        self.signal_registry = SignalRegistry()
        
        # Log v1 configuration
        logger.info("=== V1 System Configuration ===")
        logger.info("Market Data: Overtime API")
        logger.info("Quotes: Overtime API")
        logger.info("Trade Monitoring: Blockchain")
        logger.info(f"Paper Trading: {self.config.paper_trading_enabled}")
        
    async def run_data_collection_cycle(self):
        """Collect market data from API only in v1."""
        logger.info("Starting v1 data collection (API only)")
        
        try:
            # Get markets from Overtime API
            logger.info("Fetching markets from Overtime API...")
            get_all_overtime_markets()
            
            # Optionally get odds from Odds API if key is available
            if self.config.overtime_api_key:
                logger.info("Fetching additional odds data...")
                get_odds_api_markets()
                
            logger.info("Data collection complete")
            
        except Exception as e:
            logger.error(f"Data collection error: {e}")
    
    async def run_trade_monitoring_cycle(self):
        """Monitor blockchain for executed trades only."""
        logger.info("Starting trade monitoring (blockchain)")
        
        try:
            # Get current block
            current_block = self.blockchain_reader.w3.eth.block_number
            from_block = max(0, current_block - 100)  # Look back 100 blocks
            
            # Scan for trade events
            trades_count = self.blockchain_reader.scan_trades(
                from_block=from_block,
                to_block=current_block
            )
            
            if trades_count:
                logger.info(f"Found {trades_count} on-chain trades")
            else:
                logger.debug("No on-chain trades in recent blocks")
                
        except Exception as e:
            logger.error(f"Trade monitoring error: {e}")
    
    async def run_v1_cycle(self):
        """Run a complete v1 cycle."""
        logger.info("=== Starting V1 Cycle ===")
        
        # 1. Collect market data from API
        if self.config.use_overtime_api:
            await self.run_data_collection_cycle()
        
        # 2. Monitor blockchain for trades
        if self.config.use_blockchain:
            await self.run_trade_monitoring_cycle()
        
        # 3. Generate signals (if enabled)
        if self.config.paper_trading_enabled or self.config.live_trading_enabled:
            logger.info("Signal generation would run here...")
            # Signal generation code here
        
        logger.info("=== V1 Cycle Complete ===")
    
    def get_quote(self, market_id: str) -> Optional[Dict]:
        """Get quote from Overtime API for v1."""
        # In v1, quotes come from Overtime API
        # This would make an API call to get current quote
        logger.debug(f"Getting quote for {market_id} from Overtime API")
        # Implementation would go here
        return None
    
    async def run_continuous(self, interval_minutes: int = None):
        """Run continuous v1 cycles."""
        interval = interval_minutes or self.config.api_update_frequency
        logger.info(f"Starting continuous v1 operation (interval: {interval} minutes)")
        
        while True:
            try:
                await self.run_v1_cycle()
                await asyncio.sleep(interval * 60)
            except KeyboardInterrupt:
                logger.info("Stopping v1 system...")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error


def setup_v1_system():
    """Setup and validate v1 system."""
    logger.info("Setting up Ominari V1 System")
    
    # Validate configuration
    if not v1_config.validate_v1_setup():
        raise ValueError("V1 configuration validation failed")
    
    # Show data flow
    import json
    logger.info("V1 Data Flow:")
    logger.info(json.dumps(v1_config.get_data_flow(), indent=2))
    
    return V1SystemIntegrator()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup and run v1 system
    integrator = setup_v1_system()
    asyncio.run(integrator.run_continuous())