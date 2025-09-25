#!/usr/bin/env python3
"""Start the blockchain sync daemon for real-time data collection"""

import asyncio
import logging
import sys
import os
from blockchain_sync_daemon import BlockchainSyncDaemon

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('blockchain_sync.log')
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Start the blockchain sync daemon."""
    logger.info("🚀 Starting Ominari Blockchain Sync Daemon...")
    
    # Configure daemon
    daemon = BlockchainSyncDaemon(
        networks=['optimism', 'arbitrum'],
        sync_interval=60,  # Check for new markets every 60 seconds
        odds_update_interval=300  # Update odds every 5 minutes
    )
    
    try:
        # Initialize
        await daemon.initialize()
        
        # Run daemon
        logger.info("✅ Blockchain sync daemon initialized successfully")
        logger.info("📡 Starting continuous sync...")
        
        await daemon.run()
        
    except KeyboardInterrupt:
        logger.info("⏹️  Shutting down daemon...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())