#!/usr/bin/env python3
"""
Integrated trading system with portfolio heartbeat
Combines paper/real trading with hourly status updates
"""

import asyncio
import logging
import os
import sys

# Load environment variables from .env file
from load_env import load_dotenv
load_dotenv()

# Add project root to path  
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integrated_trading_system import IntegratedTradingSystem
from portfolio_heartbeat_robust import RobustPortfolioHeartbeat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedTradingWithHeartbeat:
    """Combined trading system with heartbeat monitoring"""
    
    def __init__(self):
        self.trading_system = IntegratedTradingSystem()
        self.heartbeat = RobustPortfolioHeartbeat()
        self.is_running = False
        
    async def start(self):
        """Start both trading and heartbeat systems"""
        self.is_running = True
        
        logger.info("🚀 Starting Integrated Trading System with Heartbeat")
        
        # Create tasks for both systems
        trading_task = asyncio.create_task(self.trading_system.start())
        heartbeat_task = asyncio.create_task(self.heartbeat.run_forever())
        
        # Run both concurrently
        try:
            await asyncio.gather(trading_task, heartbeat_task)
        except Exception as e:
            logger.error(f"Error in combined system: {e}")
            
    def stop(self):
        """Stop both systems"""
        self.is_running = False
        self.trading_system.is_running = False
        self.heartbeat.stop()
        logger.info("Stopping integrated trading system with heartbeat")


async def main():
    """Main entry point"""
    system = IntegratedTradingWithHeartbeat()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("\nShutdown requested...")
    finally:
        system.stop()
        logger.info("System stopped")


if __name__ == "__main__":
    asyncio.run(main())