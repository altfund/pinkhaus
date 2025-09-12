#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Start Paper Trading
Quick script to start paper trading immediately.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project directory to path
sys.path.append(str(Path(__file__).parent))

from ominari_unified import OminariUnifiedSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def start_paper_trading():
    """Start paper trading with minimal configuration."""
    logger.info("🚀 Starting Ominari Paper Trading System")
    
    # Create system with paper trading focus
    config = {
        'intervals': {
            'data_collection': 300,      # 5 minutes
            'signal_generation': 300,    # 5 minutes (same as data)
            'paper_trading': 300,        # 5 minutes (trade frequently)
            'alpha_research': 3600,      # 1 hour
            'reporting': 1800,           # 30 minutes
        }
    }
    
    system = OminariUnifiedSystem(config)
    
    # Override intervals for faster paper trading
    system.intervals = config['intervals']
    
    # Initialize
    await system.initialize()
    
    logger.info("✅ Paper trading is now active!")
    logger.info("System will:")
    logger.info("  - Collect data every 5 minutes")
    logger.info("  - Generate signals every 5 minutes")
    logger.info("  - Execute paper trades every 5 minutes")
    logger.info("  - Generate reports every 30 minutes")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    
    # Run
    await system.run_continuous()


if __name__ == "__main__":
    try:
        asyncio.run(start_paper_trading())
    except KeyboardInterrupt:
        logger.info("Paper trading stopped by user")