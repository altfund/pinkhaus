#!/usr/bin/env python3
"""
Start trading system with portfolio heartbeat
This wraps the integrated trading system with hourly updates
"""

import asyncio
import os
import sys
import logging

# Load environment variables from .env file
from load_env import load_dotenv
load_dotenv()

# Run the integrated trading with heartbeat
from integrated_trading_with_heartbeat import main

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting Integrated Trading System with Portfolio Heartbeat")
    print("📊 You will receive hourly portfolio updates via Discord")
    print("Press Ctrl+C to stop\n")
    
    asyncio.run(main())