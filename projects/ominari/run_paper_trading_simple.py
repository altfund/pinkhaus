#!/usr/bin/env python3
"""Run paper trading system directly without liquidity complications"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from load_env import load_dotenv
load_dotenv()

# Set up environment
os.environ['PG_PORT'] = '5999'

from paper_trading_live import LivePaperTrader

async def main():
    """Run paper trading directly"""
    print("🚀 Starting Direct Paper Trading System")
    print("=" * 50)
    
    trader = LivePaperTrader()
    trader.min_edge = -1.0  # Allow negative edges temporarily
    
    print(f"✅ Min edge threshold: {trader.min_edge}%")
    print(f"✅ Kelly fraction: {trader.kelly_fraction}")
    print(f"✅ Max concurrent bets: {trader.max_concurrent_bets}")
    print()
    
    # Run the trading loop
    await trader.run_trading_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✋ Stopped by user")