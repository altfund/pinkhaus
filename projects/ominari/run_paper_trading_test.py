#!/usr/bin/env python3
"""Quick test of paper trading"""

import asyncio
import os
import sys

os.environ['PG_PORT'] = '5999'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from paper_trading_live import LivePaperTrader

async def main():
    print("Testing paper trading for 30 seconds...")
    trader = LivePaperTrader()
    
    # Run for 30 seconds
    task = asyncio.create_task(trader.run_trading_loop())
    await asyncio.sleep(30)
    
    # Stop
    trader.is_running = False
    await trader.stop()
    
    print("\nTest complete!")

if __name__ == "__main__":
    asyncio.run(main())