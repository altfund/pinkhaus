#!/usr/bin/env python3
"""
Run just the portfolio heartbeat system
Useful for testing or running separately from trading
"""

import asyncio
import os
import sys
import argparse
import logging

# Load environment variables
from load_env import load_dotenv
load_dotenv()

# Set up environment
os.environ['PG_PORT'] = '5999'

from portfolio_heartbeat import PortfolioHeartbeat

async def main(interval_minutes: int = 60):
    """Run heartbeat with custom interval"""
    heartbeat = PortfolioHeartbeat()
    
    # Override the default interval if specified
    if interval_minutes != 60:
        print(f"⚡ Running with custom interval: {interval_minutes} minutes")
        # Monkey patch the interval
        original_run = heartbeat.run
        async def custom_run():
            heartbeat.is_running = True
            logging.info(f"🫀 Starting portfolio heartbeat system ({interval_minutes} minute interval)")
            
            # Send initial heartbeat
            await heartbeat.send_heartbeat()
            
            while heartbeat.is_running:
                try:
                    # Wait for custom interval
                    await asyncio.sleep(interval_minutes * 60)
                    
                    # Send heartbeat
                    await heartbeat.send_heartbeat()
                    
                except Exception as e:
                    logging.error(f"Heartbeat loop error: {e}")
                    await asyncio.sleep(60)
        
        heartbeat.run = custom_run
    
    try:
        await heartbeat.run()
    except KeyboardInterrupt:
        print("\n✋ Stopping heartbeat...")
        heartbeat.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run portfolio heartbeat system')
    parser.add_argument(
        '--interval', 
        type=int, 
        default=60,
        help='Heartbeat interval in minutes (default: 60)'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Send one heartbeat and exit'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.once:
        print("📊 Sending single portfolio heartbeat...")
        asyncio.run(PortfolioHeartbeat().send_heartbeat())
        print("✅ Done!")
    else:
        print(f"🫀 Starting Portfolio Heartbeat System")
        print(f"⏰ Interval: {args.interval} minutes")
        print(f"📢 Notifications will be sent to Discord")
        print(f"Press Ctrl+C to stop\n")
        
        asyncio.run(main(args.interval))