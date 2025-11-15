#!/usr/bin/env python3
"""
Robust portfolio heartbeat system with error recovery and logging
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from load_env import load_dotenv
load_dotenv()

# Set up environment
os.environ['PG_PORT'] = '5999'

from portfolio_heartbeat import PortfolioHeartbeat

# Set up file logging
log_file = '/tmp/portfolio_heartbeat.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RobustPortfolioHeartbeat(PortfolioHeartbeat):
    """Enhanced heartbeat with error recovery"""
    
    def __init__(self):
        super().__init__()
        self.consecutive_failures = 0
        self.max_failures = 5
        
    async def send_heartbeat_safe(self):
        """Send heartbeat with error handling"""
        try:
            await self.send_heartbeat()
            self.consecutive_failures = 0  # Reset on success
            logger.info(f"✅ Heartbeat sent successfully at {datetime.now(timezone.utc)}")
        except Exception as e:
            self.consecutive_failures += 1
            logger.error(f"❌ Heartbeat failed (attempt {self.consecutive_failures}/{self.max_failures}): {e}")
            logger.error(traceback.format_exc())
            
            if self.consecutive_failures >= self.max_failures:
                logger.critical(f"🚨 Too many consecutive failures ({self.max_failures}), but continuing...")
                # Reset counter to keep trying
                self.consecutive_failures = 0
    
    async def run_forever(self, interval_minutes: int = 60):
        """Run heartbeat loop with robust error handling"""
        self.is_running = True
        logger.info(f"🫀 Starting robust heartbeat system (interval: {interval_minutes} minutes)")
        logger.info(f"📄 Logging to: {log_file}")
        
        # Send initial heartbeat
        await self.send_heartbeat_safe()
        
        while self.is_running:
            try:
                # Calculate next run time
                next_run = datetime.now(timezone.utc) + timedelta(minutes=interval_minutes)
                logger.info(f"⏰ Next heartbeat scheduled for: {next_run.strftime('%Y-%m-%d %H:%M UTC')}")
                
                # Wait for interval
                await asyncio.sleep(interval_minutes * 60)
                
                # Send heartbeat
                await self.send_heartbeat_safe()
                
            except asyncio.CancelledError:
                logger.info("Heartbeat cancelled by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in heartbeat loop: {e}")
                logger.error(traceback.format_exc())
                # Continue after a short delay
                await asyncio.sleep(60)


async def main():
    """Main entry point with signal handling"""
    heartbeat = RobustPortfolioHeartbeat()
    
    try:
        await heartbeat.run_forever(interval_minutes=60)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        heartbeat.is_running = False
        logger.info("Heartbeat system stopped")


if __name__ == "__main__":
    print("🫀 Robust Portfolio Heartbeat System")
    print(f"📄 Logs will be written to: {log_file}")
    print("⏰ Heartbeat interval: 60 minutes")
    print("🔁 Auto-recovery enabled")
    print("Press Ctrl+C to stop\n")
    
    asyncio.run(main())