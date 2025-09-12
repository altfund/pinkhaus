#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ominari V1 System Runner
Simplified runner using v1 configuration: API for data/quotes, blockchain for trades only.
"""

import asyncio
import logging
import sys
import subprocess
import signal
from pathlib import Path

# Add project directory to path
sys.path.append(str(Path(__file__).parent))

from integrations_v1 import setup_v1_system
from config_v1 import v1_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ominari_v1.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class OminariV1Runner:
    """Manages the Ominari V1 trading system lifecycle."""
    
    def __init__(self):
        self.integrator = setup_v1_system()
        self.api_process = None
        self.monitor_process = None
        
    async def start_monitor_server(self):
        """Start the web monitor server in a subprocess."""
        logger.info("Starting web monitor...")
        
        # Start monitor as subprocess
        self.monitor_process = subprocess.Popen(
            [sys.executable, "web_monitor.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give it time to start
        await asyncio.sleep(3)
        
        if self.monitor_process.poll() is None:
            logger.info("Web monitor started successfully at http://localhost:8888/")
        else:
            logger.warning("Web monitor failed to start")
            
    def stop_monitor_server(self):
        """Stop the monitor server gracefully."""
        if self.monitor_process:
            logger.info("Stopping web monitor...")
            self.monitor_process.terminate()
            try:
                self.monitor_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Monitor didn't stop gracefully, forcing...")
                self.monitor_process.kill()
                
    async def run_continuous_v1_cycles(self):
        """Run continuous v1 system cycles."""
        logger.info("=== Starting V1 Continuous Operation ===")
        logger.info(f"Update intervals: API={v1_config.api_update_frequency}min, "
                   f"Blockchain={v1_config.blockchain_scan_frequency}min")
        
        while not shutdown_requested:
            try:
                # Run v1 cycle
                await self.integrator.run_v1_cycle()
                
                # Wait for next cycle (use shorter of the two frequencies)
                wait_time = min(
                    v1_config.api_update_frequency,
                    v1_config.blockchain_scan_frequency
                ) * 60
                
                logger.info(f"Waiting {wait_time}s until next cycle...")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Error in v1 cycle: {e}", exc_info=True)
                # Wait before retry
                await asyncio.sleep(60)
                
    async def health_check_loop(self):
        """Monitor system health."""
        while not shutdown_requested:
            try:
                # Check monitor health
                if self.monitor_process and self.monitor_process.poll() is not None:
                    logger.warning("Web monitor stopped")
                    
                # Simple v1 health check
                health_status = {
                    "api_configured": bool(v1_config.overtime_api_key),
                    "blockchain_configured": bool(v1_config.rpc_url),
                    "paper_trading": v1_config.paper_trading_enabled,
                    "live_trading": v1_config.live_trading_enabled
                }
                
                logger.debug(f"Health status: {health_status}")
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)
                
    async def run(self):
        """Main run method."""
        logger.info("=== Ominari V1 Trading System Starting ===")
        
        # Show v1 configuration
        print("\n" + "="*50)
        print("OMINARI V1 CONFIGURATION LOCKED")
        print("="*50)
        print("✓ Market Data: Overtime API")
        print("✓ Quotes: Overtime API")
        print("✓ Trade Monitoring: Blockchain")
        print("✗ GraphQL: Disabled")
        print(f"\nMode: {'Paper Trading' if v1_config.paper_trading_enabled else 'Live Trading'}")
        print(f"Update Frequency: {v1_config.api_update_frequency} minutes")
        print("="*50 + "\n")
        
        # Validate configuration
        if not v1_config.validate_v1_setup():
            logger.error("V1 configuration validation failed!")
            sys.exit(1)
            
        # Start web monitor
        await self.start_monitor_server()
        
        # Create tasks
        tasks = [
            asyncio.create_task(self.run_continuous_v1_cycles()),
            asyncio.create_task(self.health_check_loop())
        ]
        
        try:
            # Wait for shutdown or task failure
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            # Cleanup
            logger.info("Shutting down...")
            self.stop_monitor_server()
            
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
                    
        logger.info("=== Ominari V1 Trading System Stopped ===")


def main():
    """Entry point for v1 system."""
    runner = OminariV1Runner()
    
    try:
        # Run the async v1 system
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()