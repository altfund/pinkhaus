#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ominari System Runner
Integrates with the existing scheduler to run the complete trading system.
"""

import asyncio
import logging
import sys
import os
import subprocess
import signal
from datetime import datetime, timezone
from pathlib import Path

# Add project directory to path
sys.path.append(str(Path(__file__).parent))

from integrations import SystemIntegrator
from config import settings
from monitoring import initialize_monitoring

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ominari_system.log'),
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


class OminariSystemRunner:
    """Manages the Ominari trading system lifecycle."""
    
    def __init__(self):
        self.integrator = SystemIntegrator()
        self.api_process = None
        self.cycle_interval = 300  # 5 minutes between cycles
        
    async def start_api_server(self):
        """Start the FastAPI server in a subprocess."""
        logger.info("Starting API server...")
        
        # Start API server as subprocess
        self.api_process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, 'ENVIRONMENT': settings.environment}
        )
        
        # Give it time to start
        await asyncio.sleep(5)
        
        if self.api_process.poll() is None:
            logger.info("API server started successfully")
        else:
            logger.error("API server failed to start")
            raise RuntimeError("API server startup failed")
            
    def stop_api_server(self):
        """Stop the API server gracefully."""
        if self.api_process:
            logger.info("Stopping API server...")
            self.api_process.terminate()
            try:
                self.api_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("API server didn't stop gracefully, forcing...")
                self.api_process.kill()
                
    async def run_continuous_cycles(self):
        """Run continuous system cycles."""
        logger.info("Starting continuous system cycles")
        
        while not shutdown_requested:
            try:
                # Run full system cycle
                logger.info("Starting system cycle")
                await self.integrator.run_full_cycle()
                
                # Generate hourly reports
                if datetime.now(timezone.utc).minute < 5:
                    report = self.integrator.generate_daily_report()
                    logger.info(f"Generated report: {report['date']}")
                    
                # Wait for next cycle
                logger.info(f"Cycle complete, waiting {self.cycle_interval}s...")
                await asyncio.sleep(self.cycle_interval)
                
            except Exception as e:
                logger.error(f"Error in system cycle: {e}", exc_info=True)
                # Wait before retry
                await asyncio.sleep(60)
                
    async def health_check_loop(self):
        """Monitor system health."""
        while not shutdown_requested:
            try:
                # Check API health
                if self.api_process and self.api_process.poll() is not None:
                    logger.error("API server died, restarting...")
                    await self.start_api_server()
                    
                # Check system components
                health = self.integrator._get_system_health()
                unhealthy = [k for k, v in health.items() if not v]
                
                if unhealthy:
                    logger.warning(f"Unhealthy components: {unhealthy}")
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)
                
    async def run(self):
        """Main run method."""
        logger.info("=== Ominari Trading System Starting ===")
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Features: Paper={settings.features.paper_trading}, "
                   f"Live={settings.features.live_trading}")
        
        # Initialize monitoring
        if settings.monitoring.enabled:
            initialize_monitoring()
            
        # Start API server
        await self.start_api_server()
        
        # Create tasks
        tasks = [
            asyncio.create_task(self.run_continuous_cycles()),
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
            self.stop_api_server()
            
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
                    
            # Generate final report
            try:
                report = self.integrator.generate_daily_report()
                logger.info(f"Final report saved: {report['date']}")
            except Exception as e:
                logger.error(f"Failed to generate final report: {e}")
                
        logger.info("=== Ominari Trading System Stopped ===")


def main():
    """Entry point for scheduler integration."""
    runner = OminariSystemRunner()
    
    try:
        # Run the async system
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()