#!/usr/bin/env python3
"""Monitor blockchain events for Overtime markets in real-time"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
import time

# Set up database environment
os.environ['USE_POSTGRESQL'] = '1'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import and_

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlockchainEventMonitor:
    """Monitors blockchain events for Overtime V2 markets"""
    
    def __init__(self):
        self.chain = "optimism"
        self.polling_interval = 12  # seconds (Optimism block time)
        self.event_handlers = {}
        self.running = False
        
        # Load blockchain connections
        self.blockchain_connections = {}
        try:
            with open('blockchain_connections.json', 'r') as f:
                self.blockchain_connections = json.load(f)
                logger.info(f"Loaded {len(self.blockchain_connections)} blockchain connections")
        except FileNotFoundError:
            logger.error("No blockchain connections found")
        
        # Track last processed block
        self.last_block = self.load_last_block()
    
    def load_last_block(self) -> int:
        """Load last processed block from file"""
        try:
            with open('.last_block.json', 'r') as f:
                data = json.load(f)
                return data.get('last_block', 0)
        except:
            return 0
    
    def save_last_block(self, block_number: int):
        """Save last processed block to file"""
        with open('.last_block.json', 'w') as f:
            json.dump({'last_block': block_number}, f)
    
    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler for specific event type"""
        self.event_handlers[event_type] = handler
        logger.info(f"Registered handler for {event_type}")
    
    async def fetch_events(self, from_block: int, to_block: int) -> List[Dict]:
        """Fetch blockchain events in block range"""
        # In production, this would use web3.py to fetch real events
        # Simulated events for demonstration
        
        events = []
        
        # Simulate some events
        if from_block % 100 == 0:  # Every 100 blocks
            events.append({
                'event': 'MarketCreated',
                'address': '0x' + 'a' * 40,
                'blockNumber': from_block,
                'transactionHash': '0x' + 'b' * 64,
                'args': {
                    'marketId': '0x' + 'c' * 64,
                    'homeTeam': 'Team A',
                    'awayTeam': 'Team B',
                    'maturityDate': int(time.time()) + 86400
                }
            })
        
        if from_block % 50 == 0:  # Every 50 blocks
            events.append({
                'event': 'OddsUpdated',
                'address': '0x' + 'd' * 40,
                'blockNumber': from_block,
                'transactionHash': '0x' + 'e' * 64,
                'args': {
                    'marketId': '0x' + 'f' * 64,
                    'position': 0,  # home
                    'odds': 250  # 2.50 in basis points
                }
            })
        
        if from_block % 200 == 0:  # Every 200 blocks
            events.append({
                'event': 'MarketResolved',
                'address': '0x' + 'g' * 40,
                'blockNumber': from_block,
                'transactionHash': '0x' + 'h' * 64,
                'args': {
                    'marketId': '0x' + 'i' * 64,
                    'winningPosition': 1  # away won
                }
            })
        
        return events
    
    async def process_event(self, event: Dict):
        """Process a single blockchain event"""
        event_type = event['event']
        logger.info(f"Processing {event_type} event from block {event['blockNumber']}")
        
        # Call registered handler if exists
        if event_type in self.event_handlers:
            try:
                await self.event_handlers[event_type](event)
            except Exception as e:
                logger.error(f"Error handling {event_type}: {e}")
        
        # Store event in database
        await self.store_event(event)
    
    async def store_event(self, event: Dict):
        """Store blockchain event in database"""
        # In production, store events for audit trail
        event_data = {
            'event_type': event['event'],
            'block_number': event['blockNumber'],
            'transaction_hash': event['transactionHash'],
            'address': event['address'],
            'args': json.dumps(event['args']),
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Log for now
        logger.debug(f"Stored event: {event_data}")
    
    async def get_current_block(self) -> int:
        """Get current blockchain block number"""
        # In production, query blockchain RPC
        # Simulate with timestamp-based blocks
        return int(time.time() / self.polling_interval)
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting blockchain event monitor...")
        
        while self.running:
            try:
                # Get current block
                current_block = await self.get_current_block()
                
                # Process blocks since last check
                if self.last_block < current_block:
                    logger.info(f"Processing blocks {self.last_block + 1} to {current_block}")
                    
                    # Fetch events in batches
                    batch_size = 100
                    for start_block in range(self.last_block + 1, current_block + 1, batch_size):
                        end_block = min(start_block + batch_size - 1, current_block)
                        
                        events = await self.fetch_events(start_block, end_block)
                        
                        # Process each event
                        for event in events:
                            await self.process_event(event)
                    
                    # Update last block
                    self.last_block = current_block
                    self.save_last_block(current_block)
                
                # Wait before next check
                await asyncio.sleep(self.polling_interval)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(self.polling_interval)
    
    async def start(self):
        """Start monitoring"""
        self.running = True
        await self.monitor_loop()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        logger.info("Stopping blockchain event monitor")


# Event Handlers
async def handle_market_created(event: Dict):
    """Handle MarketCreated event"""
    args = event['args']
    logger.info(f"New market created: {args['homeTeam']} vs {args['awayTeam']}")
    
    # Store in database
    # In production, create/update market record

async def handle_odds_updated(event: Dict):
    """Handle OddsUpdated event"""
    args = event['args']
    position_map = {0: 'home', 1: 'away', 2: 'draw'}
    position = position_map.get(args['position'], 'unknown')
    odds = args['odds'] / 100  # Convert from basis points
    
    logger.info(f"Odds updated for market {args['marketId'][:8]}... - {position}: {odds}")
    
    # Update odds in database
    # In production, update odds records

async def handle_market_resolved(event: Dict):
    """Handle MarketResolved event"""
    args = event['args']
    position_map = {0: 'home', 1: 'away', 2: 'draw'}
    winner = position_map.get(args['winningPosition'], 'unknown')
    
    logger.info(f"Market resolved: {args['marketId'][:8]}... - Winner: {winner}")
    
    # Update market status and settle positions
    # In production, trigger settlement process


async def main():
    """Run blockchain event monitor"""
    monitor = BlockchainEventMonitor()
    
    # Register event handlers
    monitor.register_handler('MarketCreated', handle_market_created)
    monitor.register_handler('OddsUpdated', handle_odds_updated)
    monitor.register_handler('MarketResolved', handle_market_resolved)
    
    print("🔗 Blockchain Event Monitor")
    print(f"Chain: {monitor.chain}")
    print(f"Starting from block: {monitor.last_block}")
    print(f"Polling interval: {monitor.polling_interval}s")
    print("\nMonitoring for events... (Press Ctrl+C to stop)\n")
    
    try:
        await monitor.start()
    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
        monitor.stop()


if __name__ == "__main__":
    asyncio.run(main())