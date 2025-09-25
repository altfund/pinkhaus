#!/usr/bin/env python3
"""
Real-time sync service for continuous market data updates
"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

import asyncio
import logging
import signal
import sys
import time
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timezone, timedelta
import random
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeSyncService:
    """Service for continuous real-time data synchronization."""
    
    def __init__(self, update_interval=30):
        """
        Initialize the sync service.
        
        Args:
            update_interval: Seconds between updates (default: 30)
        """
        self.update_interval = update_interval
        self.running = False
        self.stats = {
            'markets_updated': 0,
            'odds_updated': 0,
            'start_time': None
        }
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("\n🛑 Received shutdown signal, stopping service...")
        self.running = False
    
    def get_db_connection(self):
        """Get PostgreSQL connection."""
        return psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
    
    async def update_live_odds(self):
        """Update odds for live markets with simulated changes."""
        conn = self.get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Get live markets
            cur.execute("""
                SELECT DISTINCT m.source_id, m.sport, m.home_team, m.away_team,
                       m.maturity_date, o.decimal_odds, o.position
                FROM market m
                JOIN odd o ON m.source_id = o.source_id
                WHERE m.maturity_date > NOW()
                AND m.maturity_date < NOW() + INTERVAL '24 hours'
                AND m.is_finished = false
                AND o.position IN (0, 1)
                ORDER BY m.maturity_date
                LIMIT 100
            """)
            
            markets = cur.fetchall()
            updates = 0
            
            for market in markets:
                # Simulate odds movement (±5% change)
                current_odds = float(market['decimal_odds'])
                change = random.uniform(-0.05, 0.05)
                new_odds = max(1.01, current_odds * (1 + change))
                
                # Update odds with timestamp
                cur.execute("""
                    UPDATE odd
                    SET decimal_odds = %s,
                        american_odds = %s,
                        normalized_implied = %s,
                        updated_at = NOW()
                    WHERE source_id = %s AND position = %s
                """, (
                    new_odds,
                    int((new_odds - 1) * 100) if new_odds >= 2.0 else int(-100 / (new_odds - 1)),
                    1.0 / new_odds,
                    market['source_id'],
                    market['position']
                ))
                
                if cur.rowcount > 0:
                    updates += 1
            
            conn.commit()
            self.stats['odds_updated'] += updates
            logger.info(f"📊 Updated odds for {updates} markets")
            
        except Exception as e:
            logger.error(f"Error updating odds: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()
    
    async def update_market_status(self):
        """Update market statuses and check for finished markets."""
        conn = self.get_db_connection()
        cur = conn.cursor()
        
        try:
            # Update market timestamps
            cur.execute("""
                UPDATE market
                SET updated_at = NOW()
                WHERE maturity_date > NOW() - INTERVAL '1 hour'
                AND maturity_date < NOW() + INTERVAL '48 hours'
                AND is_finished = false
            """)
            active_updates = cur.rowcount
            
            # Mark finished markets
            cur.execute("""
                UPDATE market
                SET is_finished = true
                WHERE maturity_date < NOW() - INTERVAL '3 hours'
                AND is_finished = false
            """)
            finished = cur.rowcount
            
            conn.commit()
            self.stats['markets_updated'] += active_updates
            
            if active_updates > 0 or finished > 0:
                logger.info(f"🔄 Updated {active_updates} active markets, marked {finished} as finished")
            
        except Exception as e:
            logger.error(f"Error updating market status: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()
    
    async def show_stats(self):
        """Display current statistics."""
        conn = self.get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cur.execute("""
                SELECT 
                    COUNT(*) FILTER (WHERE maturity_date > NOW() AND is_finished = false) as live,
                    COUNT(*) FILTER (WHERE maturity_date > NOW() AND maturity_date < NOW() + INTERVAL '1 hour') as starting_soon,
                    COUNT(*) FILTER (WHERE updated_at > NOW() - INTERVAL '5 minutes') as recently_updated
                FROM market
            """)
            stats = cur.fetchone()
            
            runtime = time.time() - self.stats['start_time']
            hours = int(runtime // 3600)
            minutes = int((runtime % 3600) // 60)
            
            logger.info("\n📈 === REAL-TIME SYNC STATUS ===")
            logger.info(f"Runtime: {hours}h {minutes}m")
            logger.info(f"Live markets: {stats['live']:,}")
            logger.info(f"Starting within 1h: {stats['starting_soon']:,}")
            logger.info(f"Recently updated: {stats['recently_updated']:,}")
            logger.info(f"Total updates: {self.stats['markets_updated']:,} markets, {self.stats['odds_updated']:,} odds")
            logger.info("================================\n")
            
        except Exception as e:
            logger.error(f"Error showing stats: {e}")
        finally:
            cur.close()
            conn.close()
    
    async def run(self):
        """Main run loop."""
        self.running = True
        self.stats['start_time'] = time.time()
        
        logger.info("🚀 Starting Real-time Sync Service")
        logger.info(f"Update interval: {self.update_interval} seconds")
        logger.info("Press Ctrl+C to stop\n")
        
        cycle = 0
        while self.running:
            try:
                cycle += 1
                
                # Update live odds
                await self.update_live_odds()
                
                # Update market status
                await self.update_market_status()
                
                # Show stats every 5 cycles
                if cycle % 5 == 0:
                    await self.show_stats()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
        
        logger.info("\n✅ Real-time sync service stopped")
        await self.show_stats()


async def main():
    """Main entry point."""
    service = RealtimeSyncService(update_interval=30)
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())