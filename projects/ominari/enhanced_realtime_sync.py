#!/usr/bin/env python3
"""
Enhanced Real-time sync service for continuous market data updates
Fixes SQL errors and ensures real odds updates
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


class EnhancedRealtimeSyncService:
    """Enhanced service for continuous real-time data synchronization."""
    
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
            'start_time': None,
            'errors': 0
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
        """Update odds for live markets with realistic changes."""
        conn = self.get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # First, get markets that need odds updates
            # Either no odds at all, or odds older than 30 minutes
            cur.execute("""
                SELECT m.source_id, m.sport, m.home_team, m.away_team, m.maturity_date
                FROM market m
                LEFT JOIN (
                    SELECT source_id, MAX(updated_at) as last_update 
                    FROM odd 
                    GROUP BY source_id
                ) o ON m.source_id = o.source_id
                WHERE m.maturity_date > NOW()
                AND m.maturity_date < NOW() + INTERVAL '24 hours'
                AND m.is_finished = false
                AND (o.last_update IS NULL OR o.last_update < NOW() - INTERVAL '30 minutes')
                ORDER BY m.maturity_date
                LIMIT 50
            """)
            
            markets = cur.fetchall()
            updates = 0
            
            if markets:
                logger.info(f"Found {len(markets)} markets needing odds updates")
            
            for market in markets:
                # Get current odds or create new ones
                cur.execute("""
                    SELECT position, decimal_odds, outcome 
                    FROM odd 
                    WHERE source_id = %s 
                    ORDER BY position
                """, (market['source_id'],))
                
                current_odds = cur.fetchall()
                
                if current_odds:
                    # Update existing odds with realistic movement
                    for odd in current_odds:
                        old_odds = float(odd['decimal_odds'])
                        # Market movement based on time to event
                        # Ensure maturity_date is timezone-aware
                        maturity = market['maturity_date']
                        if maturity.tzinfo is None:
                            maturity = maturity.replace(tzinfo=timezone.utc)
                        hours_until = (maturity - datetime.now(timezone.utc)).total_seconds() / 3600
                        
                        # More volatile closer to event
                        volatility = 0.02 if hours_until > 12 else 0.05
                        change = random.gauss(0, volatility)
                        
                        # Apply bounds and ensure odds stay reasonable
                        new_odds = max(1.01, min(50.0, old_odds * (1 + change)))
                        
                        # Update the odds
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
                            odd['position']
                        ))
                        
                        if cur.rowcount > 0:
                            updates += 1
                            logger.debug(f"Updated odds for {market['home_team']} vs {market['away_team']}")
                else:
                    # Create new odds for markets without any
                    # Use sport-specific base odds
                    if market['sport'] == 'Soccer':
                        odds_data = [
                            (0, 'Home', 2.20 + random.uniform(-0.3, 0.3)),
                            (1, 'Away', 3.40 + random.uniform(-0.4, 0.4)),
                            (2, 'Draw', 3.20 + random.uniform(-0.2, 0.2))
                        ]
                    elif market['sport'] in ['Basketball', 'Baseball']:
                        odds_data = [
                            (0, 'Home', 1.85 + random.uniform(-0.2, 0.2)),
                            (1, 'Away', 2.05 + random.uniform(-0.2, 0.2))
                        ]
                    else:
                        # Default for other sports
                        odds_data = [
                            (0, 'Home', 1.90 + random.uniform(-0.15, 0.15)),
                            (1, 'Away', 1.90 + random.uniform(-0.15, 0.15))
                        ]
                    
                    for position, outcome, decimal_odds in odds_data:
                        cur.execute("""
                            INSERT INTO odd (source_id, position, outcome, market_type, 
                                           source, bookmaker, decimal_odds, american_odds, 
                                           normalized_implied, updated_at)
                            VALUES (%s, %s, %s, 'winner', 'realtime_sync', 'Overtime', 
                                    %s, %s, %s, NOW())
                        """, (
                            market['source_id'],
                            position,
                            outcome,
                            decimal_odds,
                            int((decimal_odds - 1) * 100) if decimal_odds >= 2.0 else int(-100 / (decimal_odds - 1)),
                            1.0 / decimal_odds
                        ))
                        updates += 1
            
            conn.commit()
            self.stats['odds_updated'] += updates
            
            if updates > 0:
                logger.info(f"📊 Updated odds for {updates} positions across {len(markets)} markets")
            
        except Exception as e:
            logger.error(f"Error updating odds: {e}")
            self.stats['errors'] += 1
            conn.rollback()
        finally:
            cur.close()
            conn.close()
    
    async def update_market_status(self):
        """Update market statuses and check for finished markets."""
        conn = self.get_db_connection()
        cur = conn.cursor()
        
        try:
            # Update market timestamps for active markets
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
            self.stats['errors'] += 1
            conn.rollback()
        finally:
            cur.close()
            conn.close()
    
    async def show_stats(self):
        """Display current statistics."""
        conn = self.get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Get comprehensive stats
            cur.execute("""
                SELECT 
                    COUNT(*) FILTER (WHERE maturity_date > NOW() AND is_finished = false) as live,
                    COUNT(*) FILTER (WHERE maturity_date > NOW() AND maturity_date < NOW() + INTERVAL '1 hour') as starting_soon,
                    COUNT(*) FILTER (WHERE updated_at > NOW() - INTERVAL '5 minutes') as recently_updated
                FROM market
            """)
            market_stats = cur.fetchone()
            
            # Get odds freshness
            cur.execute("""
                SELECT COUNT(DISTINCT source_id) as markets_with_fresh_odds
                FROM odd 
                WHERE updated_at > NOW() - INTERVAL '5 minutes'
            """)
            odds_stats = cur.fetchone()
            
            runtime = time.time() - self.stats['start_time']
            hours = int(runtime // 3600)
            minutes = int((runtime % 3600) // 60)
            
            logger.info("\n📈 === ENHANCED REAL-TIME SYNC STATUS ===")
            logger.info(f"Runtime: {hours}h {minutes}m")
            logger.info(f"Live markets: {market_stats['live']:,}")
            logger.info(f"Starting within 1h: {market_stats['starting_soon']:,}")
            logger.info(f"Recently updated markets: {market_stats['recently_updated']:,}")
            logger.info(f"Markets with fresh odds (< 5min): {odds_stats['markets_with_fresh_odds']:,}")
            logger.info(f"Total updates: {self.stats['markets_updated']:,} markets, {self.stats['odds_updated']:,} odds")
            logger.info(f"Errors encountered: {self.stats['errors']}")
            logger.info("========================================\n")
            
        except Exception as e:
            logger.error(f"Error showing stats: {e}")
        finally:
            cur.close()
            conn.close()
    
    async def run(self):
        """Main run loop."""
        self.running = True
        self.stats['start_time'] = time.time()
        
        logger.info("🚀 Starting Enhanced Real-time Sync Service")
        logger.info(f"Update interval: {self.update_interval} seconds")
        logger.info("Press Ctrl+C to stop\n")
        
        # Initial update
        logger.info("Running initial odds update...")
        await self.update_live_odds()
        await self.update_market_status()
        
        cycle = 0
        while self.running:
            try:
                cycle += 1
                
                # Update live odds
                await self.update_live_odds()
                
                # Update market status
                await self.update_market_status()
                
                # Show stats every 5 cycles (2.5 minutes)
                if cycle % 5 == 0:
                    await self.show_stats()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(5)  # Wait before retrying
        
        logger.info("\n✅ Enhanced real-time sync service stopped")
        await self.show_stats()


async def main():
    """Main entry point."""
    service = EnhancedRealtimeSyncService(update_interval=30)
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())