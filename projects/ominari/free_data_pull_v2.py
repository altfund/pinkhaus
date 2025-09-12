#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Free Data Pull V2 - Refactored to prioritize blockchain/GraphQL
Uses the DataSourceManager to fetch data with automatic fallback.
API calls are only used as a last resort.
"""

import asyncio
import pandas as pd
import sqlite3
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import List, Optional

from data_source_manager import DataSourceManager

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DB_NAME = "sport_odds.db"
UPDATE_INTERVAL = 60  # seconds


class DataCollector:
    """Handles data collection from multiple sources."""
    
    def __init__(self, db_name: str = DB_NAME):
        self.db_name = db_name
        self.manager = DataSourceManager()
        self._initialized = False
        
    async def initialize(self, network: str = 'optimism'):
        """Initialize the data collector."""
        await self.manager.initialize(network)
        self._initialized = True
        
    def last_update_time(self, table_name: str) -> Optional[datetime]:
        """Get last update time for a table."""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT last_updated FROM table_metadata WHERE table_name = ?",
                (table_name,),
            )
            result = cursor.fetchone()
            return pd.to_datetime(result[0]) if result and result[0] else None
    
    def needs_update(self, table_name: str) -> bool:
        """Check if table needs updating."""
        last_updated = self.last_update_time(table_name)
        if last_updated is None:
            return True
        return (pd.Timestamp.now() - last_updated).total_seconds() > UPDATE_INTERVAL
    
    def update_table_metadata(self, table_name: str):
        """Update the last update time for a table."""
        with sqlite3.connect(self.db_name) as conn:
            conn.execute(
                "REPLACE INTO table_metadata (table_name, last_updated) VALUES (?, ?)",
                (table_name, pd.Timestamp.now().isoformat()),
            )
    
    async def collect_markets(self, force_update: bool = False) -> pd.DataFrame:
        """Collect market data from available sources."""
        if not force_update and not self.needs_update("market"):
            logger.info("Markets table is up-to-date; skipping update")
            return pd.DataFrame()
        
        logger.info("Collecting market data...")
        
        try:
            # Get markets from the last 7 days and next 30 days
            start_date = datetime.now() - timedelta(days=7)
            end_date = datetime.now() + timedelta(days=30)
            
            markets = await self.manager.get_markets(
                start_date=start_date,
                end_date=end_date
            )
            
            if not markets:
                logger.warning("No markets fetched")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(markets)
            
            # Ensure required columns
            required_cols = ['source_id', 'sport', 'league', 'home_team', 'away_team', 
                           'market_type', 'maturity_date']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None
            
            # Add metadata
            df['updated_at'] = datetime.now()
            
            logger.info(f"Collected {len(df)} markets")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting markets: {e}")
            # Try backup API if all else fails
            logger.info("Attempting to use backup API...")
            try:
                markets = await self.manager.get_markets(use_backup=True)
                df = pd.DataFrame(markets)
                df['updated_at'] = datetime.now()
                logger.info(f"Collected {len(df)} markets from backup API")
                return df
            except Exception as backup_error:
                logger.error(f"Backup API also failed: {backup_error}")
                return pd.DataFrame()
    
    async def collect_odds(self, market_ids: List[str]) -> pd.DataFrame:
        """Collect odds data for specific markets."""
        if not market_ids:
            return pd.DataFrame()
        
        logger.info(f"Collecting odds for {len(market_ids)} markets...")
        
        try:
            odds_data = await self.manager.get_odds(market_ids)
            
            if not odds_data:
                logger.warning("No odds data fetched")
                return pd.DataFrame()
            
            df = pd.DataFrame(odds_data)
            df['timestamp'] = datetime.now()
            
            logger.info(f"Collected {len(df)} odds records")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting odds: {e}")
            return pd.DataFrame()
    
    def save_markets(self, df: pd.DataFrame):
        """Save markets to database."""
        if df.empty:
            return
        
        with sqlite3.connect(self.db_name, timeout=10) as conn:
            # Upsert markets
            for _, row in df.iterrows():
                market_dict = row.to_dict()
                
                # Insert or update
                conn.execute("""
                    INSERT OR REPLACE INTO market (
                        source_id, source, sport, league, home_team, away_team,
                        market_type, maturity_date, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    market_dict.get('source_id'),
                    market_dict.get('source', 'graphql'),
                    market_dict.get('sport'),
                    market_dict.get('league'),
                    market_dict.get('home_team'),
                    market_dict.get('away_team'),
                    market_dict.get('market_type', 'moneyline'),
                    market_dict.get('maturity_date'),
                    market_dict.get('updated_at')
                ))
        
        self.update_table_metadata("market")
        logger.info(f"Saved {len(df)} markets to database")
    
    def save_odds(self, df: pd.DataFrame):
        """Save odds to database."""
        if df.empty:
            return
        
        with sqlite3.connect(self.db_name, timeout=10) as conn:
            df.to_sql('odd', conn, if_exists='append', index=False)
        
        logger.info(f"Saved {len(df)} odds records to database")
    
    async def run_collection_cycle(self):
        """Run a complete data collection cycle."""
        if not self._initialized:
            await self.initialize()
        
        # Log data source status
        status = await self.manager.get_status()
        logger.info("Data source availability:")
        for source in status['sources']:
            logger.info(f"  - {source['name']} ({source['priority']}): "
                       f"{'Available' if source['available'] else 'Unavailable'}")
        
        # Collect markets
        markets_df = await self.collect_markets()
        if not markets_df.empty:
            self.save_markets(markets_df)
            
            # Collect odds for these markets
            market_ids = markets_df['source_id'].unique().tolist()
            odds_df = await self.collect_odds(market_ids[:100])  # Limit for performance
            if not odds_df.empty:
                self.save_odds(odds_df)
        
        logger.info("Data collection cycle completed")


async def main():
    """Main entry point."""
    logger.info("Starting free_data_pull_v2...")
    collector = DataCollector()
    
    # Initialize database if needed
    try:
        logger.info("Running database migrations...")
        from alembic.config import Config
        from alembic import command
        cfg = Config("alembic.ini")
        command.upgrade(cfg, "head")
        logger.info("Migrations completed")
    except Exception as e:
        logger.warning(f"Could not run migrations: {e}")
    
    # Run collection
    logger.info("Starting data collection cycle...")
    await collector.run_collection_cycle()
    logger.info("Data collection completed")


if __name__ == "__main__":
    asyncio.run(main())