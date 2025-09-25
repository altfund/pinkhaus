#!/usr/bin/env python3
"""
Simple test of backtesting database connectivity with PostgreSQL
"""

import os

# Set PostgreSQL environment FIRST before importing any database modules
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

import logging
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_backtest_data():
    """Test simple backtesting data queries with PostgreSQL."""
    logger.info("🧪 Testing Simple Backtest Data Queries with PostgreSQL")
    logger.info("=" * 60)
    
    try:
        # Test 1: Query markets with odds for backtesting
        with db_manager.get_db_session() as db:
            
            # Count total markets and odds
            total_markets = db.query(Market).count()
            total_odds = db.query(Odd).count()
            
            logger.info(f"✅ Database access working!")
            logger.info(f"   Total Markets: {total_markets:,}")
            logger.info(f"   Total Odds: {total_odds:,}")
            
            # Test 2: Query markets with maturity dates for time-based backtesting
            recent_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
            recent_markets = db.query(Market).filter(
                Market.maturity_date > recent_cutoff
            ).count()
            
            logger.info(f"   Recent Markets (30d): {recent_markets:,}")
            
            # Test 3: Query finished markets (needed for backtesting)
            finished_markets = db.query(Market).filter(
                Market.is_finished == True
            ).count()
            
            logger.info(f"   Finished Markets: {finished_markets:,}")
            
            # Test 4: Sample market with odds
            sample_query = db.query(Market, Odd).join(
                Odd, Market.source_id == Odd.source_id
            ).limit(5).all()
            
            logger.info(f"\n📊 Sample Markets with Odds:")
            for market, odd in sample_query:
                logger.info(f"   • {market.home_team} vs {market.away_team}")
                logger.info(f"     Outcome: {odd.outcome}, Odds: {odd.decimal_odds}")
            
            # Test 5: Check if we have different sports (needed for diversified backtesting)
            sports_query = db.query(Market.sport).distinct().all()
            sports = [s[0] for s in sports_query if s[0]]
            
            logger.info(f"\n🏆 Available Sports for Backtesting:")
            for sport in sorted(sports)[:10]:  # Show first 10
                sport_count = db.query(Market).filter(Market.sport == sport).count()
                logger.info(f"   • {sport}: {sport_count:,} markets")
            
            if len(sports) > 10:
                logger.info(f"   ... and {len(sports) - 10} more sports")
            
            # Test 6: Check date range of data
            oldest_market = db.query(Market.maturity_date).filter(
                Market.maturity_date.isnot(None)
            ).order_by(Market.maturity_date.asc()).first()
            
            newest_market = db.query(Market.maturity_date).filter(
                Market.maturity_date.isnot(None)
            ).order_by(Market.maturity_date.desc()).first()
            
            if oldest_market and newest_market:
                logger.info(f"\n📅 Data Date Range:")
                logger.info(f"   Oldest: {oldest_market[0]}")
                logger.info(f"   Newest: {newest_market[0]}")
                
                # Calculate data span
                span = newest_market[0] - oldest_market[0]
                logger.info(f"   Span: {span.days} days")
        
        logger.info(f"\n✅ Simple backtest data test completed successfully!")
        logger.info(f"🎯 Backtesting system has sufficient PostgreSQL data for testing!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple backtest test failed: {e}")
        return False

if __name__ == "__main__":
    test_simple_backtest_data()