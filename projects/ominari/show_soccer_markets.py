#!/usr/bin/env python3
"""
Show only soccer markets as a quick test
"""
import os

# Set environment for PostgreSQL on port 5999
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

# Set sport filter to Soccer only
os.environ['ALLOWED_SPORTS'] = 'Soccer'

from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import and_, not_
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def show_soccer_markets():
    with db_manager.get_db_session() as db:
        # Get soccer markets with real odds
        query = db.query(Market, Odd).join(Odd, Market.source_id == Odd.source_id).filter(
            Market.sport == 'Soccer',
            not_(and_(
                Odd.decimal_odds.in_([2.5, 2.8, 3.0])
            ))
        ).order_by(Market.maturity_date.desc()).limit(20)
        
        results = query.all()
        
        logger.info(f"Found {len(results)} soccer markets with real odds")
        logger.info("\nTop Soccer Markets:")
        logger.info("=" * 80)
        
        for market, odd in results:
            logger.info(f"{market.home_team} vs {market.away_team}")
            logger.info(f"  League: {market.league_name}")
            logger.info(f"  Date: {market.maturity_date}")
            logger.info(f"  {odd.outcome}: {odd.decimal_odds}")
            logger.info("-" * 40)
        
        # Show league distribution
        from sqlalchemy import func
        league_dist = db.query(Market.league_name, func.count(Market.source_id)).filter(
            Market.sport == 'Soccer'
        ).group_by(Market.league_name).order_by(func.count(Market.source_id).desc()).limit(10).all()
        
        logger.info("\nTop Soccer Leagues:")
        for league, count in league_dist:
            logger.info(f"  {league}: {count} markets")

if __name__ == "__main__":
    show_soccer_markets()