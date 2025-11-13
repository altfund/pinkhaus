#!/usr/bin/env python3
"""
Check English soccer markets specifically
"""
import os

# Set environment for PostgreSQL on port 5999
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import and_, or_, not_, func
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_english_soccer():
    """Check English soccer markets only"""
    
    with db_manager.get_db_session() as db:
        # Focus on England only first
        query = db.query(Market, Odd).join(Odd, Market.source_id == Odd.source_id).filter(
            # Exclude default odds
            not_(and_(
                Odd.decimal_odds.in_([2.5, 2.8, 3.0])
            )),
            Market.sport == 'Soccer',
            Market.nation == 'England'
        ).order_by(Market.maturity_date.desc()).limit(30)
        
        results = query.all()
        
        logger.info(f"\n=== English Soccer Markets ===")
        logger.info(f"Total found: {len(results)}")
        
        # Group by league
        leagues = {}
        for market, odd in results:
            league = market.league_name or 'Unknown'
            if league not in leagues:
                leagues[league] = 0
            leagues[league] += 1
        
        logger.info(f"\nLeagues found:")
        for league, count in leagues.items():
            logger.info(f"  {league}: {count} markets")
        
        # Show sample markets
        logger.info(f"\nSample English markets:")
        for i, (market, odd) in enumerate(results[:15]):
            logger.info(f"\n{i+1}. {market.home_team} vs {market.away_team}")
            logger.info(f"   League: {market.league_name}")
            logger.info(f"   Odds: {odd.decimal_odds} ({odd.outcome})")

if __name__ == "__main__":
    check_english_soccer()