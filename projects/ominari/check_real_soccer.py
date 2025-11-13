#!/usr/bin/env python3
"""
Check for real soccer matches
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

def check_real_soccer():
    """Check for actual soccer matches"""
    
    with db_manager.get_db_session() as db:
        # Look for markets with soccer-specific indicators
        query = db.query(Market, Odd).join(Odd, Market.source_id == Odd.source_id).filter(
            # Exclude default odds
            not_(and_(
                Odd.decimal_odds.in_([2.5, 2.8, 3.0])
            )),
            Market.sport == 'Soccer',
            # Look for actual soccer team indicators
            or_(
                Market.home_team.like('%FC%'),
                Market.away_team.like('%FC%'),
                Market.home_team.like('%United%'),
                Market.away_team.like('%United%'),
                Market.home_team.like('%City%'),
                Market.away_team.like('%City%'),
                Market.home_team.like('%Athletic%'),
                Market.away_team.like('%Athletic%'),
                Market.league_name.like('%Premier League%'),
                Market.league_name.like('%La Liga%'),
                Market.league_name.like('%Serie A%'),
                Market.league_name.like('%Bundesliga%'),
                Market.league_name.like('%Ligue 1%'),
                Market.league_name.like('%Championship%'),
                Market.league_name.like('%UEFA%')
            )
        ).order_by(Market.maturity_date.desc()).limit(50)
        
        results = query.all()
        
        logger.info(f"\n=== Real Soccer Matches (Top 50) ===")
        logger.info(f"Total found: {len(results)}")
        
        # Show sample markets
        logger.info(f"\nSample markets:")
        for i, (market, odd) in enumerate(results[:20]):
            logger.info(f"\n{i+1}. {market.home_team} vs {market.away_team}")
            logger.info(f"   Sport: {market.sport}")
            logger.info(f"   League: {market.league_name}")
            logger.info(f"   Nation: {market.nation}")
            logger.info(f"   Odds: {odd.decimal_odds} ({odd.outcome})")
            
            # Check links
            if market.source_id:
                if market.source_id.startswith('0x'):
                    market_hex = market.source_id[2:]
                    logger.info(f"   Overtime: https://overtimemarketsv2.com/#/markets/{market_hex}")
                else:
                    logger.info(f"   Overtime: https://overtimemarketsv2.com/#/markets/{market.source_id}")

        # Count by nation
        logger.info("\n=== Soccer by Nation ===")
        nation_counts = db.query(
            Market.nation,
            func.count(Market.id).label('count')
        ).filter(
            Market.sport == 'Soccer',
            Market.nation.in_(['England', 'Spain', 'Italy', 'Germany', 'France'])
        ).group_by(Market.nation).all()
        
        for nation, count in nation_counts:
            logger.info(f"  {nation}: {count} markets")

if __name__ == "__main__":
    check_real_soccer()