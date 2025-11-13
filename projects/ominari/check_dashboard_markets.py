#!/usr/bin/env python3
"""
Check current markets in dashboard view
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

def check_dashboard_markets():
    """Check what markets would show in dashboard"""
    
    with db_manager.get_db_session() as db:
        # Same query as dashboard
        query = db.query(Market, Odd).join(Odd, Market.source_id == Odd.source_id).filter(
            # Exclude default odds
            not_(and_(
                Odd.decimal_odds.in_([2.5, 2.8, 3.0])
            )),
            # Sport filter
            Market.sport == 'Soccer',
            # Nation filter
            Market.nation.in_(['England', 'International', 'Europe'])
        ).order_by(Market.maturity_date.desc()).limit(50)
        
        results = query.all()
        
        logger.info(f"\n=== Dashboard Markets (Top 50) ===")
        logger.info(f"Total found: {len(results)}")
        
        # Group by sport to verify
        sports = {}
        for market, odd in results:
            if market.sport not in sports:
                sports[market.sport] = 0
            sports[market.sport] += 1
        
        logger.info(f"\nSports distribution:")
        for sport, count in sports.items():
            logger.info(f"  {sport}: {count}")
        
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
                    logger.info(f"   Blockchain: https://optimistic.etherscan.io/address/{market.source_id}")
                else:
                    logger.info(f"   Overtime: https://overtimemarketsv2.com/#/markets/{market.source_id}")

if __name__ == "__main__":
    check_dashboard_markets()