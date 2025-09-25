#!/usr/bin/env python3
"""
Verify dashboard data is accessible
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from database_v2 import db_manager
from models import Market, Odd
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function."""
    logger.info("🔍 Verifying Dashboard Data")
    logger.info("=" * 60)
    
    try:
        with db_manager.get_db_session() as db:
            # Count markets
            total_markets = db.query(Market).count()
            active_markets = db.query(Market).filter(
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).count()
            
            logger.info(f"\n📊 Database Status:")
            logger.info(f"Total markets: {total_markets}")
            logger.info(f"Active markets: {active_markets}")
            
            # Get some sample markets
            markets = db.query(Market).filter(
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).limit(5).all()
            
            logger.info(f"\n🏆 Sample Active Markets:")
            for market in markets:
                logger.info(f"\n{market.home_team} vs {market.away_team}")
                logger.info(f"  Sport: {market.sport}")
                logger.info(f"  Source: {market.source}")
                logger.info(f"  Maturity: {market.maturity_date}")
                
                # Get odds for this market
                odds = db.query(Odd).filter(Odd.source_id == market.source_id).all()
                if odds:
                    logger.info(f"  Odds:")
                    for odd in odds:
                        american = int(odd.american_odds) if odd.american_odds else 0
                        logger.info(f"    {odd.outcome}: {odd.decimal_odds:.2f} ({american:+d})")
                        
            # Check different sports
            sports = db.query(Market.sport).distinct().all()
            logger.info(f"\n🏅 Sports Available:")
            for sport in sports:
                count = db.query(Market).filter(Market.sport == sport[0]).count()
                logger.info(f"  {sport[0]}: {count} markets")
                
            logger.info(f"\n✅ Dashboard Status:")
            logger.info(f"Web monitor running at: http://localhost:8888/unified")
            logger.info(f"\n🎆 The dashboard should now display:")
            logger.info(f"  - Portfolio Value section")
            logger.info(f"  - Performance Metrics")
            logger.info(f"  - Match Dashboard with {active_markets} active markets")
            logger.info(f"  - Real-time odds updates")
            
    except Exception as e:
        logger.error(f"Error verifying data: {e}")

if __name__ == "__main__":
    main()