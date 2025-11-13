#!/usr/bin/env python3
"""
Verify sport classifications more accurately
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
from models import Market
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_classifications():
    """Verify sport classifications based on more reliable indicators"""
    
    with db_manager.get_db_session() as db:
        # 1. Check markets with opticOddsName for verification
        logger.info("=== Checking markets with opticOddsName ===")
        
        sample_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.optic_odds_name.isnot(None),
            Market.nation.in_(['England', 'International', 'Europe'])
        ).limit(30).all()
        
        logger.info(f"\nSample Soccer markets with API names:")
        for market in sample_markets[:20]:
            logger.info(f"\n{market.home_team} vs {market.away_team}")
            logger.info(f"  League: {market.league_name}")
            logger.info(f"  Nation: {market.nation}")
            logger.info(f"  API Name: {market.optic_odds_name}")
        
        # 2. Look for definitive soccer leagues
        logger.info("\n=== Soccer markets by league ===")
        
        soccer_leagues = [
            'Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1',
            'Championship', 'UEFA Champions League', 'Europa League', 'FA Cup',
            'EFL Cup', 'Copa del Rey', 'DFB-Pokal', 'Coppa Italia'
        ]
        
        for league in soccer_leagues:
            count = db.query(Market).filter(
                Market.sport == 'Soccer',
                Market.league_name.like(f'%{league}%')
            ).count()
            if count > 0:
                logger.info(f"  {league}: {count} markets")
        
        # 3. Check if "International Football" league is actually soccer
        logger.info("\n=== International Football league analysis ===")
        intl_football_sample = db.query(Market).filter(
            Market.league_name == 'International Football'
        ).limit(10).all()
        
        for market in intl_football_sample:
            logger.info(f"  {market.home_team} vs {market.away_team} - Sport: {market.sport}")
            if market.optic_odds_name:
                logger.info(f"    API: {market.optic_odds_name}")

if __name__ == "__main__":
    verify_classifications()