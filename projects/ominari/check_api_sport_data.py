#!/usr/bin/env python3
"""
Check what sport data we have from API
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
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_api_data():
    """Check what sport data we have from API"""
    
    with db_manager.get_db_session() as db:
        # Check markets with metadata
        logger.info("=== Checking markets with metadata ===")
        
        # Look for Missouri, Yale, etc.
        problem_teams = ['Missouri', 'South Carolina', 'Yale', 'Holy Cross', 
                        'Louisville', 'Bowling Green', 'Massachusetts Maritime']
        
        for team in problem_teams:
            markets = db.query(Market).filter(
                Market.home_team.like(f'%{team}%') | 
                Market.away_team.like(f'%{team}%')
            ).limit(5).all()
            
            for market in markets:
                logger.info(f"\n{market.home_team} vs {market.away_team}")
                logger.info(f"  Sport: {market.sport}")
                logger.info(f"  League: {market.league_name}")
                logger.info(f"  Nation: {market.nation}")
                logger.info(f"  Source: {market.source}")
                logger.info(f"  Source ID: {market.source_id}")
                
                # Check if we have any metadata
                if hasattr(market, 'metadata') and market.metadata:
                    try:
                        meta = json.loads(market.metadata)
                        logger.info(f"  Metadata: {meta}")
                    except:
                        pass
        
        # Check if "International Football" is consistently one sport
        logger.info("\n=== International Football League Sport Distribution ===")
        sports_dist = db.query(
            Market.sport,
            db.func.count().label('count')
        ).filter(
            Market.league_name == 'International Football'
        ).group_by(Market.sport).all()
        
        for sport, count in sports_dist:
            logger.info(f"  {sport}: {count} markets")

if __name__ == "__main__":
    check_api_data()