#!/usr/bin/env python3
"""
Investigate specific misclassifications
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

def investigate():
    """Investigate specific misclassifications"""
    
    with db_manager.get_db_session() as db:
        # 1. Check "International Football" league
        logger.info("=== International Football League Analysis ===")
        intl_football = db.query(Market).filter(
            Market.league_name == 'International Football'
        ).limit(30).all()
        
        logger.info(f"Found {len(intl_football)} 'International Football' markets")
        logger.info("\nSample teams:")
        for market in intl_football[:20]:
            logger.info(f"  {market.home_team} vs {market.away_team}")
            logger.info(f"    Sport: {market.sport}, Nation: {market.nation}")
        
        # 2. Check Fayetteville State specifically
        logger.info("\n=== Fayetteville State Check ===")
        fayetteville = db.query(Market).filter(
            Market.home_team.like('%Fayetteville State%') | 
            Market.away_team.like('%Fayetteville State%')
        ).all()
        
        for market in fayetteville:
            logger.info(f"  {market.home_team} vs {market.away_team}")
            logger.info(f"    Sport: {market.sport}")
            logger.info(f"    League: {market.league_name}")
            logger.info(f"    Nation: {market.nation}")
            logger.info(f"    Governing Body: {market.governing_body}")
        
        # 3. Check "European Football" league with American teams
        logger.info("\n=== European Football League with US Teams ===")
        
        # US state names
        us_states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
                     'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
                     'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
                     'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
                     'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
                     'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
                     'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
                     'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
                     'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
                     'West Virginia', 'Wisconsin', 'Wyoming']
        
        # Find European Football with US state names
        misclassified = []
        for state in us_states:
            markets = db.query(Market).filter(
                Market.league_name.like('%European Football%'),
                (Market.home_team.like(f'%{state}%') | 
                 Market.away_team.like(f'%{state}%'))
            ).all()
            misclassified.extend(markets)
        
        logger.info(f"Found {len(misclassified)} US teams in 'European Football' league")
        for market in misclassified[:10]:
            logger.info(f"  {market.home_team} vs {market.away_team} - Nation: {market.nation}")

if __name__ == "__main__":
    investigate()