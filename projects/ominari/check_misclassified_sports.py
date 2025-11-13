#!/usr/bin/env python3
"""
Check for misclassified sports in the database
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

def check_suspicious_soccer():
    """Find soccer markets that are likely misclassified"""
    
    with db_manager.get_db_session() as db:
        # Check for MLB in soccer
        logger.info("=== Checking for MLB in Soccer ===")
        mlb_soccer = db.query(Market).filter(
            Market.sport == 'Soccer',
            (Market.home_team.like('%MLB%') | 
             Market.away_team.like('%MLB%') |
             Market.league_name.like('%MLB%'))
        ).limit(10).all()
        
        if mlb_soccer:
            logger.info(f"Found {len(mlb_soccer)} MLB markets classified as Soccer:")
            for m in mlb_soccer:
                logger.info(f"  {m.home_team} vs {m.away_team} ({m.league_name})")
        
        # Check for Esports in soccer
        logger.info("\n=== Checking for Esports in Soccer ===")
        esports_soccer = db.query(Market).filter(
            Market.sport == 'Soccer',
            (Market.home_team.like('%Esports%') | 
             Market.away_team.like('%Esports%') |
             Market.home_team.like('%Gaming%') |
             Market.away_team.like('%Gaming%') |
             Market.league_name.like('%MLBB%') |
             Market.league_name.like('%Esports%'))
        ).limit(10).all()
        
        if esports_soccer:
            logger.info(f"Found {len(esports_soccer)} Esports markets classified as Soccer:")
            for m in esports_soccer:
                logger.info(f"  {m.home_team} vs {m.away_team} ({m.league_name})")
        
        # Check for fighter names in soccer
        logger.info("\n=== Checking for Fighter Names in Soccer ===")
        fighter_patterns = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.league_name.like('%Championship%'),
            ~Market.league_name.like('%Football Championship%'),
            Market.home_team.like('% %'),
            Market.away_team.like('% %'),
            ~Market.home_team.like('% FC%'),
            ~Market.home_team.like('% United%'),
            ~Market.away_team.like('% FC%'),
            ~Market.away_team.like('% United%')
        ).limit(10).all()
        
        if fighter_patterns:
            logger.info(f"Found {len(fighter_patterns)} potential fighter markets in Soccer:")
            for m in fighter_patterns:
                logger.info(f"  {m.home_team} vs {m.away_team} ({m.league_name})")
        
        # Check college sports in soccer
        logger.info("\n=== Checking for College Sports in Soccer ===")
        college_soccer = db.query(Market).filter(
            Market.sport == 'Soccer',
            (Market.home_team.like('% State') | 
             Market.away_team.like('% State') |
             Market.home_team.like('% University') |
             Market.away_team.like('% University') |
             Market.home_team.like('% College') |
             Market.away_team.like('% College')),
            Market.league_name.notlike('%Soccer%'),
            Market.league_name.notlike('%Football%')
        ).limit(10).all()
        
        if college_soccer:
            logger.info(f"Found {len(college_soccer)} college sports in Soccer:")
            for m in college_soccer:
                logger.info(f"  {m.home_team} vs {m.away_team} ({m.league_name})")
        
        # Summary of international markets
        logger.info("\n=== International Soccer Markets ===")
        intl_soccer = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.nation == 'International'
        ).limit(20).all()
        
        logger.info(f"Sample of International soccer markets:")
        for m in intl_soccer:
            logger.info(f"  {m.home_team} vs {m.away_team} ({m.league_name})")

if __name__ == "__main__":
    check_suspicious_soccer()