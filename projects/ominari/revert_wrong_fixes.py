#!/usr/bin/env python3
"""
Revert wrongly classified esports and fix American Football
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

def fix_classifications():
    """Fix misclassifications more carefully"""
    
    with db_manager.get_db_session() as db:
        # Fix American Football games wrongly marked as Esports
        logger.info("=== Fixing American Football ===")
        
        # College football patterns
        college_football = db.query(Market).filter(
            Market.sport == 'Esports',
            (Market.home_team.like('%State') | 
             Market.away_team.like('%State') |
             Market.home_team.like('%College') |
             Market.away_team.like('%College') |
             Market.home_team.like('%University') |
             Market.away_team.like('%University')),
            Market.league_name.like('%Football%')
        ).all()
        
        for market in college_football:
            market.sport = 'American Football'
        
        if college_football:
            db.commit()
            logger.info(f"Fixed {len(college_football)} college football games")
        
        # Fix NFL teams
        nfl_teams = ['Vikings', 'Falcons', 'Patriots', 'Cowboys', 'Eagles', 'Giants', 
                     'Bears', 'Packers', 'Lions', 'Saints', 'Buccaneers', 'Panthers',
                     'Rams', '49ers', 'Cardinals', 'Seahawks', 'Steelers', 'Ravens',
                     'Browns', 'Bengals', 'Titans', 'Colts', 'Jaguars', 'Texans',
                     'Broncos', 'Chiefs', 'Raiders', 'Chargers', 'Bills', 'Dolphins',
                     'Jets', 'Commanders']
        
        for team in nfl_teams:
            markets = db.query(Market).filter(
                Market.sport.in_(['Esports', 'Soccer']),
                (Market.home_team.like(f'%{team}%') | Market.away_team.like(f'%{team}%')),
                Market.league_name.notlike('%Soccer%')
            ).all()
            
            for market in markets:
                market.sport = 'American Football'
                
        db.commit()
        logger.info(f"Fixed NFL teams")
        
        # Fix actual soccer teams wrongly marked as Esports
        logger.info("\n=== Reverting real Soccer teams ===")
        
        # Teams with FC, United, City etc are definitely soccer
        soccer_patterns = ['%FC%', '%United%', '%City%', '%Town%', '%Albion%', 
                          '%Rovers%', '%Athletic%', '%Wanderers%', '%Hotspur%']
        
        for pattern in soccer_patterns:
            markets = db.query(Market).filter(
                Market.sport == 'Esports',
                (Market.home_team.like(pattern) | Market.away_team.like(pattern))
            ).all()
            
            for market in markets:
                market.sport = 'Soccer'
        
        db.commit()
        logger.info("Reverted soccer teams back to Soccer")
        
        # Fix remaining esports more carefully
        logger.info("\n=== Fixing true Esports ===")
        
        # Only mark as esports if it has clear esports indicators
        esports_keywords = ['Gaming', 'Esports', 'G2', 'Liquid', 'Fnatic', 'TSM', 
                           'Cloud9', 'FaZe', 'NaVi', 'OG ', 'MLBB', 'MOBA', 'FPS',
                           'Dota', 'League of Legends', 'VALORANT', 'CS:GO']
        
        for keyword in esports_keywords:
            markets = db.query(Market).filter(
                Market.sport == 'Soccer',
                (Market.home_team.like(f'%{keyword}%') | 
                 Market.away_team.like(f'%{keyword}%') |
                 Market.league_name.like(f'%{keyword}%'))
            ).all()
            
            for market in markets:
                market.sport = 'Esports'
        
        db.commit()
        logger.info("Fixed true esports teams")
        
        # Show remaining soccer sample
        logger.info("\n=== Soccer Markets Sample ===")
        soccer_sample = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.nation.in_(['England', 'International', 'Europe'])
        ).limit(20).all()
        
        for market in soccer_sample:
            logger.info(f"  {market.home_team} vs {market.away_team} ({market.league_name})")

if __name__ == "__main__":
    fix_classifications()