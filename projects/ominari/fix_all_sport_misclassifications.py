#!/usr/bin/env python3
"""
Comprehensive fix for all sport misclassifications
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

def fix_all_misclassifications():
    """Fix all sport misclassifications comprehensively"""
    
    with db_manager.get_db_session() as db:
        total_fixed = 0
        
        # Fix MLB - Baseball
        logger.info("=== Fixing MLB Markets ===")
        mlb_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            (Market.home_team == 'MLB') | (Market.away_team == 'MLB')
        ).all()
        
        for market in mlb_markets:
            market.sport = 'Baseball'
        
        if mlb_markets:
            db.commit()
            logger.info(f"Fixed {len(mlb_markets)} MLB markets → Baseball")
            total_fixed += len(mlb_markets)
        
        # Fix Esports
        logger.info("\n=== Fixing Esports Markets ===")
        esports_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            (Market.home_team.like('%Esports%') | 
             Market.away_team.like('%Esports%') |
             Market.home_team.like('%Gaming%') |
             Market.away_team.like('%Gaming%') |
             Market.league_name.like('%MLBB%') |
             Market.league_name.like('%OCS%') |
             Market.league_name.like('%The International%') |
             Market.league_name.like('%League of Legends%') |
             Market.league_name.like('%Esports%'))
        ).all()
        
        for market in esports_markets:
            market.sport = 'Esports'
        
        if esports_markets:
            db.commit()
            logger.info(f"Fixed {len(esports_markets)} Esports markets")
            total_fixed += len(esports_markets)
        
        # Fix American Football
        logger.info("\n=== Fixing American Football Markets ===")
        football_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.league_name == 'International Football',
            (Market.home_team.like('%College%') |
             Market.away_team.like('%College%') |
             Market.home_team.in_(['Texas', 'UTEP', 'Buffalo', 'Troy']) |
             Market.away_team.in_(['Texas', 'UTEP', 'Buffalo', 'Troy']))
        ).all()
        
        for market in football_markets:
            market.sport = 'American Football'
        
        if football_markets:
            db.commit()
            logger.info(f"Fixed {len(football_markets)} American Football markets")
            total_fixed += len(football_markets)
        
        # Fix Fighting/MMA 
        logger.info("\n=== Fixing Fighting Sports ===")
        # Pattern: FirstName LastName vs FirstName LastName
        fighting_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.home_team.notlike('%Esports%'),
            Market.away_team.notlike('%Esports%'),
            Market.home_team.like('% %'),  # Has space (first last)
            Market.away_team.like('% %'),  # Has space (first last)
            ~Market.home_team.like('% FC%'),
            ~Market.home_team.like('% United%'),
            ~Market.home_team.like('% City%'),
            ~Market.away_team.like('% FC%'),
            ~Market.away_team.like('% United%'),
            ~Market.away_team.like('% City%'),
            ~Market.home_team.like('%College%') == False,
            ~Market.away_team.like('%College%') == False,
            ~Market.home_team.like('% State%') == False,
            ~Market.away_team.like('% State%') == False
        ).all()
        
        # Further filter by checking if they look like person names
        real_fighting = []
        for market in fighting_markets:
            home_parts = market.home_team.split()
            away_parts = market.away_team.split()
            
            # Typical fighter name pattern: 2-3 words, first word capitalized
            if (2 <= len(home_parts) <= 3 and 
                2 <= len(away_parts) <= 3 and
                home_parts[0][0].isupper() and 
                away_parts[0][0].isupper() and
                'International Football' in str(market.league_name)):
                real_fighting.append(market)
                market.sport = 'MMA'
        
        if real_fighting:
            db.commit()
            logger.info(f"Fixed {len(real_fighting)} fighting/MMA markets")
            for m in real_fighting[:5]:
                logger.info(f"  Example: {m.home_team} vs {m.away_team}")
            total_fixed += len(real_fighting)
        
        # Fix remaining college sports
        logger.info("\n=== Fixing College Sports ===")
        # College basketball patterns
        college_basketball = db.query(Market).filter(
            Market.sport == 'Soccer',
            (Market.home_team.like('%College%') |
             Market.away_team.like('%College%') |
             Market.home_team.like('% State') |
             Market.away_team.like('% State')),
            Market.league_name.notlike('%Soccer%'),
            Market.league_name.notlike('%Football%')
        ).all()
        
        for market in college_basketball:
            # Determine sport based on context
            if 'Ligue 1' in market.league_name:
                # This is misnamed - likely basketball
                market.sport = 'Basketball'
            elif 'Playoffs' in market.league_name:
                market.sport = 'American Football'
            else:
                market.sport = 'Basketball'  # Default for colleges
        
        if college_basketball:
            db.commit()
            logger.info(f"Fixed {len(college_basketball)} college sports markets")
            total_fixed += len(college_basketball)
        
        # Fix Cricket
        logger.info("\n=== Fixing Cricket Markets ===")
        cricket_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            (Market.home_team.like('%Police%') |
             Market.away_team.like('%Police%') |
             Market.home_team.like('%Customs%') |
             Market.away_team.like('%Customs%') |
             Market.home_team.like('%Sangha%') |
             Market.away_team.like('%Sangha%'))
        ).all()
        
        for market in cricket_markets:
            market.sport = 'Cricket'
        
        if cricket_markets:
            db.commit()
            logger.info(f"Fixed {len(cricket_markets)} cricket markets")
            total_fixed += len(cricket_markets)
        
        logger.info(f"\n✅ Total markets fixed: {total_fixed}")
        
        # Re-sync sport from API if needed
        logger.info("\n=== Running API sport sync ===")
        from sync_sports_from_api import sync_sports_from_overtime_api
        try:
            sync_sports_from_overtime_api()
        except Exception as e:
            logger.warning(f"Could not run API sync: {e}")

if __name__ == "__main__":
    fix_all_misclassifications()