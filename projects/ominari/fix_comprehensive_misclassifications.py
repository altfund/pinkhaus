#!/usr/bin/env python3
"""
Comprehensive fix for sport and nation misclassifications
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
    """Fix all sport and nation misclassifications"""
    
    with db_manager.get_db_session() as db:
        fixed_count = 0
        
        # 1. Fix US teams in European Football
        logger.info("=== Fixing US Teams in European Football ===")
        
        # US states and cities
        us_indicators = [
            # States
            'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
            'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
            'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
            'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
            'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
            'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
            'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
            'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
            'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
            'West Virginia', 'Wisconsin', 'Wyoming',
            # Major cities
            'Miami', 'Dallas', 'Houston', 'Phoenix', 'Denver', 'Seattle',
            'Portland', 'San Diego', 'San Francisco', 'Los Angeles', 'Las Vegas',
            'Salt Lake City', 'Kansas City', 'St. Louis', 'Chicago', 'Detroit',
            'Minneapolis', 'Milwaukee', 'Cincinnati', 'Cleveland', 'Columbus',
            'Indianapolis', 'Nashville', 'Memphis', 'New Orleans', 'Atlanta',
            'Charlotte', 'Tampa', 'Orlando', 'Jacksonville', 'Boston', 'Baltimore',
            'Philadelphia', 'Pittsburgh', 'Buffalo',
            # College indicators
            'State University', 'State College', 'University', 'College'
        ]
        
        for indicator in us_indicators:
            markets = db.query(Market).filter(
                Market.league_name.like('%European Football%'),
                (Market.home_team.like(f'%{indicator}%') | 
                 Market.away_team.like(f'%{indicator}%')),
                Market.nation == 'Europe'
            ).all()
            
            for market in markets:
                # Determine sport based on team names
                home_lower = market.home_team.lower()
                away_lower = market.away_team.lower()
                
                # American Football indicators
                if any(term in home_lower + away_lower for term in 
                       ['vikings', 'falcons', 'patriots', 'cowboys', 'eagles', 
                        'giants', 'bears', 'packers', 'lions', 'saints']):
                    market.sport = 'American Football'
                    market.nation = 'United States'
                    market.governing_body = 'NFL'
                # Check for FC/United (soccer)
                elif any(term in home_lower + away_lower for term in
                         ['fc', ' united', ' city', 'wave fc', 'current']):
                    # Keep as soccer but fix nation
                    market.nation = 'United States'
                    market.governing_body = 'USSF'
                else:
                    # Likely American Football college
                    market.sport = 'American Football'
                    market.nation = 'United States'
                    market.governing_body = 'NCAA'
                
                fixed_count += 1
        
        logger.info(f"Fixed {fixed_count} US teams in European Football")
        
        # 2. Fix remaining Soccer teams that should be American Football
        logger.info("\n=== Fixing remaining American colleges ===")
        
        # Teams you saw that are definitely American Football
        american_football_teams = [
            'Puget Sound', 'California Lutheran', 'North Dakota', 'Valparaiso',
            'Notre Dame', 'Purdue', 'Liberty', 'James Madison', 'Ole Miss',
            'Tulane', 'Norwich', 'Maine Maritime', 'Coe', 'Nebraska Wesleyan',
            'The Citadel', 'Mercer', 'South Dakota', 'Drake', 'Illinois Wesleyan',
            'Elmhurst', 'Robert Morris', 'Dayton', 'Miami Florida', 'Dartmouth',
            'New Hampshire', 'Carson Newman', 'Lenoir Rhyne'
        ]
        
        af_count = 0
        for team in american_football_teams:
            markets = db.query(Market).filter(
                Market.sport == 'Soccer',
                (Market.home_team.like(f'%{team}%') | 
                 Market.away_team.like(f'%{team}%'))
            ).all()
            
            for market in markets:
                market.sport = 'American Football'
                # Fix nation if it's International
                if market.nation == 'International':
                    market.nation = 'United States'
                    market.governing_body = 'NCAA'
                af_count += 1
        
        logger.info(f"Fixed {af_count} American Football teams")
        
        # 3. Fix Fayetteville State specifically
        logger.info("\n=== Fixing Fayetteville State ===")
        fayetteville = db.query(Market).filter(
            Market.home_team.like('%Fayetteville State%') | 
            Market.away_team.like('%Fayetteville State%')
        ).all()
        
        fay_count = 0
        for market in fayetteville:
            # Fayetteville State is in North Carolina, USA
            if market.nation == 'Europe':
                market.nation = 'United States'
            
            # It's an HBCU that plays American Football
            if market.sport == 'Soccer':
                market.sport = 'American Football'
            
            if market.governing_body in ['UEFA', 'FIFA']:
                market.governing_body = 'NCAA'
            
            fay_count += 1
        
        logger.info(f"Fixed {fay_count} Fayetteville State markets")
        
        # Commit all changes
        db.commit()
        logger.info(f"\n✅ Total fixes committed: {fixed_count + af_count + fay_count}")

if __name__ == "__main__":
    fix_all_misclassifications()