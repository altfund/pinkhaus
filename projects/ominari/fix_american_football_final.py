#!/usr/bin/env python3
"""
Final fix for American Football teams misclassified as Soccer
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

def fix_american_football():
    """Fix American college and professional football teams"""
    
    with db_manager.get_db_session() as db:
        # American college patterns
        college_patterns = [
            '%State%', '%University%', '%College%', 
            'Georgia Tech', 'Temple', 'Northern Illinois',
            'Northern Arizona', 'Incarnate Word', 'San Jose State',
            'Idaho', 'Missouri Southern', 'Northwest Missouri',
            'Fort Hays', 'Pittsburg State', 'East Tennessee',
            'Elon', 'Morehead', 'Kentucky Christian',
            'Mississippi State', 'UNLV', 'Nevada', 'Wyoming',
            'Colorado State', 'Fresno State', 'San Diego State',
            'Utah State', 'Air Force', 'Army', 'Navy',
            'Boise State', 'Hawaii', 'New Mexico',
            # Common college abbreviations
            'UCLA', 'USC', 'LSU', 'TCU', 'SMU', 'BYU',
            'UAB', 'UTEP', 'UTSA', 'FIU', 'FAU', 'WKU',
            'MIT', 'NYU', 'USF', 'UCF', 'ECU', 'WCU'
        ]
        
        american_football_count = 0
        
        # Fix by patterns
        for pattern in college_patterns:
            markets = db.query(Market).filter(
                Market.sport == 'Soccer',
                (Market.home_team.like(f'%{pattern}%') | 
                 Market.away_team.like(f'%{pattern}%')),
                # Exclude actual international soccer
                ~Market.home_team.like('%FC%'),
                ~Market.away_team.like('%FC%'),
                ~Market.home_team.like('%United%'),
                ~Market.away_team.like('%United%'),
                ~Market.home_team.like('%City%'),
                ~Market.away_team.like('%City%')
            ).all()
            
            for market in markets:
                # Double check it's not actual soccer
                home_lower = market.home_team.lower()
                away_lower = market.away_team.lower()
                
                # Skip if it has soccer indicators
                soccer_terms = ['fc', 'united', 'city', 'town', 'athletic', 'rovers', 
                               'wanderers', 'hotspur', 'albion', 'real', 'inter']
                
                is_soccer = any(term in home_lower or term in away_lower for term in soccer_terms)
                
                if not is_soccer:
                    market.sport = 'American Football'
                    american_football_count += 1
        
        # Also check International Football league specifically
        intl_football = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.league_name == 'International Football'
        ).all()
        
        for market in intl_football:
            # These are likely American Football in "International Football" league
            home_lower = market.home_team.lower()
            away_lower = market.away_team.lower()
            
            # Check if they're American teams
            american_indicators = ['state', 'university', 'college', 'tech', 'southern',
                                 'northern', 'western', 'eastern', 'central']
            
            has_american = any(ind in home_lower or ind in away_lower for ind in american_indicators)
            
            if has_american:
                market.sport = 'American Football'
                american_football_count += 1
        
        if american_football_count > 0:
            db.commit()
            logger.info(f"✅ Fixed {american_football_count} American Football markets")
        else:
            logger.info("No additional American Football markets found")
        
        # Show sample of remaining soccer
        logger.info("\n=== Remaining Soccer Sample ===")
        soccer_sample = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.nation.in_(['England', 'International', 'Europe'])
        ).limit(20).all()
        
        for market in soccer_sample:
            logger.info(f"  {market.home_team} vs {market.away_team} ({market.league_name}) - {market.nation}")

if __name__ == "__main__":
    fix_american_football()