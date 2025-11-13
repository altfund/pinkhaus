#!/usr/bin/env python3
"""
URGENT: Fix sport misclassifications showing in dashboard
These are clearly not soccer matches!
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

def fix_obvious_misclassifications():
    """Fix the most obvious sport misclassifications"""
    
    fixes = [
        # MLB - Baseball
        {
            'pattern': "MLB %",
            'correct_sport': 'Baseball',
            'reason': 'MLB is Major League Baseball'
        },
        # PFL/UFC - MMA
        {
            'pattern': "% PFL %",
            'correct_sport': 'MMA',
            'reason': 'PFL is Professional Fighters League'
        },
        {
            'pattern': "% UFC %",
            'correct_sport': 'MMA',
            'reason': 'UFC is Ultimate Fighting Championship'
        },
        # Esports
        {
            'pattern': "% Esports %",
            'correct_sport': 'Esports',
            'reason': 'Contains Esports in team name'
        },
        {
            'pattern': "% vs % Esports",
            'correct_sport': 'Esports',
            'reason': 'Esports team matchup'
        },
        {
            'pattern': "Esports % vs %",
            'correct_sport': 'Esports',
            'reason': 'Esports team matchup'
        },
        {
            'pattern': "% MLBB %",
            'correct_sport': 'Esports',
            'reason': 'MLBB is Mobile Legends: Bang Bang'
        },
        # Fighting sports by fighter names (firstname lastname format)
        {
            'pattern': "% vs %",
            'additional_check': 'fighter_names',
            'correct_sport': 'MMA',
            'reason': 'Fighter name format (not team names)'
        },
        # Basketball
        {
            'pattern': "% State vs % Christian",
            'correct_sport': 'Basketball',  
            'reason': 'College basketball teams'
        }
    ]
    
    with db_manager.get_db_session() as db:
        total_fixed = 0
        
        for fix in fixes:
            if 'additional_check' in fix and fix['additional_check'] == 'fighter_names':
                # Special handling for fighter names
                markets = db.query(Market).filter(
                    Market.sport == 'Soccer',
                    Market.home_team.like('% %'),  # First Last format
                    Market.away_team.like('% %'),  # First Last format
                    ~Market.home_team.like('% FC%'),
                    ~Market.home_team.like('% United%'),
                    ~Market.home_team.like('% City%'),
                    ~Market.away_team.like('% FC%'),
                    ~Market.away_team.like('% United%'),
                    ~Market.away_team.like('% City%'),
                    Market.league_name.like('%PFL%') | Market.league_name.like('%UFC%') | 
                    Market.league_name.like('%Championships%')
                ).all()
                
                # Check if these look like fighter names
                fighter_markets = []
                for market in markets:
                    # Fighter names typically don't have common team suffixes
                    if not any(suffix in market.home_team + market.away_team 
                              for suffix in ['FC', 'United', 'City', 'Real', 'Club', 'Team']):
                        fighter_markets.append(market)
                
                if fighter_markets:
                    for market in fighter_markets:
                        market.sport = fix['correct_sport']
                    db.commit()
                    logger.info(f"Fixed {len(fighter_markets)} markets: {fix['reason']}")
                    total_fixed += len(fighter_markets)
            else:
                # Standard pattern matching
                query = db.query(Market).filter(
                    Market.sport == 'Soccer',
                    Market.home_team.like(fix['pattern']) | 
                    Market.away_team.like(fix['pattern']) |
                    Market.league_name.like(fix['pattern'])
                )
                
                markets = query.all()
                if markets:
                    for market in markets:
                        market.sport = fix['correct_sport']
                    db.commit()
                    
                    logger.info(f"Fixed {len(markets)} markets: {fix['reason']}")
                    logger.info(f"  Pattern: {fix['pattern']} → {fix['correct_sport']}")
                    
                    # Show examples
                    for market in markets[:3]:
                        logger.info(f"  Example: {market.home_team} vs {market.away_team}")
                    
                    total_fixed += len(markets)
        
        logger.info(f"\n✅ Total markets fixed: {total_fixed}")
        
        # Check remaining suspicious "soccer" markets
        logger.info("\n=== Checking for remaining suspicious soccer markets ===")
        
        suspicious_patterns = [
            "SELECT source_id, home_team, away_team, league_name FROM market WHERE sport = 'Soccer' AND (home_team LIKE '%MLB%' OR away_team LIKE '%MLB%' OR league_name LIKE '%MLB%')",
            "SELECT source_id, home_team, away_team, league_name FROM market WHERE sport = 'Soccer' AND (home_team LIKE '%Esports%' OR away_team LIKE '%Esports%' OR league_name LIKE '%Esports%')",
            "SELECT source_id, home_team, away_team, league_name FROM market WHERE sport = 'Soccer' AND (league_name LIKE '%PFL%' OR league_name LIKE '%UFC%')",
            "SELECT source_id, home_team, away_team, league_name FROM market WHERE sport = 'Soccer' AND league_name LIKE '%MLBB%'"
        ]
        
        for sql in suspicious_patterns:
            result = db.execute(sql)
            rows = result.fetchall()
            if rows:
                logger.warning(f"Still found {len(rows)} suspicious soccer markets:")
                for row in rows[:5]:
                    logger.warning(f"  {row[1]} vs {row[2]} ({row[3]})")

if __name__ == "__main__":
    fix_obvious_misclassifications()