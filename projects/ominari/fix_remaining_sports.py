#!/usr/bin/env python3
"""
Fix remaining sport misclassifications based on additional team names
"""
import os

# Set environment for PostgreSQL on port 5999
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

import psycopg2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Additional Baseball teams
BASEBALL_TEAMS = [
    'Reds', 'Brewers', 'Rays', 'Rangers', 'White Sox', 'Tigers',
    'Marlins', 'Rockies', 'Padres', 'Pirates', 'Angels', 'Astros',
    'Blue Jays', 'Braves', 'Nationals', 'Phillies', 'Mets',
    'Diamondbacks', 'Royals', 'Guardians'
]

# Additional Hockey teams  
HOCKEY_TEAMS = [
    'Lightning', 'Flames', 'Oilers', 'Senators', 'Sabres',
    'Devils', 'Islanders', 'Flyers', 'Hurricanes', 'Panthers',
    'Red Wings', 'Blue Jackets', 'Blackhawks', 'Wild', 'Jets',
    'Stars', 'Blues', 'Predators', 'Sharks', 'Ducks', 'Kings',
    'Kraken', 'Coyotes', 'Utah Hockey Club'
]

# Handball teams (often mistaken for soccer)
HANDBALL_TEAMS = [
    'MT Melsungen', 'TBV Lemgo', 'THW Kiel', 'SG Flensburg',
    'Rhein-Neckar Löwen', 'SC Magdeburg', 'Füchse Berlin'
]

def fix_remaining_sports():
    conn = psycopg2.connect(
        host=os.environ['PG_HOST'],
        port=os.environ['PG_PORT'],
        user=os.environ['PG_USER'],
        password=os.environ['PG_PASSWORD'],
        database=os.environ['PG_DB']
    )
    cur = conn.cursor()
    
    total_fixed = 0
    
    # Fix Baseball teams
    cur.execute("""
        UPDATE market 
        SET sport = 'Baseball' 
        WHERE sport != 'Baseball' 
        AND (home_team ILIKE ANY(ARRAY['%Reds%', '%Brewers%', '%Rays%', '%Rangers%', '%Sox%', '%Tigers%',
                                      '%Marlins%', '%Rockies%', '%Padres%', '%Pirates%', '%Angels%', '%Astros%',
                                      '%Jays%', '%Braves%', '%Nationals%', '%Phillies%', '%Mets%',
                                      '%Diamondbacks%', '%Royals%', '%Guardians%'])
            OR away_team ILIKE ANY(ARRAY['%Reds%', '%Brewers%', '%Rays%', '%Rangers%', '%Sox%', '%Tigers%',
                                      '%Marlins%', '%Rockies%', '%Padres%', '%Pirates%', '%Angels%', '%Astros%',
                                      '%Jays%', '%Braves%', '%Nationals%', '%Phillies%', '%Mets%',
                                      '%Diamondbacks%', '%Royals%', '%Guardians%']))
    """)
    fixed = cur.rowcount
    if fixed > 0:
        logger.info(f"Fixed {fixed} markets for baseball teams")
        total_fixed += fixed
    
    # Fix Hockey teams
    # First, let's check what teams contain "Rays" or "Rangers" to avoid confusion
    cur.execute("""
        UPDATE market 
        SET sport = 'Hockey' 
        WHERE sport = 'Hockey'  -- Keep existing hockey
           OR (sport != 'Baseball' AND sport != 'Hockey' 
               AND ((home_team ILIKE '%Lightning%' OR away_team ILIKE '%Lightning%')
                 OR (home_team ILIKE '%Tampa Bay%' AND away_team NOT ILIKE '%Rays%' AND home_team NOT ILIKE '%Rays%')))
    """)
    logger.info(f"Fixed {cur.rowcount} Lightning hockey markets")
    
    # Fix specific baseball teams that might be confused
    cur.execute("""
        UPDATE market 
        SET sport = 'Baseball' 
        WHERE (home_team = 'Tampa Bay Rays' OR away_team = 'Tampa Bay Rays'
            OR home_team = 'Texas Rangers' OR away_team = 'Texas Rangers')
    """)
    logger.info(f"Fixed {cur.rowcount} specific baseball teams")
    
    # Fix Handball teams
    for team in HANDBALL_TEAMS:
        cur.execute("""
            UPDATE market 
            SET sport = 'Handball' 
            WHERE (home_team ILIKE %s OR away_team ILIKE %s)
        """, (f'%{team}%', f'%{team}%'))
        fixed = cur.rowcount
        if fixed > 0:
            logger.info(f"Fixed {fixed} markets for handball team: {team}")
            total_fixed += fixed
    
    # Update league names for newly corrected sports
    cur.execute("""
        UPDATE market 
        SET league_name = 'MLB' 
        WHERE sport = 'Baseball' 
        AND league_name IN ('N/A', 'Regular Season', 'Soccer League')
        AND home_team NOT ILIKE ANY(ARRAY['%Doosan%', '%Samsung%', '%SSG%', '%Kia%', '%NC Dinos%', '%LG Twins%'])
    """)
    logger.info(f"Updated {cur.rowcount} more MLB league names")
    
    cur.execute("""
        UPDATE market 
        SET league_name = 'NHL' 
        WHERE sport = 'Hockey' 
        AND league_name IN ('N/A', 'Regular Season', 'Soccer League')
    """)
    logger.info(f"Updated {cur.rowcount} NHL league names")
    
    cur.execute("""
        UPDATE market 
        SET league_name = 'Handball Bundesliga' 
        WHERE sport = 'Handball' 
        AND league_name IN ('N/A', 'Regular Season', 'Soccer League')
    """)
    logger.info(f"Updated {cur.rowcount} Handball league names")
    
    conn.commit()
    
    logger.info(f"\n✅ Total markets fixed: {total_fixed}")
    
    # Show updated distribution
    cur.execute("""
        SELECT sport, COUNT(*) as count 
        FROM market 
        WHERE sport IS NOT NULL 
        GROUP BY sport 
        ORDER BY count DESC
    """)
    
    logger.info("\n📊 Updated Sport Distribution:")
    for row in cur.fetchall():
        logger.info(f"  {row[0]}: {row[1]} markets")
    
    # Show remaining N/A leagues
    cur.execute("""
        SELECT sport, COUNT(*) as count 
        FROM market 
        WHERE league_name IN ('N/A', 'Regular Season')
        GROUP BY sport 
        ORDER BY count DESC
        LIMIT 10
    """)
    
    logger.info("\n⚠️  Remaining generic leagues by sport:")
    for row in cur.fetchall():
        logger.info(f"  {row[0]}: {row[1]} markets with generic leagues")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    fix_remaining_sports()