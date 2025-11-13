#!/usr/bin/env python3
"""
Fix specific sport misclassifications based on team names
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

# Basketball teams (NBA, WNBA, etc)
BASKETBALL_TEAMS = [
    'Lakers', 'Celtics', 'Warriors', 'Nets', 'Bulls', 'Heat', 
    'Spurs', 'Mavericks', 'Rockets', 'Cavaliers', 'Clippers',
    'Golden State Valkyries', 'Minnesota Lynx', 'Indiana Fever',
    'Washington Mystics', 'Seattle Storm', 'Dallas Wings',
    'Las Vegas Aces', 'Phoenix Mercury', 'Chicago Sky'
]

# Baseball teams (MLB, etc)
BASEBALL_TEAMS = [
    'Yankees', 'Red Sox', 'Dodgers', 'Giants', 'Athletics',
    'Mariners', 'Orioles', 'Twins', 'Cardinals', 'Cubs',
    'Doosan Bears', 'Samsung Lions', 'SSG Landers', 
    'Kia Tigers', 'NC Dinos', 'LG Twins'
]

# Hockey teams
HOCKEY_TEAMS = [
    'Maple Leafs', 'Canadiens', 'Rangers', 'Bruins', 'Penguins',
    'Capitals', 'Lightning', 'Avalanche', 'Golden Knights'
]

# American Football teams
FOOTBALL_TEAMS = [
    'Patriots', 'Cowboys', 'Packers', 'Chiefs', 'Eagles',
    '49ers', 'Steelers', 'Ravens', 'Bills', 'Dolphins'
]

def fix_misclassifications():
    conn = psycopg2.connect(
        host=os.environ['PG_HOST'],
        port=os.environ['PG_PORT'],
        user=os.environ['PG_USER'],
        password=os.environ['PG_PASSWORD'],
        database=os.environ['PG_DB']
    )
    cur = conn.cursor()
    
    total_fixed = 0
    
    # Fix Basketball teams
    for team in BASKETBALL_TEAMS:
        cur.execute("""
            UPDATE market 
            SET sport = 'Basketball' 
            WHERE sport != 'Basketball' 
            AND (home_team ILIKE %s OR away_team ILIKE %s)
        """, (f'%{team}%', f'%{team}%'))
        fixed = cur.rowcount
        if fixed > 0:
            logger.info(f"Fixed {fixed} markets for basketball team: {team}")
            total_fixed += fixed
    
    # Fix Baseball teams
    for team in BASEBALL_TEAMS:
        cur.execute("""
            UPDATE market 
            SET sport = 'Baseball' 
            WHERE sport != 'Baseball' 
            AND (home_team ILIKE %s OR away_team ILIKE %s)
        """, (f'%{team}%', f'%{team}%'))
        fixed = cur.rowcount
        if fixed > 0:
            logger.info(f"Fixed {fixed} markets for baseball team: {team}")
            total_fixed += fixed
    
    # Fix Hockey teams
    for team in HOCKEY_TEAMS:
        cur.execute("""
            UPDATE market 
            SET sport = 'Hockey' 
            WHERE sport != 'Hockey' 
            AND (home_team ILIKE %s OR away_team ILIKE %s)
        """, (f'%{team}%', f'%{team}%'))
        fixed = cur.rowcount
        if fixed > 0:
            logger.info(f"Fixed {fixed} markets for hockey team: {team}")
            total_fixed += fixed
    
    # Fix American Football teams
    for team in FOOTBALL_TEAMS:
        cur.execute("""
            UPDATE market 
            SET sport = 'American Football' 
            WHERE sport != 'American Football' 
            AND (home_team ILIKE %s OR away_team ILIKE %s)
        """, (f'%{team}%', f'%{team}%'))
        fixed = cur.rowcount
        if fixed > 0:
            logger.info(f"Fixed {fixed} markets for football team: {team}")
            total_fixed += fixed
    
    # Fix league names
    # Replace "N/A" with more specific names
    cur.execute("""
        UPDATE market 
        SET league_name = 'NBA' 
        WHERE sport = 'Basketball' 
        AND league_name IN ('N/A', 'Regular Season')
        AND (home_team ILIKE ANY(ARRAY[%s]) OR away_team ILIKE ANY(ARRAY[%s]))
    """, (
        [f'%{t}%' for t in ['Lakers', 'Celtics', 'Warriors', 'Nets', 'Bulls', 'Heat']],
        [f'%{t}%' for t in ['Lakers', 'Celtics', 'Warriors', 'Nets', 'Bulls', 'Heat']]
    ))
    logger.info(f"Updated {cur.rowcount} NBA league names")
    
    cur.execute("""
        UPDATE market 
        SET league_name = 'WNBA' 
        WHERE sport = 'Basketball' 
        AND league_name IN ('N/A', 'Regular Season')
        AND (home_team ILIKE ANY(ARRAY['%Valkyries%', '%Lynx%', '%Fever%', '%Mystics%', '%Storm%', '%Wings%', '%Aces%', '%Mercury%', '%Sky%']))
    """)
    logger.info(f"Updated {cur.rowcount} WNBA league names")
    
    cur.execute("""
        UPDATE market 
        SET league_name = 'MLB' 
        WHERE sport = 'Baseball' 
        AND league_name IN ('N/A', 'Regular Season')
        AND (home_team ILIKE ANY(ARRAY['%Yankees%', '%Red Sox%', '%Dodgers%', '%Giants%', '%Athletics%', '%Mariners%', '%Orioles%']))
    """)
    logger.info(f"Updated {cur.rowcount} MLB league names")
    
    cur.execute("""
        UPDATE market 
        SET league_name = 'KBO' 
        WHERE sport = 'Baseball' 
        AND league_name IN ('N/A', 'Regular Season')
        AND (home_team ILIKE ANY(ARRAY['%Doosan%', '%Samsung%', '%SSG%', '%Kia%', '%NC Dinos%', '%LG Twins%']))
    """)
    logger.info(f"Updated {cur.rowcount} KBO league names")
    
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
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    fix_misclassifications()