#!/usr/bin/env python3
"""
Fix league names based on sport, teams, and patterns
"""
import os
import re

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

# League mappings by sport
SOCCER_LEAGUES = {
    # Top European Leagues
    'teams': {
        'Real Madrid|Barcelona|Atletico Madrid|Sevilla|Valencia': 'La Liga',
        'Bayern Munich|Borussia Dortmund|RB Leipzig|Bayer Leverkusen': 'Bundesliga',
        'Manchester United|Liverpool|Chelsea|Arsenal|Manchester City': 'Premier League',
        'Juventus|AC Milan|Inter Milan|Roma|Napoli': 'Serie A',
        'PSG|Lyon|Marseille|Monaco|Lille': 'Ligue 1',
        'Ajax|PSV|Feyenoord|AZ Alkmaar': 'Eredivisie',
        'Benfica|Porto|Sporting': 'Primeira Liga',
        'Celtic|Rangers': 'Scottish Premiership',
    },
    # South American
    'patterns': {
        r'Flamengo|Palmeiras|Santos|Corinthians': 'Brasileirão',
        r'River Plate|Boca Juniors|Racing|Independiente': 'Primera División Argentina',
        r'Colo-Colo|Universidad de Chile': 'Primera División de Chile',
        r'Nacional|Peñarol': 'Primera División Uruguay',
    }
}

BASKETBALL_LEAGUES = {
    'nba_teams': [
        'Lakers', 'Celtics', 'Warriors', 'Nets', 'Bulls', 'Heat', 'Spurs',
        'Mavericks', 'Rockets', 'Cavaliers', 'Clippers', 'Knicks', 'Sixers',
        'Nuggets', 'Bucks', 'Suns', 'Jazz', 'Trail Blazers', 'Kings', 'Pistons',
        'Pacers', 'Hornets', 'Hawks', 'Wizards', 'Magic', 'Pelicans', 'Grizzlies',
        'Timberwolves', 'Thunder', 'Raptors'
    ],
    'wnba_teams': [
        'Aces', 'Sky', 'Sun', 'Storm', 'Mercury', 'Mystics', 'Liberty',
        'Fever', 'Lynx', 'Wings', 'Dream', 'Sparks', 'Valkyries'
    ],
    'euroleague': [
        'Real Madrid', 'Barcelona', 'CSKA Moscow', 'Fenerbahce', 'Olympiacos',
        'Panathinaikos', 'Maccabi Tel Aviv', 'Bayern Munich', 'Alba Berlin'
    ]
}

ESPORTS_LEAGUES = {
    'CCT': 'CCT Europe',
    'ESEA': 'ESEA League',
    'ESL': 'ESL Pro League',
    'BLAST': 'BLAST Premier',
    'PGL': 'PGL Major',
    'VCT': 'VCT Champions',
    'MPL': 'Mobile Legends Pro League',
    'LEC': 'League of Legends European Championship',
    'LCS': 'League of Legends Championship Series',
    'DACH': 'DACH Masters',
}

def fix_league_names():
    conn = psycopg2.connect(
        host=os.environ['PG_HOST'],
        port=os.environ['PG_PORT'],
        user=os.environ['PG_USER'],
        password=os.environ['PG_PASSWORD'],
        database=os.environ['PG_DB']
    )
    cur = conn.cursor()
    
    total_fixed = 0
    
    # Fix Soccer leagues
    logger.info("=== Fixing Soccer Leagues ===")
    
    # European leagues by team patterns
    for pattern, league in SOCCER_LEAGUES['teams'].items():
        cur.execute("""
            UPDATE market 
            SET league_name = %s 
            WHERE sport = 'Soccer' 
            AND league_name IN ('Regular Season', 'N/A', '')
            AND (home_team ~* %s OR away_team ~* %s)
        """, (league, pattern, pattern))
        fixed = cur.rowcount
        if fixed > 0:
            logger.info(f"  {league}: {fixed} markets updated")
            total_fixed += fixed
    
    # Fix Basketball leagues
    logger.info("\n=== Fixing Basketball Leagues ===")
    
    # NBA
    nba_pattern = '|'.join(BASKETBALL_LEAGUES['nba_teams'])
    cur.execute("""
        UPDATE market 
        SET league_name = 'NBA' 
        WHERE sport = 'Basketball' 
        AND league_name IN ('Regular Season', 'N/A', '')
        AND (home_team ~* %s OR away_team ~* %s)
    """, (nba_pattern, nba_pattern))
    logger.info(f"  NBA: {cur.rowcount} markets updated")
    total_fixed += cur.rowcount
    
    # WNBA
    wnba_pattern = '|'.join(BASKETBALL_LEAGUES['wnba_teams'])
    cur.execute("""
        UPDATE market 
        SET league_name = 'WNBA' 
        WHERE sport = 'Basketball' 
        AND league_name IN ('Regular Season', 'N/A', '')
        AND (home_team ~* %s OR away_team ~* %s)
    """, (wnba_pattern, wnba_pattern))
    logger.info(f"  WNBA: {cur.rowcount} markets updated")
    total_fixed += cur.rowcount
    
    # Fix Esports leagues
    logger.info("\n=== Fixing Esports Leagues ===")
    
    for pattern, league in ESPORTS_LEAGUES.items():
        cur.execute("""
            UPDATE market 
            SET league_name = %s 
            WHERE sport = 'Esports' 
            AND (league_name IN ('Regular Season', 'N/A', '') 
                 OR league_name ILIKE %s)
        """, (league, f'%{pattern}%'))
        fixed = cur.rowcount
        if fixed > 0:
            logger.info(f"  {league}: {fixed} markets updated")
            total_fixed += fixed
    
    # Fix misclassified Esports (in Soccer/Golf)
    logger.info("\n=== Fixing Misclassified Esports Leagues ===")
    
    for pattern, league in ESPORTS_LEAGUES.items():
        # First update sport to Esports
        cur.execute("""
            UPDATE market 
            SET sport = 'Esports', league_name = %s
            WHERE league_name ILIKE %s
            AND sport != 'Esports'
        """, (league, f'%{pattern}%'))
        fixed = cur.rowcount
        if fixed > 0:
            logger.info(f"  Moved {fixed} {pattern} markets to Esports")
            total_fixed += fixed
    
    # Fix Cricket leagues
    logger.info("\n=== Fixing Cricket Leagues ===")
    
    cricket_patterns = {
        'T20|Twenty20': 'T20 League',
        'IPL|Indian Premier': 'Indian Premier League',
        'BBL|Big Bash': 'Big Bash League',
        'CPL|Caribbean': 'Caribbean Premier League',
        'PSL|Pakistan Super': 'Pakistan Super League',
    }
    
    for pattern, league in cricket_patterns.items():
        cur.execute("""
            UPDATE market 
            SET league_name = %s 
            WHERE sport = 'Cricket' 
            AND league_name IN ('Regular Season', 'N/A', '')
            AND (home_team ~* %s OR away_team ~* %s OR league_name ~* %s)
        """, (league, pattern, pattern, pattern))
        fixed = cur.rowcount
        if fixed > 0:
            logger.info(f"  {league}: {fixed} markets updated")
            total_fixed += fixed
    
    # Fix American Football leagues
    logger.info("\n=== Fixing American Football Leagues ===")
    
    nfl_teams = ['Patriots', 'Cowboys', 'Packers', 'Chiefs', 'Eagles', '49ers', 
                 'Steelers', 'Ravens', 'Bills', 'Dolphins', 'Jets', 'Bengals',
                 'Browns', 'Titans', 'Colts', 'Jaguars', 'Texans', 'Broncos',
                 'Raiders', 'Chargers', 'Cardinals', 'Rams', 'Seahawks', 'Buccaneers',
                 'Saints', 'Falcons', 'Panthers', 'Vikings', 'Bears', 'Lions', 'Giants']
    
    nfl_pattern = '|'.join(nfl_teams)
    cur.execute("""
        UPDATE market 
        SET league_name = 'NFL' 
        WHERE sport = 'American Football' 
        AND league_name IN ('Regular Season', 'N/A', '')
        AND (home_team ~* %s OR away_team ~* %s)
    """, (nfl_pattern, nfl_pattern))
    logger.info(f"  NFL: {cur.rowcount} markets updated")
    total_fixed += cur.rowcount
    
    # Generic soccer leagues - try to identify by region
    logger.info("\n=== Fixing Generic Soccer Leagues by Region ===")
    
    # If still generic, at least make it more specific
    cur.execute("""
        UPDATE market 
        SET league_name = CASE
            WHEN home_team ~* 'United|City|FC|Athletic|Real|CF' THEN 'European Football'
            WHEN home_team ~* 'SC|FK|SK|BK|IK' THEN 'European Football'
            WHEN home_team ~* 'Esports?|Gaming|eSports' THEN 'Esports League'
            ELSE 'International Football'
        END
        WHERE sport = 'Soccer' 
        AND league_name IN ('Regular Season', 'N/A')
    """)
    logger.info(f"  Regional classifications: {cur.rowcount} markets updated")
    total_fixed += cur.rowcount
    
    conn.commit()
    
    logger.info(f"\n✅ Total markets fixed: {total_fixed}")
    
    # Show remaining generic leagues
    cur.execute("""
        SELECT sport, league_name, COUNT(*) 
        FROM market 
        WHERE league_name IN ('Regular Season', 'N/A', '')
        GROUP BY sport, league_name
        ORDER BY COUNT(*) DESC
    """)
    
    remaining = cur.fetchall()
    if remaining:
        logger.info("\n⚠️  Remaining generic leagues:")
        for sport, league, count in remaining:
            logger.info(f"  {sport} - '{league}': {count} markets")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    fix_league_names()