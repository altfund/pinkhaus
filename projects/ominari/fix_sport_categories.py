#!/usr/bin/env python3
"""
Fix sport categorization for markets
Using team names and league information
"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

import psycopg2
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Esports indicators
ESPORTS_INDICATORS = [
    'esport', 'gaming', 'team liquid', 'fnatic', 'g2 ', 'cloud9', 'c9',
    'vitality', 'natus vincere', 'navi', 'oxygen esports', 'darkzero',
    'wildcard gaming', 'shopify rebellion', 'spacestation', 'lfo',
    'rrq', 'dewa united esports', 'fish123', 'wopa', 'nexus', 'envy',
    'chiefs esports', 'team cruelty', 'jijiehao', 'esc gaming'
]

# League to sport mappings
LEAGUE_SPORT_MAP = {
    # Esports leagues
    'MPL': 'eSports',
    'North America League': 'eSports',
    'ESL': 'eSports',
    'Asia-Pacific League': 'eSports',
    'Proving Grounds': 'eSports',
    'United21': 'eSports',
    'VCT': 'eSports',
    'LCS': 'eSports',
    'LEC': 'eSports',
    'LPL': 'eSports',
    'LCK': 'eSports',
    
    # Soccer leagues
    'Premier League': 'Soccer',
    'La Liga': 'Soccer',
    'Serie A': 'Soccer',
    'Bundesliga': 'Soccer',
    'Ligue 1': 'Soccer',
    'Eredivisie': 'Soccer',
    'Championship': 'Soccer',
    'Champions League': 'Soccer',
    'Europa League': 'Soccer',
    'MLS': 'Soccer',
    'Liga MX': 'Soccer',
    'Brasileirão': 'Soccer',
    'Argentine Primera': 'Soccer',
    'Scottish Premiership': 'Soccer',
    'Primeira Liga': 'Soccer',
    
    # American Football
    'NFL': 'Football',
    'NCAAF': 'College Sports',
    
    # Basketball
    'NBA': 'Basketball',
    'NCAAB': 'College Sports',
    'Euroleague': 'Basketball',
    
    # Hockey
    'NHL': 'Hockey',
    'KHL': 'Hockey',
    
    # Baseball
    'MLB': 'Baseball',
    'NPB': 'Baseball',
    
    # Table Tennis
    'WTT': 'TableTennis',
    'ITTF': 'TableTennis',
    'Pro Tour': 'TableTennis',
    'TT Elite': 'TableTennis',
    
    # Tennis
    'ATP': 'Tennis',
    'WTA': 'Tennis',
    'Grand Slam': 'Tennis',
    
    # Golf
    'PGA': 'Golf',
    'European Tour': 'Golf',
    
    # Fighting/MMA
    'UFC': 'Fighting',
    'Bellator': 'Fighting',
    'ONE Championship': 'Fighting',
    'PFL': 'Fighting',
}

# Team name patterns for sports
TEAM_PATTERNS = {
    'TableTennis': re.compile(r'\b(player|singles|doubles)\b', re.I),
    'Tennis': re.compile(r'\b(atp|wta|open|grand slam)\b', re.I),
    'Golf': re.compile(r'\b(pga|tour|open|masters|championship)\b', re.I),
    'Fighting': re.compile(r'\b(ufc|bellator|one|pfl|mma|boxing)\b', re.I),
    'Soccer': re.compile(r'\b(fc|cf|afc|united|city|real|atletico|juventus|bayern|psg|chelsea|arsenal|liverpool|barcelona)\b', re.I),
    'Basketball': re.compile(r'\b(lakers|celtics|warriors|heat|bulls|knicks|nets|clippers|bucks|suns)\b', re.I),
    'Football': re.compile(r'\b(patriots|chiefs|bills|packers|cowboys|49ers|eagles|steelers|ravens|saints)\b', re.I),
    'Baseball': re.compile(r'\b(yankees|red sox|dodgers|giants|cubs|cardinals|astros|braves|mets|phillies)\b', re.I),
    'Hockey': re.compile(r'\b(rangers|bruins|penguins|blackhawks|canadiens|maple leafs|oilers|avalanche|lightning|capitals)\b', re.I)
}

def determine_sport(home_team, away_team, league_name, current_sport):
    """Determine the correct sport for a market"""
    
    # Check if it's esports
    teams_lower = f"{home_team} {away_team}".lower()
    for indicator in ESPORTS_INDICATORS:
        if indicator in teams_lower:
            return 'eSports'
    
    # Check league mapping
    if league_name:
        for league_pattern, sport in LEAGUE_SPORT_MAP.items():
            if league_pattern.lower() in league_name.lower():
                return sport
    
    # Check team name patterns
    combined_names = f"{home_team} {away_team}"
    for sport, pattern in TEAM_PATTERNS.items():
        if pattern.search(combined_names):
            return sport
    
    # Special cases
    if any(char in combined_names for char in ['vs', 'v.', '-']) and len(home_team.split()) == 2 and len(away_team.split()) == 2:
        # Likely individual names (tennis, table tennis, fighting)
        if 'table' in league_name.lower() or 'tt' in league_name.lower():
            return 'TableTennis'
        elif 'ufc' in league_name.lower() or 'boxing' in league_name.lower():
            return 'Fighting'
        elif 'atp' in league_name.lower() or 'wta' in league_name.lower():
            return 'Tennis'
    
    # If still unknown and was originally Soccer, keep it as Soccer unless we're sure it's not
    if current_sport == 'Soccer' and 'Unknown' not in [home_team, away_team]:
        # Check if it looks like a real soccer team
        soccer_words = ['fc', 'cf', 'united', 'city', 'athletic', 'sporting', 'real', 'inter']
        if any(word in teams_lower for word in soccer_words):
            return 'Soccer'
    
    # Default to current sport if we can't determine
    return current_sport if current_sport != 'Unknown' else 'Soccer'


def fix_sports():
    """Fix sport categorizations"""
    conn = psycopg2.connect(
        host='localhost',
        port=5999,
        database='ominari_production',
        user='ominari_user',
        password='ominari_2025_secure'
    )
    cur = conn.cursor()
    
    # Get all markets that need review
    logger.info("Fetching markets to review...")
    cur.execute("""
        SELECT source_id, home_team, away_team, league_name, sport
        FROM market
        WHERE maturity_date > NOW()
        AND (sport = 'Unknown' OR sport = 'Soccer')
    """)
    
    markets = cur.fetchall()
    logger.info(f"Found {len(markets)} markets to review")
    
    updates = {
        'eSports': 0,
        'Soccer': 0,
        'TableTennis': 0,
        'Basketball': 0,
        'Football': 0,
        'Other': 0
    }
    
    for market in markets:
        source_id, home_team, away_team, league_name, current_sport = market
        new_sport = determine_sport(home_team, away_team, league_name or '', current_sport)
        
        if new_sport != current_sport:
            cur.execute("""
                UPDATE market
                SET sport = %s
                WHERE source_id = %s
            """, (new_sport, source_id))
            
            updates[new_sport] = updates.get(new_sport, 0) + 1
            
            if updates[new_sport] <= 5:  # Show first 5 of each type
                logger.info(f"  {current_sport} → {new_sport}: {home_team} vs {away_team}")
    
    conn.commit()
    
    # Show summary
    logger.info("\n=== Sport Categorization Fixes ===")
    for sport, count in updates.items():
        if count > 0:
            logger.info(f"  Updated to {sport}: {count}")
    
    # Show new distribution
    cur.execute("""
        SELECT sport, COUNT(*) as count
        FROM market
        WHERE maturity_date > NOW()
        GROUP BY sport
        ORDER BY count DESC
    """)
    
    logger.info("\n=== New Sport Distribution ===")
    for row in cur.fetchall():
        logger.info(f"  {row[0]}: {row[1]}")
    
    cur.close()
    conn.close()


if __name__ == "__main__":
    fix_sports()