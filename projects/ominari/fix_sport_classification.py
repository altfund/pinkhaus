#!/usr/bin/env python3
"""Fix sport classification to be deterministic based on league names and team names"""

import os
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Deterministic sport classification rules
SPORT_KEYWORDS = {
    'Soccer': {
        'leagues': ['premier league', 'la liga', 'bundesliga', 'serie a', 'ligue 1', 
                   'champions league', 'europa league', 'world cup', 'euro', 'copa america',
                   'mls', 'championship', 'eredivisie', 'primeira liga', 'super league',
                   'uefa', 'fifa', 'fa cup', 'carabao cup', 'league cup', 'soccer', 'football'],
        'teams': ['united', 'city', 'fc', 'cf', 'real', 'atletico', 'juventus', 'milan',
                 'inter', 'bayern', 'dortmund', 'psg', 'marseille', 'ajax', 'benfica', 'porto']
    },
    'Basketball': {
        'leagues': ['nba', 'ncaa', 'euroleague', 'basketball', 'fiba', 'wnba', 'g league'],
        'teams': ['lakers', 'celtics', 'bulls', 'warriors', 'heat', 'knicks', 'nets', 
                 'bucks', 'suns', 'mavericks', '76ers', 'sixers']
    },
    'American Football': {
        'leagues': ['nfl', 'ncaaf', 'football', 'super bowl', 'afc', 'nfc'],
        'teams': ['patriots', 'cowboys', 'packers', 'chiefs', 'bills', 'eagles', 
                 'steelers', 'ravens', 'broncos', '49ers', 'niners']
    },
    'Baseball': {
        'leagues': ['mlb', 'world series', 'american league', 'national league', 'baseball'],
        'teams': ['yankees', 'red sox', 'dodgers', 'giants', 'cubs', 'mets', 'astros',
                 'braves', 'cardinals', 'phillies']
    },
    'Hockey': {
        'leagues': ['nhl', 'hockey', 'stanley cup', 'ice hockey', 'iihf'],
        'teams': ['rangers', 'bruins', 'maple leafs', 'canadiens', 'blackhawks', 
                 'penguins', 'capitals', 'lightning', 'avalanche', 'oilers']
    },
    'Tennis': {
        'leagues': ['atp', 'wta', 'grand slam', 'wimbledon', 'us open', 'french open',
                   'australian open', 'tennis', 'davis cup'],
        'teams': []  # Tennis is individual sport
    },
    'Golf': {
        'leagues': ['pga', 'lpga', 'masters', 'us open', 'british open', 'golf', 'ryder cup'],
        'teams': []  # Golf is individual sport
    },
    'Boxing': {
        'leagues': ['boxing', 'wba', 'wbc', 'ibf', 'wbo', 'heavyweight', 'middleweight',
                   'lightweight', 'welterweight'],
        'teams': []  # Boxing is individual sport
    },
    'MMA': {
        'leagues': ['ufc', 'mma', 'bellator', 'one championship', 'pfl'],
        'teams': []  # MMA is individual sport
    },
    'Cricket': {
        'leagues': ['cricket', 'ipl', 't20', 'test match', 'odi', 'world cup cricket',
                   'ashes', 'big bash'],
        'teams': ['india', 'australia', 'england', 'pakistan', 'south africa', 'new zealand']
    },
    'Rugby': {
        'leagues': ['rugby', 'six nations', 'rugby world cup', 'super rugby', 'premiership rugby'],
        'teams': ['all blacks', 'springboks', 'wallabies']
    },
    'Esports': {
        'leagues': ['lol', 'league of legends', 'dota', 'cs:go', 'csgo', 'valorant', 
                   'overwatch', 'rocket league', 'esports', 'e-sports'],
        'teams': ['fnatic', 'g2', 'liquid', 'cloud9', 'tsm', 'navi', 'faze', 'vitality',
                 'astralis', 't1', 'dwg', 'edg', 'rng']
    }
}

def normalize_text(text):
    """Normalize text for matching"""
    if not text:
        return ''
    return text.lower().strip()

def determine_sport(league_name, home_team, away_team):
    """Determine sport based on league and team names"""
    league_norm = normalize_text(league_name)
    home_norm = normalize_text(home_team)
    away_norm = normalize_text(away_team)
    
    # Check each sport's keywords
    for sport, keywords in SPORT_KEYWORDS.items():
        # Check league name first (highest priority)
        for league_keyword in keywords['leagues']:
            if league_keyword in league_norm:
                return sport
        
        # Check team names
        for team_keyword in keywords['teams']:
            if team_keyword in home_norm or team_keyword in away_norm:
                return sport
    
    # Special case for esports teams with numbers/special chars
    if any(char.isdigit() or char in ['_', '-', '.'] for char in home_team + away_team):
        if not any(word in league_norm for word in ['mlb', 'nba', 'nfl', 'nhl']):  # Not major US sports
            return 'Esports'
    
    # Default based on common patterns
    if 'vs' in league_norm and any(word in league_norm for word in ['winner', 'champion']):
        return 'Other'
    
    # Ultimate default
    return 'Soccer'  # Most common sport globally

def fix_sport_classifications():
    """Update all markets with deterministic sport classifications"""
    logger.info("🏆 Fixing sport classifications...")
    
    with db_manager.get_db_session() as db:
        # Get all markets
        markets = db.query(Market).all()
        
        logger.info(f"Found {len(markets)} markets to classify")
        
        # Track changes
        sport_counts = {}
        changes = 0
        
        for market in markets:
            old_sport = market.sport
            new_sport = determine_sport(market.league_name, market.home_team, market.away_team)
            
            if old_sport != new_sport:
                market.sport = new_sport
                changes += 1
                logger.info(f"Changed: {market.home_team} vs {market.away_team} from {old_sport} to {new_sport}")
            
            sport_counts[new_sport] = sport_counts.get(new_sport, 0) + 1
        
        db.commit()
        
        logger.info(f"\n✅ Updated {changes} markets")
        logger.info("\n📊 Sport distribution:")
        for sport, count in sorted(sport_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {sport}: {count} markets")

def test_classification():
    """Test some classifications"""
    test_cases = [
        ("Premier League", "Manchester United", "Liverpool"),
        ("NBA Regular Season", "Lakers", "Celtics"),
        ("MLB", "Yankees", "Red Sox"),
        ("Valorant Champions", "Fnatic", "G2 Esports"),
        ("", "AMKAL ESPORTS", "Oramond"),
        ("World Series Winner", "MLB", "American League Winner"),
        ("UEFA Champions League", "Real Madrid", "Bayern Munich"),
        ("CS:GO Major", "NaVi", "FaZe Clan"),
    ]
    
    logger.info("\n🧪 Testing classifications:")
    for league, home, away in test_cases:
        sport = determine_sport(league, home, away)
        logger.info(f"{home} vs {away} ({league}) -> {sport}")

if __name__ == "__main__":
    logger.info("🎯 DETERMINISTIC SPORT CLASSIFICATION")
    logger.info("=" * 60)
    
    # Test first
    test_classification()
    
    # Then fix all markets
    logger.info("\n" + "=" * 60)
    fix_sport_classifications()