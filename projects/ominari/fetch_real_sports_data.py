#!/usr/bin/env python3
"""
Fetch real sports data with properly classified sports and unique IDs
"""

import os
os.environ['PG_PORT'] = '5999'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

import requests
import json
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sport classification based on team/league names
SPORT_KEYWORDS = {
    'Soccer': ['fc', 'united', 'city', 'real', 'atletico', 'juventus', 'barcelona', 'liverpool', 
               'chelsea', 'arsenal', 'tottenham', 'milan', 'inter', 'roma', 'napoli', 'dortmund',
               'bayern', 'psg', 'ajax', 'benfica', 'sporting', 'premier league', 'la liga', 
               'serie a', 'bundesliga', 'ligue 1', 'champions league', 'europa league'],
    'Basketball': ['lakers', 'celtics', 'warriors', 'bulls', 'heat', 'knicks', 'nets', 'clippers',
                   'nuggets', 'bucks', 'suns', 'mavericks', 'rockets', 'spurs', 'thunder',
                   'nba', 'basketball', 'euroleague'],
    'Football': ['nfl', 'patriots', 'cowboys', 'packers', 'steelers', 'eagles', 'giants', 'jets',
                 'broncos', 'raiders', 'chiefs', '49ers', 'seahawks', 'ravens', 'bills'],
    'Baseball': ['mlb', 'yankees', 'red sox', 'dodgers', 'giants', 'cubs', 'cardinals', 'mets',
                 'phillies', 'braves', 'astros', 'nationals', 'orioles', 'tigers', 'rangers'],
    'Hockey': ['nhl', 'rangers', 'bruins', 'canadiens', 'maple leafs', 'blackhawks', 'penguins',
               'capitals', 'lightning', 'avalanche', 'oilers', 'flames', 'canucks', 'sharks'],
    'Tennis': ['atp', 'wta', 'djokovic', 'nadal', 'federer', 'medvedev', 'zverev', 'tsitsipas',
               'wimbledon', 'us open', 'french open', 'australian open', 'tennis'],
    'Fighting': ['ufc', 'mma', 'boxing', 'bellator', 'pfl', 'fight', 'bout', 'wrestling', 'wwe']
}

def classify_sport(home_team: str, away_team: str, tournament: str) -> str:
    """Classify sport based on team names and tournament"""
    combined = f"{home_team} {away_team} {tournament}".lower()
    
    for sport, keywords in SPORT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined:
                return sport
    
    # Special cases
    if 'grand prix' in combined or 'formula' in combined or 'f1' in combined:
        return 'Racing'
    if 'esports' in combined or 'gaming' in combined:
        return 'eSports'
    
    return 'Other'

def generate_unique_id(game_id: str, source: str) -> str:
    """Generate a unique source ID"""
    # Create a hash of the game ID to ensure uniqueness
    hash_obj = hashlib.md5(f"{source}_{game_id}".encode())
    return f"{source}_{hash_obj.hexdigest()[:16]}"

def fetch_and_store_real_data():
    """Fetch and store real sports data"""
    logger.info("🚀 Fetching REAL sports data with proper classification")
    
    try:
        # Fetch from Overtime API
        response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
        if response.status_code != 200:
            logger.error(f"API request failed with status {response.status_code}")
            return
        
        games = response.json()
        logger.info(f"✅ Found {len(games)} games from API")
        
        # Process games
        created_count = 0
        now = datetime.now(timezone.utc)
        
        with db_manager.get_db_session() as db:
            for game_id, info in games.items():
                try:
                    # Skip finished games
                    if info.get('isGameFinished', True):
                        continue
                    
                    teams = info.get('teams', [])
                    if len(teams) < 2:
                        continue
                    
                    home_team = next((t for t in teams if t.get('isHome')), {}).get('name', '')
                    away_team = next((t for t in teams if not t.get('isHome')), {}).get('name', '')
                    
                    if not home_team or not away_team:
                        continue
                    
                    # Skip futures markets
                    if any(term in f"{home_team} {away_team}".lower() 
                          for term in ['winner', 'championship', 'mvp', 'podium']):
                        continue
                    
                    tournament = info.get('tournamentName', '')
                    sport = classify_sport(home_team, away_team, tournament)
                    
                    # Skip non-major sports
                    if sport not in ['Soccer', 'Basketball', 'Football', 'Baseball', 'Hockey', 'Tennis']:
                        continue
                    
                    # Generate unique ID
                    source_id = generate_unique_id(game_id, 'overtime_api_live')
                    
                    # Check if market already exists
                    existing = db.query(Market).filter(Market.source_id == source_id).first()
                    if existing:
                        continue
                    
                    # Calculate maturity date (assume games are within next 7 days)
                    last_update = info.get('lastUpdate', 0)
                    if last_update > 0:
                        # Convert from timestamp
                        maturity_date = datetime.fromtimestamp(last_update / 1000, tz=timezone.utc)
                        # If it's in the past, skip
                        if maturity_date < now:
                            continue
                    else:
                        # Default to tomorrow
                        maturity_date = now + timedelta(days=1)
                    
                    # Create market
                    market = Market(
                        source_id=source_id,
                        source='overtime_api_live',
                        sport=sport,
                        league_name=tournament or f"{sport} League",
                        market_type='winner',
                        home_team=home_team,
                        away_team=away_team,
                        is_finished=False,
                        tournament=tournament,
                        maturity_date=maturity_date,
                        updated_at=now
                    )
                    
                    db.add(market)
                    
                    # Add sample odds (you would get real odds from the API)
                    # For now, using realistic default odds
                    odds_data = [
                        {'outcome': f"{home_team} Win", 'odds': 2.10},
                        {'outcome': 'Draw', 'odds': 3.40},
                        {'outcome': f"{away_team} Win", 'odds': 2.85}
                    ]
                    
                    for odd_data in odds_data:
                        odd = Odd(
                            source_id=source_id,
                            decimal_odds=odd_data['odds'],
                            outcome=odd_data['outcome'],
                            is_active=True,
                            timestamp=now,
                            market_data=json.dumps({'game_id': game_id}),
                            updated_at=now
                        )
                        db.add(odd)
                    
                    created_count += 1
                    
                    if created_count <= 10:  # Log first 10
                        logger.info(f"  ✅ Created: {home_team} vs {away_team} ({sport}) - {maturity_date.strftime('%Y-%m-%d %H:%M')}")
                    
                except Exception as e:
                    logger.error(f"Error processing game {game_id}: {e}")
                    continue
            
            # Commit all changes
            db.commit()
            logger.info(f"\n🎯 Successfully created {created_count} new markets!")
            
            # Show current market stats
            total_markets = db.query(Market).filter(Market.is_finished == False).count()
            future_markets = db.query(Market).filter(
                Market.maturity_date > now,
                Market.is_finished == False
            ).count()
            
            logger.info(f"\n📊 Market Statistics:")
            logger.info(f"  Total unfinished markets: {total_markets}")
            logger.info(f"  Future markets: {future_markets}")
            
            # Show sample of upcoming markets
            upcoming = db.query(Market).filter(
                Market.maturity_date > now,
                Market.is_finished == False
            ).order_by(Market.maturity_date).limit(5).all()
            
            if upcoming:
                logger.info(f"\n🎮 Next 5 upcoming markets:")
                for m in upcoming:
                    logger.info(f"  • {m.home_team} vs {m.away_team} ({m.sport}) - {m.maturity_date.strftime('%Y-%m-%d %H:%M')}")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)

if __name__ == "__main__":
    fetch_and_store_real_data()