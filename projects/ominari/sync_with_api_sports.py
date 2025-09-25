#!/usr/bin/env python3
"""
Sync markets with proper sport classification using Overtime API sports definitions
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import json
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sport_mappings():
    """Get sport definitions from Overtime API"""
    try:
        response = requests.get('https://api.overtime.io/overtime-v2/sports', timeout=10)
        return response.json()
    except:
        return {}

def build_sport_detector(sport_mappings):
    """Build comprehensive sport detection based on API definitions and known patterns"""
    
    # Extract patterns from API sport definitions
    sport_patterns = {}
    
    for sport_id, info in sport_mappings.items():
        sport_name = info.get('sport', 'Unknown')
        label = info.get('label', '')
        optic_name = info.get('opticOddsName', '')
        
        if sport_name not in sport_patterns:
            sport_patterns[sport_name] = {
                'leagues': [],
                'teams': [],
                'keywords': []
            }
        
        # Add league names
        if label:
            sport_patterns[sport_name]['leagues'].append(label)
        if optic_name:
            sport_patterns[sport_name]['leagues'].append(optic_name)
    
    # Add known team patterns based on sports
    team_patterns = {
        'Football': [
            'Raiders', 'Chargers', 'Chiefs', 'Broncos', 'Cowboys', 'Eagles', '49ers',
            'Seahawks', 'Cardinals', 'Rams', 'Packers', 'Bears', 'Lions', 'Vikings',
            'Buccaneers', 'Saints', 'Falcons', 'Panthers', 'Patriots', 'Bills',
            'Dolphins', 'Jets', 'Steelers', 'Ravens', 'Browns', 'Bengals', 'Titans',
            'Jaguars', 'Colts', 'Texans', 'Commanders', 'Giants',
            'Crimson Tide', 'Buckeyes', 'Wolverines', 'Fighting Irish', 'Trojans'
        ],
        'Baseball': [
            'Yankees', 'Dodgers', 'Giants', 'Cubs', 'Red Sox', 'Astros', 'Rangers',
            'Rays', 'Blue Jays', 'Orioles', 'Twins', 'Mariners', 'Royals', 'Athletics',
            'Phillies', 'Braves', 'Marlins', 'Mets', 'Padres', 'Rockies', 'Diamondbacks',
            'Brewers', 'Pirates', 'Reds', 'Nationals', 'Cardinals', 'White Sox', 'Angels',
            'Tigers', 'Guardians',
            # Japanese teams
            'Fighters', 'Marines', 'Lions', 'Hawks', 'Swallows', 'Tigers', 'Carp',
            'Dragons', 'BayStars', 'Giants'
        ],
        'Basketball': [
            'Lakers', 'Celtics', 'Warriors', 'Bulls', 'Heat', 'Spurs', 'Nets', 'Knicks',
            'Clippers', 'Suns', 'Mavericks', 'Rockets', 'Thunder', 'Jazz', 'Trail Blazers',
            'Kings', 'Pelicans', 'Grizzlies', 'Hornets', 'Hawks', 'Magic', 'Wizards',
            'Pacers', 'Cavaliers', 'Pistons', 'Raptors', 'Bucks', 'Timberwolves',
            'Nuggets', '76ers', 'Sixers',
            # International
            'Maccabi', 'CSKA', 'Olimpia', 'Baskonia', 'Real Madrid', 'Barcelona'
        ],
        'Hockey': [
            'Oilers', 'Panthers', 'Lightning', 'Avalanche', 'Bruins', 'Canadiens',
            'Rangers', 'Islanders', 'Devils', 'Flyers', 'Penguins', 'Capitals',
            'Hurricanes', 'Blue Jackets', 'Red Wings', 'Blackhawks', 'Blues',
            'Predators', 'Jets', 'Wild', 'Stars', 'Coyotes', 'Golden Knights',
            'Sharks', 'Kings', 'Ducks', 'Canucks', 'Flames', 'Senators', 'Sabres',
            'Maple Leafs', 'Kraken'
        ],
        'Soccer': [
            'FC', 'City', 'United', 'Real', 'Club', 'Atletico', 'Barcelona', 'Madrid',
            'Chelsea', 'Arsenal', 'Liverpool', 'Milan', 'Inter', 'Juventus', 'Roma',
            'Dortmund', 'Bayern', 'PSG', 'Lyon', 'Marseille', 'Monaco',
            'Galaxy', 'LAFC', 'Sounders', 'Timbers', 'Whitecaps', 'Impact',
            'Wanderers', 'Victory', 'Phoenix'
        ],
        'Fighting': [
            'UFC', 'PFL', 'Bellator', 'ONE', 'vs',
            'Lightweight', 'Welterweight', 'Middleweight', 'Heavyweight'
        ],
        'Tennis': [
            'Open', 'Masters', 'Grand Slam', 'Wimbledon', 'ATP', 'WTA',
            'Djokovic', 'Nadal', 'Federer', 'Murray', 'Sinner', 'Alcaraz'
        ],
        'Golf': [
            'Open Championship', 'Masters', 'PGA', 'Round Leader', 'Winner'
        ],
        'eSports': [
            'Gaming', 'Esports', 'G2', 'Fnatic', 'Secret', 'Virtus', 'Academy',
            'Cloud9', 'TSM', 'FaZe', 'Liquid', 'NaVi', 'Astralis'
        ]
    }
    
    # Add team patterns
    for sport, teams in team_patterns.items():
        if sport in sport_patterns:
            sport_patterns[sport]['teams'].extend(teams)
    
    return sport_patterns

def detect_sport(home_team, away_team, tournament_name, sport_patterns):
    """Detect sport using comprehensive patterns"""
    
    # Combine text for searching
    search_text = f"{home_team} {away_team} {tournament_name}".upper()
    
    sport_scores = {}
    
    for sport, patterns in sport_patterns.items():
        score = 0
        
        # Check league names (highest priority)
        for league in patterns.get('leagues', []):
            if league.upper() in search_text:
                score += 10
        
        # Check team patterns
        for team in patterns.get('teams', []):
            if team.upper() in search_text:
                score += 3
        
        # Check keywords
        for keyword in patterns.get('keywords', []):
            if keyword.upper() in search_text:
                score += 1
        
        if score > 0:
            sport_scores[sport] = score
    
    # Special cases
    if 'ESPORTS' in search_text or 'GAMING' in search_text:
        sport_scores['eSports'] = sport_scores.get('eSports', 0) + 20
    
    # If no match found, use tournament patterns
    if not sport_scores:
        if 'Regular Season' in tournament_name:
            # Try to infer from team names
            if any(team in search_text for team in ['YANKEES', 'DODGERS', 'CUBS', 'SOX']):
                return 'Baseball'
            elif any(team in search_text for team in ['LAKERS', 'CELTICS', 'HEAT']):
                return 'Basketball'
        elif 'Preseason' in tournament_name:
            if any(team in search_text for team in ['RAIDERS', 'COWBOYS', 'EAGLES']):
                return 'Football'
    
    if sport_scores:
        return max(sport_scores, key=sport_scores.get)
    
    return 'Unknown'

def sync_overtime_markets():
    """Sync markets with proper sport classification"""
    logger.info("🎯 Syncing Overtime markets with API sport definitions...")
    
    try:
        # Get sport mappings from API
        sport_mappings = get_sport_mappings()
        logger.info(f"📊 Found {len(sport_mappings)} sport definitions from API")
        
        # Build sport detector
        sport_patterns = build_sport_detector(sport_mappings)
        
        # Get games from API
        response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=10)
        games_data = response.json()
        
        logger.info(f"📊 Found {len(games_data)} total games from API")
        
        active_games = []
        sport_distribution = {}
        
        # Process games
        for game_id, info in games_data.items():
            if not info.get('isGameFinished', True):  # Active games only
                teams = info.get('teams', [])
                if len(teams) >= 2:
                    home_team = teams[0].get('name', '?')
                    away_team = teams[1].get('name', '?')
                    tournament = info.get('tournamentName', '')
                    
                    # Detect sport
                    sport = detect_sport(home_team, away_team, tournament, sport_patterns)
                    
                    game_data = {
                        'gameId': game_id,
                        'homeTeam': home_team,
                        'awayTeam': away_team,
                        'tournamentName': tournament,
                        'sport': sport,
                        'lastUpdate': info.get('lastUpdate', 0)
                    }
                    
                    active_games.append(game_data)
                    sport_distribution[sport] = sport_distribution.get(sport, 0) + 1
        
        logger.info(f"✅ Found {len(active_games)} active games")
        
        # Show sport distribution
        logger.info("\n📊 Sport distribution:")
        for sport, count in sorted(sport_distribution.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {sport}: {count} games")
        
        # Sync to database
        markets_updated = 0
        
        with db_manager.get_db_session() as db:
            # Update existing markets
            for game in active_games:
                market_id = f"api_live_real_{game['gameId']}"
                
                existing = db.query(Market).filter(Market.source_id == market_id).first()
                
                if existing and existing.sport != game['sport']:
                    logger.info(f"🔄 Updating: {existing.home_team} vs {existing.away_team}")
                    logger.info(f"   {existing.sport} → {game['sport']}")
                    existing.sport = game['sport']
                    existing.updated_at = datetime.now(timezone.utc)
                    markets_updated += 1
            
            db.commit()
            
            # Show final stats
            sport_counts = db.query(Market.sport, db.func.count(Market.id))\
                .filter(Market.source == 'api_live_real')\
                .group_by(Market.sport)\
                .order_by(db.func.count(Market.id).desc())\
                .all()
            
            logger.info(f"\n✨ Update complete! {markets_updated} markets updated")
            logger.info("\n📊 Final database sport distribution:")
            for sport, count in sport_counts:
                logger.info(f"  {sport}: {count} markets")
        
        return True
        
    except Exception as e:
        logger.error(f"Error syncing markets: {e}")
        return False

if __name__ == "__main__":
    sync_overtime_markets()