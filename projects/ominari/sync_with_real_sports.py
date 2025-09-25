#!/usr/bin/env python3
"""
Sync Overtime API data with proper sport classification
Uses comprehensive sport detection based on team names and leagues
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sport patterns from our analysis of real data
SPORT_PATTERNS = {
    'Soccer': [
        # European soccer indicators
        'FC', ' City', 'United', 'Real ', 'Club ', 'Atletico', 'Barcelona', 'Madrid',
        'Chelsea', 'Arsenal', 'Liverpool', 'Milan', 'Inter', 'Juventus', 'Roma',
        'Dortmund', 'Bayern', 'PSG', 'Lyon', 'Marseille', 'Monaco',
        # MLS teams
        'Galaxy', 'LAFC', 'Sounders', 'Timbers', 'Whitecaps', 'Impact',
        # Other leagues
        'Wanderers', 'Victory', 'Phoenix', 'Jets', 'Mariners',
        # League names
        'Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1', 'MLS'
    ],
    'Baseball': [
        'Yankees', 'Dodgers', 'Giants', 'Cubs', 'Red Sox', 'Astros', 'Rangers',
        'Rays', 'Blue Jays', 'Orioles', 'Twins', 'Mariners', 'Royals', 'Athletics',
        'Phillies', 'Braves', 'Marlins', 'Mets', 'Padres', 'Rockies', 'Diamondbacks',
        'Brewers', 'Pirates', 'Reds', 'Nationals', 'Cardinals', 'White Sox', 'Angels',
        'Tigers', 'Guardians', 'Royals',
        # League names
        'MLB', 'AL', 'NL'
    ],
    'Basketball': [
        'Lakers', 'Celtics', 'Warriors', 'Bulls', 'Heat', 'Spurs', 'Nets', 'Knicks',
        'Clippers', 'Suns', 'Mavericks', 'Rockets', 'Thunder', 'Jazz', 'Trail Blazers',
        'Kings', 'Pelicans', 'Grizzlies', 'Hornets', 'Hawks', 'Magic', 'Wizards',
        'Pacers', 'Cavaliers', 'Pistons', 'Raptors', 'Bucks', 'Timberwolves',
        'Nuggets', '76ers',
        # College
        'Wildcats', 'Tar Heels', 'Blue Devils', 'Cardinals', 'Bruins',
        # International
        'Maccabi', 'CSKA', 'Olimpia', 'Baskonia',
        # League names
        'NBA', 'NCAA', 'Euroleague'
    ],
    'Hockey': [
        'Oilers', 'Panthers', 'Lightning', 'Avalanche', 'Bruins', 'Canadiens',
        'Rangers', 'Islanders', 'Devils', 'Flyers', 'Penguins', 'Capitals',
        'Hurricanes', 'Blue Jackets', 'Red Wings', 'Blackhawks', 'Blues',
        'Predators', 'Jets', 'Wild', 'Stars', 'Coyotes', 'Golden Knights',
        'Sharks', 'Kings', 'Ducks', 'Canucks', 'Flames', 'Senators', 'Sabres',
        'Maple Leafs', 'Kraken',
        # International patterns
        'HC ', ' HK', 'Dynamo', 'SKA', 'CSKA',
        # League names
        'NHL', 'KHL', 'SHL', 'Liiga'
    ],
    'Football': [
        'Raiders', 'Chargers', 'Chiefs', 'Broncos', 'Cowboys', 'Eagles', '49ers',
        'Seahawks', 'Cardinals', 'Rams', 'Packers', 'Bears', 'Lions', 'Vikings',
        'Buccaneers', 'Saints', 'Falcons', 'Panthers', 'Patriots', 'Bills', 'Dolphins',
        'Jets', 'Steelers', 'Ravens', 'Browns', 'Bengals', 'Titans', 'Jaguars',
        'Colts', 'Texans', 'Commanders', 'Giants',
        # College
        'Crimson Tide', 'Buckeyes', 'Wolverines', 'Fighting Irish', 'Trojans',
        'Longhorns', 'Sooners', 'Tigers', 'Bulldogs', 'Gators',
        # League names
        'NFL', 'NCAA Football', 'CFB'
    ],
    'AFL': [
        'Bulldogs', 'SUNS', 'Swans', 'Lions', 'Cats', 'Eagles', 'Crows',
        'Hawthorn', 'Melbourne', 'Richmond', 'Essendon', 'Collingwood',
        'Geelong', 'Port Adelaide', 'Fremantle', 'Carlton', 'St Kilda',
        'North Melbourne', 'West Coast', 'Brisbane', 'Adelaide', 'GWS',
        # League name
        'AFL', 'AFLW'
    ],
    'MMA': [
        'UFC', 'PFL', 'Bellator', 'ONE Championship', 'Fight Night',
        # Fighter patterns
        ' vs ', 'Lightweight', 'Welterweight', 'Middleweight', 'Heavyweight'
    ],
    'Boxing': [
        'WBC', 'WBA', 'IBF', 'WBO', 'Title Fight', 'Championship',
        # Weight classes
        'Heavyweight', 'Cruiserweight', 'Light Heavyweight', 'Middleweight'
    ],
    'Tennis': [
        'Open', 'Masters', 'Grand Slam', 'Wimbledon', 'US Open', 'French Open',
        'Australian Open', 'ATP', 'WTA', 'Davis Cup',
        # Player name patterns (common tennis surnames)
        'Djokovic', 'Nadal', 'Federer', 'Murray', 'Sinner', 'Alcaraz'
    ],
    'Golf': [
        'Open Championship', 'Masters', 'PGA', 'US Open', 'British Open',
        'Ryder Cup', 'FedEx Cup', 'Round Leader', 'Tournament Winner'
    ],
    'Racing': [
        'Grand Prix', 'NASCAR', 'Formula', 'F1', 'IndyCar', 'Speedway',
        'Circuit', 'Rally', 'Le Mans', '500'
    ],
    'Esports': [
        'Gaming', 'Esports', 'G2 ', 'Fnatic', 'Secret', 'Virtus', 'Academy',
        'Cloud9', 'TSM', 'FaZe', 'Liquid', 'NaVi', 'Astralis',
        # Game titles
        'League of Legends', 'CS:GO', 'CSGO', 'Dota', 'Valorant', 'LoL'
    ]
}

def detect_sport(home_team, away_team, tournament_name=''):
    """
    Detect sport based on team names and tournament name
    """
    # Combine all text for searching
    search_text = f"{home_team} {away_team} {tournament_name}".upper()
    
    # Check each sport's patterns
    sport_scores = {}
    
    for sport, patterns in SPORT_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if pattern.upper() in search_text:
                score += 1
                # League names are worth more
                if pattern in ['NFL', 'NBA', 'MLB', 'NHL', 'AFL', 'UFC', 'ATP', 'PGA']:
                    score += 2
        
        if score > 0:
            sport_scores[sport] = score
    
    # Return sport with highest score
    if sport_scores:
        return max(sport_scores, key=sport_scores.get)
    
    return 'Unknown'

def sync_overtime_markets():
    """Fetch and sync Overtime markets with proper sport classification"""
    logger.info("🎯 Syncing Overtime markets with real sport classification...")
    
    try:
        # Fetch from Overtime API
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
                    home_team = teams[0]
                    away_team = teams[1]
                    tournament = info.get('tournamentName', '')
                    
                    # Detect sport using our comprehensive patterns
                    sport = detect_sport(home_team, away_team, tournament)
                    
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
        markets_added = 0
        markets_updated = 0
        
        with db_manager.get_db_session() as db:
            for game in active_games:
                market_id = f"api_live_real_{game['gameId']}"
                
                # Check if exists
                existing = db.query(Market).filter(Market.source_id == market_id).first()
                
                if existing:
                    # Update sport classification if it's different or unknown
                    if existing.sport != game['sport'] and (existing.sport == 'Unknown' or game['sport'] != 'Unknown'):
                        logger.info(f"🔄 Updating sport: {existing.home_team} vs {existing.away_team}")
                        logger.info(f"   {existing.sport} → {game['sport']}")
                        existing.sport = game['sport']
                        existing.updated_at = datetime.now(timezone.utc)
                        markets_updated += 1
                else:
                    # Create new market
                    maturity = datetime.fromtimestamp(game['lastUpdate'] / 1000, tz=timezone.utc)
                    
                    market = Market(
                        source_id=market_id,
                        source="api_live_real",
                        sport=game['sport'],
                        league_name=game['tournamentName'] or 'Overtime',
                        market_type="winner",
                        home_team=game['homeTeam'],
                        away_team=game['awayTeam'],
                        maturity_date=maturity,
                        is_finished=False,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(market)
                    markets_added += 1
                    
                    # Add default odds
                    for outcome, odds_val in [('home', 2.0), ('away', 2.0)]:
                        american = 100 if odds_val >= 2.0 else int(-100 / (odds_val - 1))
                        
                        odd = Odd(
                            source_id=market_id,
                            market_type="winner",
                            outcome=outcome,
                            source="api_live",
                            bookmaker="Overtime",
                            decimal_odds=odds_val,
                            american_odds=american,
                            normalized_implied=1.0 / odds_val,
                            updated_at=datetime.now(timezone.utc)
                        )
                        db.add(odd)
            
            db.commit()
        
        logger.info(f"\n✨ Sync complete!")
        logger.info(f"  Markets added: {markets_added}")
        logger.info(f"  Sports updated: {markets_updated}")
        
        # Show final database stats
        with db_manager.get_db_session() as db:
            sport_counts = db.query(Market.sport, db.func.count(Market.id))\
                .filter(Market.source == 'api_live_real')\
                .group_by(Market.sport)\
                .order_by(db.func.count(Market.id).desc())\
                .all()
            
            logger.info("\n📊 Final database sport distribution:")
            for sport, count in sport_counts:
                logger.info(f"  {sport}: {count} markets")
        
        return True
        
    except Exception as e:
        logger.error(f"Error syncing markets: {e}")
        return False

if __name__ == "__main__":
    sync_overtime_markets()