#!/usr/bin/env python3
"""
Improved sync script with comprehensive sport classification
Based on Overtime API sport definitions and extensive pattern matching
"""

import sqlite3
import requests
import json
from datetime import datetime

DB_PATH = "sport_odds.db"

# Comprehensive sport detection patterns based on our analysis
SPORT_PATTERNS = {
    'Soccer': {
        'teams': ['FC', 'City', 'United', 'Real', 'Club', 'Athletic', 'Sporting', 'FK', 'SC', 'CF', 'IL', 'IF', 'BV', 'SA'],
        'leagues': ['MLS', 'EPL', 'Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1', 'UEFA', 'Champions League', 'Europa', 'Copa', 'Eredivisie'],
    },
    'Baseball': {
        'teams': ['Yankees', 'Dodgers', 'Giants', 'Cubs', 'Red Sox', 'Astros', 'Rangers', 'Rays', 'Blue Jays', 'Orioles', 'Twins', 'Mariners', 'Royals', 'Athletics', 'Phillies', 'Braves', 'Marlins', 'Mets', 'Padres', 'Rockies', 'Diamondbacks', 'Brewers', 'Pirates', 'Reds', 'Nationals', 'Cardinals', 'White Sox', 'Angels', 'Tigers', 'Guardians', 'Fighters', 'Marines', 'Lions', 'Hawks', 'Swallows', 'Carp', 'Dragons', 'BayStars', 'Buffaloes', 'Eagles'],
        'leagues': ['MLB', 'NPB', 'KBO', 'CPBL', 'Baseball'],
    },
    'Basketball': {
        'teams': ['Lakers', 'Celtics', 'Warriors', 'Bulls', 'Heat', 'Spurs', 'Nets', 'Knicks', 'Clippers', 'Suns', 'Mavericks', 'Rockets', 'Thunder', 'Jazz', 'Trail Blazers', 'Kings', 'Pelicans', 'Grizzlies', 'Hornets', 'Hawks', 'Magic', 'Wizards', 'Pacers', 'Cavaliers', 'Pistons', 'Raptors', 'Bucks', 'Timberwolves', 'Nuggets', '76ers', 'Sixers'],
        'leagues': ['NBA', 'WNBA', 'NCAA Basketball', 'Euroleague'],
    },
    'Hockey': {
        'teams': ['Oilers', 'Panthers', 'Lightning', 'Avalanche', 'Bruins', 'Canadiens', 'Rangers', 'Islanders', 'Devils', 'Flyers', 'Penguins', 'Capitals', 'Hurricanes', 'Blue Jackets', 'Red Wings', 'Blackhawks', 'Blues', 'Predators', 'Jets', 'Wild', 'Stars', 'Coyotes', 'Golden Knights', 'Sharks', 'Kings', 'Ducks', 'Canucks', 'Flames', 'Senators', 'Sabres', 'Maple Leafs', 'Kraken', 'Traktor', 'Avtomobilist', 'Ilves', 'Kiekko'],
        'leagues': ['NHL', 'KHL', 'SHL', 'IIHF', 'Hockey', 'Ice Hockey'],
        'patterns': ['HC ', ' HC', ' HK'],
    },
    'Football': {
        'teams': ['Raiders', 'Chargers', 'Chiefs', 'Broncos', 'Cowboys', 'Eagles', '49ers', 'Seahawks', 'Cardinals', 'Rams', 'Packers', 'Bears', 'Lions', 'Vikings', 'Buccaneers', 'Saints', 'Falcons', 'Panthers', 'Patriots', 'Bills', 'Dolphins', 'Jets', 'Steelers', 'Ravens', 'Browns', 'Bengals', 'Titans', 'Jaguars', 'Colts', 'Texans', 'Commanders', 'Giants'],
        'leagues': ['NFL', 'NCAA Football', 'CFB', 'CFL'],
    },
    'Fighting': {
        'patterns': ['vs', 'UFC', 'PFL', 'Bellator', 'ONE', 'Fight', 'Boxing', 'MMA'],
        'leagues': ['UFC', 'PFL', 'Bellator', 'ONE Championship', 'Fight Night'],
    },
    'Tennis': {
        'patterns': ['Open', 'Masters', 'Grand Slam', 'Wimbledon', 'Djokovic', 'Nadal', 'Federer'],
        'leagues': ['ATP', 'WTA', 'Tennis'],
    },
    'Golf': {
        'patterns': ['Open Championship', 'Masters', 'PGA', 'Round Leader', 'Tournament Winner'],
        'leagues': ['PGA', 'Golf'],
    },
    'eSports': {
        'teams': ['Gaming', 'Esports', 'G2', 'Fnatic', 'Secret', 'Virtus', 'Academy', 'Cloud9', 'TSM', 'FaZe', 'Liquid', 'NaVi', 'Astralis', 'XI', 'LFO'],
        'leagues': ['ESEA', 'Asia-Pacific League', 'Europe MENA League', 'ESL', 'CS:', 'Dota', 'LOL', 'Valorant'],
    },
    'Table Tennis': {
        'patterns': ['individual player names'],
        'leagues': ['Table Tennis', 'TT Cup', 'TT Liga'],
    },
    'AFL': {
        'teams': ['Bulldogs', 'Swans', 'Lions', 'Cats', 'Crows', 'Hawthorn', 'Richmond', 'Essendon', 'Collingwood', 'Geelong', 'Port Adelaide', 'Fremantle', 'Carlton', 'St Kilda', 'North Melbourne', 'West Coast', 'Adelaide', 'GWS', 'SUNS'],
        'leagues': ['AFL', 'AFLW'],
    }
}

def detect_sport(home_team, away_team, tournament_name=''):
    """Detect sport using comprehensive patterns"""
    
    # Combine text for searching
    search_text = f"{home_team} {away_team} {tournament_name}".upper()
    
    # Special cases first
    if 'SUPER BOWL' in search_text or 'WORLD SERIES' in search_text or 'MVP' in search_text:
        return 'Futures'
    
    # Check for individual player match (Table Tennis pattern)
    if tournament_name in ['N/A', ''] and ' ' in home_team and ' ' in away_team:
        if len(home_team) < 30 and len(away_team) < 30:
            if 'COLLEGE' not in search_text and 'UNIVERSITY' not in search_text:
                if home_team.count(' ') <= 2 and away_team.count(' ') <= 2:
                    return 'Table Tennis'
    
    # Check each sport's patterns
    sport_scores = {}
    
    for sport, patterns in SPORT_PATTERNS.items():
        score = 0
        
        # Check team names
        for team in patterns.get('teams', []):
            if team.upper() in search_text:
                score += 3
        
        # Check league names (higher weight)
        for league in patterns.get('leagues', []):
            if league.upper() in search_text:
                score += 10
        
        # Check general patterns
        for pattern in patterns.get('patterns', []):
            if pattern.upper() in search_text:
                score += 2
        
        if score > 0:
            sport_scores[sport] = score
    
    # College sports special handling
    if 'COLLEGE' in search_text or 'UNIVERSITY' in search_text:
        if any(fb in search_text for fb in ['FOOTBALL', 'BOWL']):
            return 'Football'
        elif 'BASKETBALL' in search_text:
            return 'Basketball'
        elif 'BASEBALL' in search_text:
            return 'Baseball'
        else:
            # Default college sport based on season/context
            return 'Football'
    
    if sport_scores:
        return max(sport_scores, key=sport_scores.get)
    
    return 'Unknown'

def sync_overtime_markets():
    """Fetch and sync markets from Overtime API with improved sport classification"""
    print("🎯 Syncing Overtime markets with improved sport classification...")
    
    try:
        # Fetch from Overtime API
        response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=10)
        games_data = response.json()
        
        print(f"📊 Found {len(games_data)} total games from API")
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        active_games = 0
        markets_added = 0
        markets_updated = 0
        sport_distribution = {}
        
        # Process each game
        for game_id, info in games_data.items():
            if not info.get('isGameFinished', True):  # Active games only
                teams = info.get('teams', [])
                if len(teams) >= 2:
                    active_games += 1
                    
                    home_team = teams[0].get('name', '?')
                    away_team = teams[1].get('name', '?')
                    tournament = info.get('tournamentName', '')
                    last_update = info.get('lastUpdate', 0)
                    
                    # Detect sport
                    sport = detect_sport(home_team, away_team, tournament)
                    sport_distribution[sport] = sport_distribution.get(sport, 0) + 1
                    
                    market_id = f"api_live_real_{game_id}"
                    
                    # Check if market exists
                    cursor.execute("SELECT source_id, sport FROM market WHERE source_id = ?", (market_id,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update if sport changed
                        if existing[1] != sport:
                            cursor.execute(
                                "UPDATE market SET sport = ?, updated_at = ? WHERE source_id = ?",
                                (sport, datetime.now().isoformat(), market_id)
                            )
                            markets_updated += 1
                    else:
                        # Insert new market
                        maturity = datetime.fromtimestamp(last_update / 1000).isoformat()
                        
                        cursor.execute("""
                            INSERT INTO market (source_id, source, sport, league_name, market_type, 
                                              home_team, away_team, maturity_date, is_finished, 
                                              updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            market_id, "api_live_real", sport, tournament or "Overtime",
                            "winner", home_team, away_team, maturity, 0,
                            datetime.now().isoformat()
                        ))
                        
                        # Add default odds
                        for i, outcome in enumerate(['home', 'away']):
                            cursor.execute("""
                                INSERT INTO odd (source_id, position, market_type, outcome, source, bookmaker,
                                               decimal_odds, american_odds, normalized_implied,
                                               updated_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                market_id, i, "winner", outcome, "api_live", "Overtime",
                                2.0, 100, 0.5,
                                datetime.now().isoformat()
                            ))
                        
                        markets_added += 1
        
        conn.commit()
        
        print(f"\n✅ Sync complete!")
        print(f"  Active games found: {active_games}")
        print(f"  Markets added: {markets_added}")
        print(f"  Markets updated: {markets_updated}")
        
        print(f"\n📊 Sport distribution in this sync:")
        for sport, count in sorted(sport_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sport}: {count} games")
        
        # Show database totals
        cursor.execute("""
            SELECT sport, COUNT(*) 
            FROM market 
            WHERE source = 'api_live_real' 
            GROUP BY sport 
            ORDER BY COUNT(*) DESC
        """)
        
        print(f"\n📊 Total database sport distribution:")
        for sport, count in cursor.fetchall():
            print(f"  {sport}: {count} markets")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    sync_overtime_markets()