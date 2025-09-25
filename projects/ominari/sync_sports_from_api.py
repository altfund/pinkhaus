#!/usr/bin/env python3
"""
Properly sync sport categories using Overtime API sport definitions
Maps game IDs to sport IDs to get accurate sport categorization
"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

import psycopg2
import requests
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_sport_mappings():
    """Get sport ID to sport name mappings from API"""
    try:
        response = requests.get('https://api.overtime.io/overtime-v2/sports', timeout=10)
        sports_data = response.json()
        
        # Create mapping of league names to sports
        league_to_sport = {}
        sport_id_to_name = {}
        
        for sport_id, info in sports_data.items():
            sport_name = info.get('sport', 'Unknown')
            label = info.get('label', '')
            
            sport_id_to_name[sport_id] = sport_name
            
            # Map leagues to sports
            if label:
                league_to_sport[label.lower()] = sport_name
                # Add variations
                if 'NCAA' in label:
                    if 'Basketball' in label:
                        league_to_sport['ncaab'] = sport_name
                        league_to_sport['college basketball'] = sport_name
                    elif 'Football' in label:
                        league_to_sport['ncaaf'] = sport_name
                        league_to_sport['cfb'] = sport_name
                        league_to_sport['college football'] = sport_name
        
        return league_to_sport, sport_id_to_name
        
    except Exception as e:
        logger.error(f"Failed to get sport mappings: {e}")
        return {}, {}


def get_markets_from_api():
    """Get current markets from API with sport info"""
    try:
        # First get sport mappings
        league_to_sport, sport_id_to_name = get_sport_mappings()
        logger.info(f"Loaded {len(league_to_sport)} league mappings")
        
        # Get games
        response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
        games = response.json()
        
        markets_with_sport = []
        
        for game_id, info in games.items():
            teams = info.get('teams', [])
            if len(teams) != 2:
                continue
                
            home = next((t for t in teams if t.get('isHome')), {}).get('name', '')
            away = next((t for t in teams if not t.get('isHome')), {}).get('name', '')
            tournament = info.get('tournamentName', '').strip()
            
            if not home or not away:
                continue
                
            # Try to determine sport from tournament/league name
            sport = 'Unknown'
            tournament_lower = tournament.lower()
            
            # Direct league mapping
            for league_key, sport_name in league_to_sport.items():
                if league_key in tournament_lower:
                    sport = sport_name
                    break
            
            # If still unknown, check for sport indicators in tournament name
            if sport == 'Unknown':
                if any(term in tournament_lower for term in ['nfl', 'football', 'ncaaf', 'cfb']):
                    sport = 'Football'
                elif any(term in tournament_lower for term in ['nba', 'basketball', 'ncaab', 'wnba']):
                    sport = 'Basketball'
                elif any(term in tournament_lower for term in ['mlb', 'baseball']):
                    sport = 'Baseball'
                elif any(term in tournament_lower for term in ['nhl', 'hockey', 'khl']):
                    sport = 'Hockey'
                elif any(term in tournament_lower for term in ['soccer', 'mls', 'premier', 'liga', 'bundesliga', 'serie a', 'champions league']):
                    sport = 'Soccer'
                elif any(term in tournament_lower for term in ['ufc', 'mma', 'boxing', 'bellator', 'pfl', 'fight']):
                    sport = 'Fighting'
                elif any(term in tournament_lower for term in ['tennis', 'atp', 'wta']):
                    sport = 'Tennis'
                elif any(term in tournament_lower for term in ['golf', 'pga']):
                    sport = 'Golf'
                elif any(term in tournament_lower for term in ['esports', 'league of legends', 'lol', 'dota', 'valorant', 'cs:']):
                    sport = 'eSports'
                elif any(term in tournament_lower for term in ['cricket', 'ipl', 't20']):
                    sport = 'Cricket'
                elif any(term in tournament_lower for term in ['handball']):
                    sport = 'Handball'
                elif any(term in tournament_lower for term in ['table tennis', 'ping pong']):
                    sport = 'TableTennis'
            
            markets_with_sport.append({
                'game_id': game_id,
                'home': home,
                'away': away,
                'tournament': tournament,
                'sport': sport
            })
        
        return markets_with_sport
        
    except Exception as e:
        logger.error(f"Failed to get markets: {e}")
        return []


def update_sports_in_db():
    """Update sports in database based on API data"""
    conn = psycopg2.connect(
        host='localhost',
        port=5999,
        database='ominari_production',
        user='ominari_user',
        password='ominari_2025_secure'
    )
    cur = conn.cursor()
    
    # Get API markets with sports
    api_markets = get_markets_from_api()
    logger.info(f"Got {len(api_markets)} markets from API")
    
    # Create lookup by teams
    team_to_sport = {}
    for market in api_markets:
        key1 = f"{market['home'].lower()}|{market['away'].lower()}"
        key2 = f"{market['away'].lower()}|{market['home'].lower()}"
        team_to_sport[key1] = market['sport']
        team_to_sport[key2] = market['sport']
    
    # Update database markets
    cur.execute("""
        SELECT source_id, home_team, away_team, sport
        FROM market
        WHERE maturity_date > NOW()
    """)
    
    db_markets = cur.fetchall()
    updates = {}
    
    for source_id, home, away, current_sport in db_markets:
        key = f"{home.lower()}|{away.lower()}"
        if key in team_to_sport:
            api_sport = team_to_sport[key]
            if api_sport != current_sport and api_sport != 'Unknown':
                cur.execute("""
                    UPDATE market
                    SET sport = %s
                    WHERE source_id = %s
                """, (api_sport, source_id))
                
                updates[api_sport] = updates.get(api_sport, 0) + 1
                
                if updates[api_sport] <= 3:  # Show first 3 examples
                    logger.info(f"  {current_sport} → {api_sport}: {home} vs {away}")
    
    conn.commit()
    
    # Summary
    logger.info("\n=== Sport Updates from API ===")
    total = sum(updates.values())
    for sport, count in sorted(updates.items()):
        logger.info(f"  Updated to {sport}: {count}")
    logger.info(f"Total updated: {total}")
    
    # Show new distribution
    cur.execute("""
        SELECT sport, COUNT(*) as count
        FROM market
        WHERE maturity_date > NOW()
        GROUP BY sport
        ORDER BY count DESC
    """)
    
    logger.info("\n=== Updated Sport Distribution ===")
    for row in cur.fetchall():
        logger.info(f"  {row[0]}: {row[1]}")
    
    cur.close()
    conn.close()


if __name__ == "__main__":
    logger.info("🏆 Syncing Sports from Overtime API")
    logger.info("=" * 50)
    update_sports_in_db()