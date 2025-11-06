#!/usr/bin/env python3
"""Parse Overtime API games properly"""

import requests
import json
from datetime import datetime, timezone

print("Fetching from Overtime API...")
response = requests.get(
    "https://api.overtime.io/overtime-v2/games-info", 
    headers={'accept': 'application/json'}
)

if response.status_code == 200:
    data = response.json()
    print(f"Total games in response: {len(data)}")
    
    # Convert the dict format to a list of games
    games = []
    now = datetime.now(timezone.utc).timestamp() * 1000  # Current time in ms
    
    for game_id, game_data in data.items():
        # Skip if not a dict (some entries might be different)
        if not isinstance(game_data, dict):
            continue
            
        # Extract game info
        game = {
            'gameId': game_id,
            'lastUpdate': game_data.get('lastUpdate', 0),
            'isFinished': game_data.get('isGameFinished', False),
            'status': game_data.get('gameStatus', ''),
            'tournament': game_data.get('tournamentName', ''),
            'positionNames': game_data.get('positionNames', []),
            'teams': game_data.get('teams', [])
        }
        
        # Try to determine if it's a future game
        if game['lastUpdate'] > now and not game['isFinished']:
            games.append(game)
    
    print(f"\nFound {len(games)} future games")
    
    # Group by sport/type based on team names
    sports = {}
    for game in games[:100]:  # Check first 100
        teams = game.get('teams', [])
        if teams and len(teams) >= 2:
            # Try to categorize based on team names
            team1 = teams[0].get('name', '')
            team2 = teams[1].get('name', '')
            
            # Simple categorization
            if any(x in team1.lower() or x in team2.lower() for x in ['fc', 'united', 'city', 'real', 'barcelona']):
                sport = 'Soccer'
            elif any(x in team1.lower() or x in team2.lower() for x in ['nba', 'lakers', 'celtics', 'heat']):
                sport = 'Basketball'
            elif any(x in team1.lower() or x in team2.lower() for x in ['nfl', 'cowboys', 'patriots']):
                sport = 'Football'
            elif 'mlb' in team1.lower() or 'world series' in team2.lower():
                sport = 'Baseball'
            else:
                sport = 'Other'
                
            sports[sport] = sports.get(sport, 0) + 1
            
    print(f"\nGames by sport (first 100):")
    for sport, count in sports.items():
        print(f"  {sport}: {count}")
    
    # Show some example games
    print(f"\nExample games:")
    for i, game in enumerate(games[:5]):
        teams = game.get('teams', [])
        if len(teams) >= 2:
            print(f"\n{i+1}. {teams[0].get('name', '?')} vs {teams[1].get('name', '?')}")
            print(f"   Game ID: {game['gameId']}")
            print(f"   Status: {game['status']}")
            print(f"   Tournament: {game['tournament']}")
            print(f"   Last Update: {datetime.fromtimestamp(game['lastUpdate']/1000, tz=timezone.utc)}")
            if game.get('positionNames'):
                print(f"   Positions: {len(game['positionNames'])} options")
    
    # Look for games with proper structure for betting
    betting_games = []
    for game_id, game_data in data.items():
        if isinstance(game_data, dict):
            teams = game_data.get('teams', [])
            # Look for games with exactly 2 teams (home vs away)
            if len(teams) == 2 and all('name' in t for t in teams):
                # Check if it has odds data
                if 'positionNames' in game_data and len(game_data['positionNames']) in [2, 3]:
                    betting_games.append({
                        'gameId': game_id,
                        'homeTeam': teams[0]['name'],
                        'awayTeam': teams[1]['name'],
                        'lastUpdate': game_data.get('lastUpdate', 0),
                        'isFinished': game_data.get('isGameFinished', False),
                        'positionCount': len(game_data['positionNames'])
                    })
    
    print(f"\n\nFound {len(betting_games)} games suitable for betting")
    
    # Show betting games
    upcoming_betting = [g for g in betting_games if g['lastUpdate'] > now and not g['isFinished']]
    print(f"Upcoming betting games: {len(upcoming_betting)}")
    
    for i, game in enumerate(upcoming_betting[:5]):
        print(f"\n{i+1}. {game['homeTeam']} vs {game['awayTeam']}")
        print(f"   Game ID: {game['gameId']}")
        print(f"   Positions: {game['positionCount']} (likely {'Home/Draw/Away' if game['positionCount'] == 3 else 'Home/Away'})")
        print(f"   Time: {datetime.fromtimestamp(game['lastUpdate']/1000, tz=timezone.utc)}")