#!/usr/bin/env python3
"""Debug Overtime API to find real soccer matches"""

import requests
import json

response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
games = response.json()

print(f'Total games in API: {len(games)}')

# Get a few example games to understand structure
count = 0
for game_id, info in games.items():
    if not info.get('isGameFinished', True) and count < 3:
        print(f'\nGame ID: {game_id}')
        print(f'Keys: {list(info.keys())}')
        
        # Show all fields except large nested ones
        for key, value in info.items():
            if key not in ['childMarkets', 'odds']:
                print(f'  {key}: {value}')
        
        count += 1

# Now specifically look for games with teams
print('\n\n=== Looking for games with home/away teams ===')
team_games = []
for game_id, info in games.items():
    teams = info.get('teams', [])
    if len(teams) >= 2 and not info.get('isGameFinished', True):
        home = next((t for t in teams if t.get('isHome')), None)
        away = next((t for t in teams if not t.get('isHome')), None)
        if home and away:
            team_games.append({
                'id': game_id,
                'home': home.get('name'),
                'away': away.get('name'),
                'tournament': info.get('tournamentName', ''),
                'tags': info.get('tags', []),
                'sport': info.get('sport'),
                'sportId': info.get('sportId'),
                'homeOdds': info.get('homeOdds'),
                'awayOdds': info.get('awayOdds'),
                'drawOdds': info.get('drawOdds'),
                'maturityDate': info.get('maturityDate')
            })

print(f'\nFound {len(team_games)} games with home/away teams')

# Group by tournament to find soccer
tournaments = {}
for game in team_games:
    tournament = game['tournament']
    if tournament not in tournaments:
        tournaments[tournament] = []
    tournaments[tournament].append(game)

# Show tournaments that might be soccer
print('\n=== Tournaments with team games ===')
for tournament, games_list in sorted(tournaments.items()):
    if len(games_list) > 0:
        print(f'\n{tournament}: {len(games_list)} games')
        # Show first game as example
        game = games_list[0]
        print(f"  Example: {game['home']} vs {game['away']}")
        print(f"  Tags: {game['tags']}")

# Look for soccer-specific terms
print('\n=== Searching for soccer-specific games ===')
soccer_terms = ['soccer', 'football', 'premier', 'liga', 'serie a', 'bundesliga', 'ligue', 'championship', 'mls', 'champions league', 'europa', 'cup']
soccer_games = []

for game in team_games:
    tournament_lower = game['tournament'].lower()
    tags_str = str(game['tags']).lower()
    
    for term in soccer_terms:
        if term in tournament_lower or term in tags_str:
            soccer_games.append(game)
            break

print(f'\nFound {len(soccer_games)} potential soccer games')
for game in soccer_games[:20]:
    print(f"\n{game['home']} vs {game['away']} ({game['tournament']})")
    if game['homeOdds'] and game['awayOdds']:
        print(f"  Odds: {game['homeOdds']} / {game['drawOdds']} / {game['awayOdds']}")