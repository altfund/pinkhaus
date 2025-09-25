#!/usr/bin/env python3
"""
Find how to get sport data directly from Overtime API
"""

import requests
import json

# Check different API endpoints
endpoints = [
    'https://api.overtime.io/overtime-v2/games-info',
    'https://api.overtime.io/overtime-v2/markets',
    'https://api.overtime.io/overtime-v2/live-markets',
    'https://api.overtime.io/overtime-v2/games',
    'https://api.overtime.io/overtime-v2/sport-markets',
]

print('🔍 SEARCHING FOR SPORT DATA IN OVERTIME API')
print('=' * 60)

for endpoint in endpoints:
    print(f'\n📌 Checking: {endpoint}')
    try:
        response = requests.get(endpoint, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f'✅ Status: 200 OK')
            
            # Check structure
            if isinstance(data, dict):
                # Check first few items
                count = 0
                for key, value in list(data.items())[:3]:
                    if isinstance(value, dict):
                        print(f'\nSample game {key}:')
                        for field, val in value.items():
                            if 'sport' in field.lower():
                                print(f'  🎯 {field}: {val}')
                        if 'sportId' not in value and 'sport' not in value:
                            print(f'  ❌ No sport field found')
                        count += 1
            elif isinstance(data, list) and len(data) > 0:
                print(f'List with {len(data)} items')
                # Check first item
                if isinstance(data[0], dict):
                    print('\nFirst item fields:')
                    for field in data[0].keys():
                        if 'sport' in field.lower():
                            print(f'  🎯 {field}: {data[0][field]}')
        else:
            print(f'❌ Status: {response.status_code}')
    except Exception as e:
        print(f'❌ Error: {e}')

# Check a specific game with more detail
print('\n\n🎮 DETAILED GAME CHECK:')
response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=10)
games = response.json()

# Find an active game
for game_id, info in games.items():
    if not info.get('isGameFinished', True):
        print(f'\nGame ID: {game_id}')
        print('Full structure:')
        print(json.dumps(info, indent=2))
        break

# Check if games have league/tournament IDs that map to sports
print('\n\n🏆 CHECKING TOURNAMENT PATTERNS:')
tournament_to_games = {}
for game_id, info in games.items():
    if not info.get('isGameFinished', True):
        tournament = info.get('tournamentName', '')
        if tournament:
            if tournament not in tournament_to_games:
                tournament_to_games[tournament] = []
            teams = info.get('teams', [])
            if len(teams) >= 2:
                tournament_to_games[tournament].append(f"{teams[0]['name']} vs {teams[1]['name']}")

print('\nTournaments with example games:')
for tournament, games_list in sorted(tournament_to_games.items())[:10]:
    print(f'\n{tournament}:')
    for game in games_list[:2]:
        print(f'  - {game}')