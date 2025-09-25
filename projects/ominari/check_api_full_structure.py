#!/usr/bin/env python3
"""
Check the full structure of Overtime API to find any sport indicators
"""

import requests
import json

# Get games from API
response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=10)
games = response.json()

print('🔍 CHECKING FULL API STRUCTURE')
print('=' * 60)

# Find active games with different patterns
samples_by_pattern = {}

for game_id, info in games.items():
    if not info.get('isGameFinished', True):  # Active games
        teams = info.get('teams', [])
        if len(teams) >= 2:
            # Categorize by team patterns
            team_text = f"{teams[0]} vs {teams[1]}"
            
            category = 'Unknown'
            if 'FC' in team_text or 'United' in team_text or 'City' in team_text:
                category = 'Soccer-like'
            elif any(x in team_text for x in ['Yankees', 'Dodgers', 'Cubs', 'Sox']):
                category = 'Baseball-like'
            elif any(x in team_text for x in ['Lakers', 'Celtics', 'Warriors', 'Heat']):
                category = 'Basketball-like'
            elif any(x in team_text for x in ['Oilers', 'Panthers', 'Lightning', 'Bruins']):
                category = 'Hockey-like'
            elif ' vs ' in team_text and 'Esports' in team_text:
                category = 'Esports-like'
            
            if category not in samples_by_pattern or len(samples_by_pattern[category]) < 2:
                if category not in samples_by_pattern:
                    samples_by_pattern[category] = []
                samples_by_pattern[category].append((game_id, info, teams))

# Show samples from each category
for category, samples in samples_by_pattern.items():
    print(f'\n🏆 {category} Games:')
    print('-' * 60)
    
    for game_id, info, teams in samples[:2]:  # Show 2 from each
        print(f'\nGame ID: {game_id}')
        print(f'Teams: {teams[0]} vs {teams[1]}')
        print('All fields:')
        
        # Show all fields except positionNames (too long)
        for key, value in sorted(info.items()):
            if key != 'positionNames':
                print(f'  {key}: {value}')
        
        # Check for any hidden sport fields
        if 'sport' in str(info).lower() and 'sport' not in [k.lower() for k in info.keys()]:
            print('  ⚠️  Found "sport" in nested data!')

print('\n\n📊 FIELD SUMMARY:')
print('Common fields across all games:')

# Get common fields
if games:
    sample_game = next(iter(games.values()))
    for field in sorted(sample_game.keys()):
        print(f'  - {field}')

# Check for alternative endpoints
print('\n\n🌐 CHECKING ALTERNATIVE ENDPOINTS:')
alternative_endpoints = [
    'https://api.overtime.io/overtime-v2/sports',
    'https://api.overtime.io/overtime-v2/sport-list',
    'https://api.overtime.io/sports',
    'https://api.overtime.io/markets',
]

for endpoint in alternative_endpoints:
    try:
        r = requests.get(endpoint, timeout=3)
        if r.status_code == 200:
            print(f'✅ {endpoint} - Status: {r.status_code}')
            try:
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    print(f'   Type: list with {len(data)} items')
                    print(f'   Sample: {data[0] if len(str(data[0])) < 200 else "Too large to display"}')
                elif isinstance(data, dict):
                    print(f'   Type: dict with keys: {list(data.keys())[:10]}')
            except:
                print(f'   Response not JSON')
        else:
            print(f'❌ {endpoint} - Status: {r.status_code}')
    except:
        print(f'❌ {endpoint} - Failed to connect')