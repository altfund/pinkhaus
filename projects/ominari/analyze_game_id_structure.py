#!/usr/bin/env python3
"""
Analyze game ID structure to find sport indicators
"""

import requests
import json

# Get sports mapping
sports_response = requests.get('https://api.overtime.io/overtime-v2/sports', timeout=10)
sports = sports_response.json()

# Get games
games_response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=10)
games_data = games_response.json()

print('🔍 ANALYZING GAME ID STRUCTURES')
print('=' * 60)

# Group games by ID patterns
id_patterns = {}
sample_games_by_pattern = {}

for game_id, info in games_data.items():
    if not info.get('isGameFinished', True):  # Active games only
        teams = info.get('teams', [])
        if len(teams) >= 2:
            # Check different ID patterns
            if game_id.startswith('0x3'):
                pattern = 'hex_3'
            elif game_id.startswith('0x'):
                pattern = 'hex_other'
            elif '_' in game_id:
                # Extract prefix before underscore
                prefix = game_id.split('_')[0]
                pattern = f'prefix_{prefix}'
            else:
                pattern = 'other'
            
            if pattern not in id_patterns:
                id_patterns[pattern] = 0
                sample_games_by_pattern[pattern] = []
            
            id_patterns[pattern] += 1
            
            if len(sample_games_by_pattern[pattern]) < 5:
                sample_games_by_pattern[pattern].append({
                    'id': game_id,
                    'home': teams[0].get('name', '?'),
                    'away': teams[1].get('name', '?'),
                    'tournament': info.get('tournamentName', '')
                })

# Show patterns
print('\n📊 GAME ID PATTERNS:')
for pattern, count in sorted(id_patterns.items(), key=lambda x: x[1], reverse=True):
    print(f'\n{pattern}: {count} games')
    
    # Show samples
    for sample in sample_games_by_pattern[pattern][:3]:
        print(f'  ID: {sample["id"][:20]}...')
        print(f'  Game: {sample["home"]} vs {sample["away"]}')