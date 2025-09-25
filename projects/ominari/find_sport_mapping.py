#!/usr/bin/env python3
"""
Find how sport IDs map to games in the Overtime API
"""

import requests
import json

# Get sports mapping
sports_response = requests.get('https://api.overtime.io/overtime-v2/sports', timeout=10)
sports = sports_response.json()

# Get games
games_response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=10)
games = games_response.json()

print('🏆 SPORT MAPPING FROM API:')
print('=' * 60)

# Create reverse mapping by league name
league_to_sport = {}
for sport_id, sport_info in sports.items():
    if 'opticOddsName' in sport_info:
        league_to_sport[sport_info['opticOddsName']] = {
            'id': sport_id,
            'sport': sport_info['sport'],
            'label': sport_info['label']
        }
    if 'label' in sport_info:
        league_to_sport[sport_info['label']] = {
            'id': sport_id,
            'sport': sport_info['sport'],
            'label': sport_info['label']
        }

print(f'Found {len(league_to_sport)} league mappings')

# Check games for tournament names
print('\n🔍 CHECKING GAME TOURNAMENT NAMES:')
print('-' * 60)

tournament_counts = {}
matched_games = 0
unmatched_tournaments = set()

for game_id, info in games.items():
    if not info.get('isGameFinished', True):  # Active games
        tournament = info.get('tournamentName', '')
        if tournament:
            tournament_counts[tournament] = tournament_counts.get(tournament, 0) + 1
            
            # Check if we can match this tournament
            if tournament in league_to_sport:
                matched_games += 1
            else:
                unmatched_tournaments.add(tournament)

# Show top tournaments
print('\nTop tournaments in active games:')
for tournament, count in sorted(tournament_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
    match = league_to_sport.get(tournament)
    if match:
        print(f'  ✅ {tournament}: {count} games -> {match["sport"]} (ID: {match["id"]})')
    else:
        print(f'  ❌ {tournament}: {count} games -> NO MATCH')

print(f'\n📊 SUMMARY:')
print(f'Total active games: {sum(tournament_counts.values())}')
print(f'Games with matched tournaments: {matched_games}')
print(f'Unique unmatched tournaments: {len(unmatched_tournaments)}')

if unmatched_tournaments:
    print(f'\n❌ Unmatched tournaments (first 10):')
    for tournament in list(unmatched_tournaments)[:10]:
        print(f'  - {tournament}')

# Try to find patterns in game IDs
print('\n🔍 ANALYZING GAME ID PATTERNS:')
sport_patterns = {
    '3_': [],  # MLB
    '4_': [],  # NBA
    '6_': [],  # NHL
    '10_': [], # MLS
}

for game_id, info in games.items():
    if not info.get('isGameFinished', True):
        # Check if game ID starts with sport ID pattern
        for pattern in sport_patterns:
            if game_id.startswith(pattern):
                teams = info.get('teams', [])
                if len(teams) >= 2:
                    sport_patterns[pattern].append(f"{teams[0]['name']} vs {teams[1]['name']}")

# Show pattern matches
for pattern, games_list in sport_patterns.items():
    if games_list:
        sport_id = pattern.rstrip('_')
        sport_info = sports.get(sport_id, {})
        print(f'\nPattern {pattern} ({sport_info.get("sport", "?")} - {sport_info.get("label", "?")}):')