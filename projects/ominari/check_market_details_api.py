#!/usr/bin/env python3
"""
Check if we can get market/game details with sport info
"""

import requests
import json

# Get sports mapping first
sports_response = requests.get('https://api.overtime.io/overtime-v2/sports', timeout=10)
sports = sports_response.json()

print('🏆 SPORTS FROM API:')
print('=' * 60)
# Create reverse mapping by league names
league_to_sport = {}
sport_id_to_info = {}

for sport_id, info in sports.items():
    sport_id_to_info[sport_id] = info
    sport_name = info.get('sport')
    label = info.get('label')
    optic_name = info.get('opticOddsName')
    
    if sport_name and label:
        league_to_sport[label.upper()] = {
            'sport_id': sport_id,
            'sport': sport_name,
            'label': label
        }
    if optic_name:
        league_to_sport[optic_name.upper()] = {
            'sport_id': sport_id, 
            'sport': sport_name,
            'label': label
        }

print(f'Found {len(sport_id_to_info)} sports')
print(f'Created {len(league_to_sport)} league mappings')

# Now check games
games_response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=10)
games = games_response.json()

print(f'\n📊 ANALYZING GAMES:')
print('-' * 60)

# Try to match games to sports
matched_count = 0
unmatched_count = 0
sport_distribution = {}

for game_id, info in games.items():
    if not info.get('isGameFinished', True):  # Active games
        teams = info.get('teams', [])
        tournament = info.get('tournamentName', '').strip()
        
        if len(teams) >= 2:
            home = teams[0].get('name', '')
            away = teams[1].get('name', '')
            
            # Try to match tournament name to sport
            sport_found = None
            sport_id = None
            
            # Direct tournament match
            if tournament and tournament.upper() in league_to_sport:
                match = league_to_sport[tournament.upper()]
                sport_found = match['sport']
                sport_id = match['sport_id']
            
            # Try partial matches
            if not sport_found and tournament:
                for league_key, league_info in league_to_sport.items():
                    if tournament.upper() in league_key or league_key in tournament.upper():
                        sport_found = league_info['sport']
                        sport_id = league_info['sport_id']
                        break
            
            # Check for specific sport keywords in tournament
            if not sport_found:
                tournament_upper = tournament.upper()
                if 'MLB' in tournament_upper or 'BASEBALL' in tournament_upper:
                    sport_found = 'Baseball'
                    sport_id = '3'
                elif 'NFL' in tournament_upper or 'FOOTBALL' in tournament_upper and 'SOCCER' not in tournament_upper:
                    sport_found = 'Football'
                    sport_id = '2'
                elif 'NBA' in tournament_upper or 'BASKETBALL' in tournament_upper:
                    sport_found = 'Basketball' 
                    sport_id = '4'
                elif 'NHL' in tournament_upper or 'HOCKEY' in tournament_upper:
                    sport_found = 'Hockey'
                    sport_id = '6'
                elif 'UFC' in tournament_upper or 'PFL' in tournament_upper or 'MMA' in tournament_upper:
                    sport_found = 'Fighting'
                    sport_id = '7'
                elif 'TENNIS' in tournament_upper or 'ATP' in tournament_upper or 'WTA' in tournament_upper:
                    sport_found = 'Tennis'
                    sport_id = '156'
            
            if sport_found:
                matched_count += 1
                sport_distribution[sport_found] = sport_distribution.get(sport_found, 0) + 1
                if matched_count <= 5:
                    print(f'\n✅ MATCHED: {home} vs {away}')
                    print(f'   Tournament: {tournament}')
                    print(f'   Sport: {sport_found} (ID: {sport_id})')
            else:
                unmatched_count += 1
                if unmatched_count <= 5:
                    print(f'\n❌ UNMATCHED: {home} vs {away}')
                    print(f'   Tournament: {tournament}')

print(f'\n\n📊 SUMMARY:')
print(f'Total active games: {matched_count + unmatched_count}')
print(f'Matched to sports: {matched_count}')
print(f'Unmatched: {unmatched_count}')

print(f'\n📊 Sport distribution:')
for sport, count in sorted(sport_distribution.items(), key=lambda x: x[1], reverse=True):
    print(f'  {sport}: {count} games')