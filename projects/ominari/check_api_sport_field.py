#!/usr/bin/env python3
"""
Check if Overtime API provides sportId field
"""
import requests
import json

# Get a sample of games to see all fields
response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=10)
games = response.json()

print('🔍 ACTUAL API FIELDS:')
print('=' * 60)

# Look at first few active games
count = 0
sport_ids_found = {}

for game_id, info in games.items():
    if not info.get('isGameFinished', True) and count < 5:  # Active games only
        print(f'\nGame ID: {game_id}')
        print(f'  Teams: {info.get("homeTeam", "?")} vs {info.get("awayTeam", "?")}')
        
        # Check for sportId
        sport_id = info.get('sportId')
        if sport_id:
            print(f'  ✅ sportId: {sport_id}')
            sport_ids_found[sport_id] = sport_ids_found.get(sport_id, 0) + 1
        else:
            print(f'  ❌ sportId: NOT PROVIDED')
            
        # Show other potentially useful fields
        for field in ['sport', 'sportName', 'tournamentName', 'type', 'category']:
            if field in info:
                print(f'  {field}: {info[field]}')
        
        count += 1

print(f'\n📊 SUMMARY:')
print(f'Total games checked: {count}')
print(f'Sport IDs found: {sport_ids_found}')

# Check the sport ID mapping
sport_map = {
    9001: "Football", 
    9002: "Basketball",
    9003: "Baseball",
    9004: "Soccer",
    9005: "Hockey",
    9006: "MMA",
    9007: "Boxing",
    9008: "Tennis"
}

print(f'\n🏆 Sport ID Mapping:')
for sport_id, count in sport_ids_found.items():
    sport_name = sport_map.get(sport_id, f'Unknown ({sport_id})')
    print(f'  {sport_id} = {sport_name}: {count} games')