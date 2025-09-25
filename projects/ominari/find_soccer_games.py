#!/usr/bin/env python3
import os
os.environ['PG_PORT'] = '5999'

import requests

response = requests.get("https://api.overtime.io/overtime-v2/games-info", timeout=30)
games = response.json()

# Look for actual soccer games
count = 0
found = []

for game_id, info in games.items():
    teams = info.get("teams", [])
    if len(teams) != 2:
        continue
    home = next((t for t in teams if t.get("isHome")), {}).get("name", "")
    away = next((t for t in teams if not t.get("isHome")), {}).get("name", "")
    tournament = info.get("tournamentName", "")
    
    # Look for soccer indicators
    words = ["fc", "united", "city", "real", "atletico", "barcelona", "soccer", "football", 
             "premier", "liga", "bundesliga", "serie", "champions", "mls", "lafc", "atlanta", 
             "portland", "seattle", "chelsea", "liverpool", "manchester", "arsenal"]
    combined = (home + " " + away + " " + tournament).lower()
    
    # Skip futures/winners
    if any(term in combined for term in ["winner", "mvp", "championship", "to win"]):
        continue
        
    if any(word in combined for word in words):
        found.append({
            'id': game_id,
            'home': home,
            'away': away,
            'tournament': tournament
        })
        count += 1
        if count >= 20:
            break

print(f"Found {len(found)} soccer games:")
for game in found:
    print(f"\nGame ID: {game['id']}")
    print(f"  Home: {game['home']}")
    print(f"  Away: {game['away']}")
    print(f"  Tournament: {game['tournament']}")