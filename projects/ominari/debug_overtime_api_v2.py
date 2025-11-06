#!/usr/bin/env python3
"""Debug Overtime API response"""

import requests
import json
from datetime import datetime, timezone

print("Fetching from Overtime API...")
response = requests.get(
    "https://api.overtime.io/overtime-v2/games-info", 
    headers={'accept': 'application/json'}
)

print(f"Status Code: {response.status_code}")
print(f"Response Headers: {dict(response.headers)}")

if response.status_code == 200:
    try:
        data = response.json()
        print(f"\nResponse keys: {list(data.keys())}")
        
        games = data.get('games', [])
        print(f"Total games: {len(games)}")
        
        if games:
            # Show first game structure
            print(f"\nFirst game structure:")
            print(json.dumps(games[0], indent=2))
            
            # Count by sport
            sports = {}
            for game in games:
                sport = game.get('sport', 'Unknown')
                sports[sport] = sports.get(sport, 0) + 1
            
            print(f"\nGames by sport:")
            for sport, count in sports.items():
                print(f"  {sport}: {count}")
                
            # Check dates
            now = datetime.now(timezone.utc).timestamp() * 1000
            upcoming = sum(1 for g in games if g.get('maturity', 0) > now)
            print(f"\nUpcoming games: {upcoming}")
            print(f"Past games: {len(games) - upcoming}")
        else:
            print("\nNo games in response")
            print(f"Full response: {json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Response text: {response.text[:500]}")
else:
    print(f"Error response: {response.text[:500]}")