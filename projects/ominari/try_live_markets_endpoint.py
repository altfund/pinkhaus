#!/usr/bin/env python3
"""Try to find live/upcoming markets from Overtime"""

import requests
import json
from datetime import datetime, timezone

# Try different possible endpoints
endpoints = [
    "https://api.overtime.io/overtime-v2/live-markets",
    "https://api.overtime.io/overtime-v2/markets",
    "https://api.overtime.io/overtime-v2/sports-markets",
    "https://api.overtime.io/overtime-markets",
    "https://api.overtime.io/live-markets",
    "https://api.overtime.io/api/markets"
]

print("Trying different Overtime API endpoints...")

for endpoint in endpoints:
    print(f"\n{'='*60}")
    print(f"Trying: {endpoint}")
    
    try:
        response = requests.get(
            endpoint, 
            headers={'accept': 'application/json'},
            timeout=5
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check response type
            if isinstance(data, dict):
                print(f"Response type: dict with {len(data)} keys")
                # Show first few keys
                keys = list(data.keys())[:5]
                print(f"First keys: {keys}")
                
                # Check for standard API response structure
                if 'data' in data or 'markets' in data or 'games' in data:
                    print("✅ Looks like a structured API response!")
                    
            elif isinstance(data, list):
                print(f"Response type: list with {len(data)} items")
                if data and isinstance(data[0], dict):
                    print(f"First item keys: {list(data[0].keys())}")
                    
            # Save successful responses
            if response.status_code == 200 and (isinstance(data, list) or 'markets' in str(data) or 'games' in str(data)):
                filename = endpoint.split('/')[-1].replace('/', '_') + '_response.json'
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"📁 Response saved to {filename}")
                
        elif response.status_code == 404:
            print("❌ Endpoint not found")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Failed: {e}")

# Also try the specific filter approach on games-info
print(f"\n{'='*60}")
print("Trying games-info with specific filtering...")

response = requests.get(
    "https://api.overtime.io/overtime-v2/games-info",
    headers={'accept': 'application/json'}
)

if response.status_code == 200:
    data = response.json()
    
    # Look for soccer games specifically
    soccer_games = []
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    
    for game_id, game_data in data.items():
        if isinstance(game_data, dict):
            teams = game_data.get('teams', [])
            tournament = game_data.get('tournamentName', '').lower()
            
            # Look for soccer indicators
            if (len(teams) == 2 and 
                any(word in tournament for word in ['soccer', 'football', 'league', 'cup', 'bundesliga', 'premier', 'serie', 'liga']) and
                not game_data.get('isGameFinished', False)):
                
                soccer_games.append({
                    'id': game_id,
                    'home': teams[0].get('name', ''),
                    'away': teams[1].get('name', ''),
                    'tournament': game_data.get('tournamentName', ''),
                    'lastUpdate': game_data.get('lastUpdate', 0)
                })
    
    print(f"\nFound {len(soccer_games)} unfinished soccer games")
    
    # Show recent ones
    soccer_games.sort(key=lambda x: x['lastUpdate'], reverse=True)
    
    for i, game in enumerate(soccer_games[:5]):
        update_time = datetime.fromtimestamp(game['lastUpdate']/1000, tz=timezone.utc)
        print(f"\n{i+1}. {game['home']} vs {game['away']}")
        print(f"   Tournament: {game['tournament']}")
        print(f"   Last Update: {update_time}")
        print(f"   ID: {game['id']}")