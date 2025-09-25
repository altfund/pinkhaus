#!/usr/bin/env python3
"""
Inspect what the real Overtime API actually provides
"""

import requests
import json
from datetime import datetime

def inspect_api():
    """Inspect the real API structure."""
    print("🔍 Inspecting Overtime V2 API...")
    
    response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
    games = response.json()
    
    print(f"📡 Total games: {len(games)}")
    
    # Look at first few games to understand structure
    sample_games = list(games.items())[:3]
    
    for i, (game_id, info) in enumerate(sample_games):
        print(f"\n=== GAME {i+1} ===")
        print(f"Game ID: {game_id}")
        print("Full game info:")
        print(json.dumps(info, indent=2))
        
        # Check what fields are available
        print(f"\nAvailable fields: {list(info.keys())}")
        
        # Look at teams structure
        teams = info.get('teams', [])
        print(f"Teams count: {len(teams)}")
        for j, team in enumerate(teams):
            print(f"  Team {j+1}: {json.dumps(team, indent=4)}")
            
        # Check for odds/markets
        if 'markets' in info:
            print(f"Markets: {info['markets']}")
        if 'odds' in info:
            print(f"Odds: {info['odds']}")
        if 'lines' in info:
            print(f"Lines: {info['lines']}")
            
        # Check for dates/timing
        if 'startTime' in info:
            print(f"Start time: {info['startTime']}")
        if 'date' in info:
            print(f"Date: {info['date']}")
        if 'gameTime' in info:
            print(f"Game time: {info['gameTime']}")
            
        print("-" * 50)

if __name__ == "__main__":
    inspect_api()