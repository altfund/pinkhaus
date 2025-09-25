#!/usr/bin/env python3
"""
Find actual games (not futures) in Overtime API
"""

import requests
import json
from datetime import datetime

def find_real_games():
    """Find actual head-to-head games."""
    print("🔍 Looking for real head-to-head games...")
    
    response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
    games = response.json()
    
    print(f"📡 Total games: {len(games)}")
    
    # Look for actual team vs team games
    real_games = []
    futures_games = []
    
    for game_id, info in games.items():
        teams = info.get('teams', [])
        if len(teams) != 2:
            continue
            
        home = next((t for t in teams if t.get('isHome')), {}).get('name', '')
        away = next((t for t in teams if not t.get('isHome')), {}).get('name', '')
        
        # Check if this looks like a future/winner market
        combined = f'{home} {away}'.lower()
        if any(term in combined for term in ['winner', 'champion', 'mlb', 'nfl', 'nba', 'league winner', 'division winner', 'world series', 'super bowl', 'finals']):
            futures_games.append({
                'id': game_id,
                'home': home,
                'away': away,
                'tournament': info.get('tournamentName', ''),
                'status': info.get('gameStatus', ''),
                'finished': info.get('isGameFinished', False)
            })
        else:
            # This might be a real game
            real_games.append({
                'id': game_id,
                'home': home,
                'away': away,
                'tournament': info.get('tournamentName', ''),
                'status': info.get('gameStatus', ''),
                'finished': info.get('isGameFinished', False),
                'last_update': info.get('lastUpdate', 0)
            })
    
    print(f"\n📊 ANALYSIS:")
    print(f"Real games found: {len(real_games)}")
    print(f"Futures markets found: {len(futures_games)}")
    
    print(f"\n🏆 Sample Real Games:")
    for i, game in enumerate(real_games[:10]):
        print(f"  {i+1}. {game['home']} vs {game['away']}")
        if game['tournament']:
            print(f"     Tournament: {game['tournament']}")
        if game['status']:
            print(f"     Status: {game['status']}")
        print(f"     Finished: {game['finished']}")
        
    print(f"\n🎯 Sample Futures Markets:")
    for i, game in enumerate(futures_games[:5]):
        print(f"  {i+1}. {game['home']} vs {game['away']}")
        
    # Check for odds data
    print(f"\n🔍 Checking for odds/market data...")
    sample_game_id = list(games.keys())[0]
    sample_info = games[sample_game_id]
    
    # Check if there are separate odds endpoints
    print("Available fields in game data:")
    for field in sample_info.keys():
        print(f"  - {field}: {type(sample_info[field])}")

if __name__ == "__main__":
    find_real_games()