#!/usr/bin/env python3
"""Debug API to see match status"""

import requests
from datetime import datetime, timezone

# Fetch from API
response = requests.get(
    "https://api.overtime.io/overtime-v2/games-info",
    headers={'accept': 'application/json'}
)

if response.status_code == 200:
    data = response.json()
    games = data.get('games', [])
    
    print(f"Total games in API: {len(games)}")
    
    # Check date range
    if games:
        dates = []
        for game in games:
            maturity = game.get('maturity', 0) / 1000  # Convert from ms to seconds
            game_date = datetime.fromtimestamp(maturity, tz=timezone.utc)
            dates.append(game_date)
        
        earliest = min(dates)
        latest = max(dates)
        print(f"Date range: {earliest} to {latest}")
    
    # Find matches from Oct 30 that should be finished
    oct_30_games = []
    for game in games:
        maturity = game.get('maturity', 0) / 1000  # Convert from ms to seconds
        game_date = datetime.fromtimestamp(maturity, tz=timezone.utc)
        
        # Check if game was on Oct 30 (any year for now)
        if game_date.month == 10 and game_date.day == 30:
            oct_30_games.append(game)
    
    print(f"\nFound {len(oct_30_games)} games from Oct 30 (any year)")
    
    if oct_30_games:
        print("\nSample games:")
        for i, game in enumerate(oct_30_games[:5]):
            maturity = game.get('maturity', 0) / 1000
            game_date = datetime.fromtimestamp(maturity, tz=timezone.utc)
            
            print(f"\n{i+1}. {game.get('homeTeam')} vs {game.get('awayTeam')}")
            print(f"   Game ID: {game.get('gameId')}")
            print(f"   Date: {game_date}")
            print(f"   isResolved: {game.get('isResolved')}")
            print(f"   isCanceled: {game.get('isCanceled')}")
            print(f"   Home Score: {game.get('homeScore')}")
            print(f"   Away Score: {game.get('awayScore')}")
            
    # Check if any are resolved
    resolved_count = sum(1 for g in oct_30_games if g.get('isResolved'))
    print(f"\nResolved games: {resolved_count}/{len(oct_30_games)}")
else:
    print(f"API error: {response.status_code}")