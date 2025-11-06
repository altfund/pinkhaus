#!/usr/bin/env python3
"""Find any upcoming games in Overtime API"""

import requests
import json
from datetime import datetime, timezone
from collections import defaultdict

print("Fetching from Overtime API...")
response = requests.get(
    "https://api.overtime.io/overtime-v2/games-info", 
    headers={'accept': 'application/json'}
)

if response.status_code == 200:
    data = response.json()
    print(f"Total entries: {len(data)}")
    
    now = datetime.now(timezone.utc)
    now_ms = now.timestamp() * 1000
    
    print(f"Current time: {now}")
    print(f"Current time (ms): {int(now_ms)}")
    
    # Analyze all games
    games_by_status = defaultdict(int)
    future_games = []
    recent_games = []
    
    for game_id, game_data in data.items():
        if not isinstance(game_data, dict):
            continue
            
        last_update = game_data.get('lastUpdate', 0)
        is_finished = game_data.get('isGameFinished', False)
        
        # Categorize
        if is_finished:
            games_by_status['finished'] += 1
        elif last_update > now_ms:
            games_by_status['future'] += 1
            future_games.append((game_id, game_data))
        else:
            games_by_status['past_unfinished'] += 1
            # Check if recent (within last 24 hours)
            if last_update > (now_ms - 86400000):  # 24 hours in ms
                recent_games.append((game_id, game_data))
    
    print(f"\nGame Status Summary:")
    for status, count in games_by_status.items():
        print(f"  {status}: {count}")
    
    print(f"\nRecent games (last 24 hours): {len(recent_games)}")
    
    # Show recent unfinished games
    if recent_games:
        print(f"\nRecent unfinished games:")
        for game_id, game_data in recent_games[:5]:
            teams = game_data.get('teams', [])
            if len(teams) >= 2:
                print(f"\n{teams[0].get('name', '?')} vs {teams[1].get('name', '?')}")
                print(f"  ID: {game_id}")
                print(f"  Last Update: {datetime.fromtimestamp(game_data['lastUpdate']/1000, tz=timezone.utc)}")
                print(f"  Status: {game_data.get('gameStatus', 'unknown')}")
    
    # Look for games with odds/positions that might be tradeable
    print(f"\n\nLooking for tradeable games...")
    tradeable = []
    
    for game_id, game_data in data.items():
        if not isinstance(game_data, dict):
            continue
            
        # Must have teams and positions
        teams = game_data.get('teams', [])
        positions = game_data.get('positionNames', [])
        
        if (len(teams) == 2 and 
            len(positions) in [2, 3] and  # 2 for moneyline, 3 for soccer (home/draw/away)
            not game_data.get('isGameFinished', False) and
            all('name' in t for t in teams)):
            
            tradeable.append({
                'gameId': game_id,
                'homeTeam': teams[0]['name'],
                'awayTeam': teams[1]['name'], 
                'positions': positions,
                'lastUpdate': game_data.get('lastUpdate', 0),
                'status': game_data.get('gameStatus', '')
            })
    
    print(f"Found {len(tradeable)} potentially tradeable games")
    
    # Sort by last update time
    tradeable.sort(key=lambda x: x['lastUpdate'], reverse=True)
    
    # Show most recent
    print(f"\nMost recent tradeable games:")
    for i, game in enumerate(tradeable[:10]):
        update_time = datetime.fromtimestamp(game['lastUpdate']/1000, tz=timezone.utc)
        hours_ago = (now - update_time).total_seconds() / 3600
        
        print(f"\n{i+1}. {game['homeTeam']} vs {game['awayTeam']}")
        print(f"   ID: {game['gameId']}")
        print(f"   Positions: {game['positions']}")
        print(f"   Last Update: {update_time} ({hours_ago:.1f} hours ago)")
        print(f"   Status: {game['status']}")
        
    # Check if this looks like the right endpoint
    print(f"\n\n⚠️  Analysis:")
    print(f"This endpoint appears to show historical game results, not upcoming markets.")
    print(f"All games have timestamps in the past.")
    print(f"We may need a different endpoint for live/upcoming markets.")