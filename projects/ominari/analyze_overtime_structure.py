#!/usr/bin/env python3
"""Analyze Overtime API structure to find real sports"""

import requests
import json
from datetime import datetime, timezone
from collections import defaultdict

print("Analyzing Overtime API games-info structure...")
response = requests.get(
    "https://api.overtime.io/overtime-v2/games-info",
    headers={'accept': 'application/json'}
)

if response.status_code == 200:
    data = response.json()
    
    # Analyze tournament names
    tournaments = defaultdict(int)
    sports = defaultdict(int)
    
    # Sample games by characteristics
    two_team_games = []
    three_position_games = []
    
    for game_id, game_data in data.items():
        if isinstance(game_data, dict):
            tournament = game_data.get('tournamentName', '')
            teams = game_data.get('teams', [])
            positions = game_data.get('positionNames', [])
            is_finished = game_data.get('isGameFinished', False)
            
            if tournament:
                tournaments[tournament] += 1
            
            # Categorize by number of teams and positions
            if len(teams) == 2 and not is_finished:
                # Traditional 2-team matchup
                team1 = teams[0].get('name', '')
                team2 = teams[1].get('name', '')
                
                # Skip esports
                if any(word in team1.lower() + team2.lower() for word in ['esport', 'gaming', 'hyperion']):
                    sports['Esports'] += 1
                # Check for real soccer teams
                elif any(word in team1.lower() + team2.lower() for word in ['fc', 'united', 'city', 'atletico', 'real madrid', 'barcelona']):
                    sports['Soccer'] += 1
                    if len(positions) == 3:  # Home/Draw/Away
                        three_position_games.append({
                            'id': game_id,
                            'home': team1,
                            'away': team2,
                            'tournament': tournament,
                            'positions': positions,
                            'lastUpdate': game_data.get('lastUpdate', 0)
                        })
                # NBA teams
                elif any(word in team1.lower() + team2.lower() for word in ['lakers', 'celtics', 'warriors', 'heat']):
                    sports['Basketball'] += 1
                # NFL teams  
                elif any(word in team1.lower() + team2.lower() for word in ['chiefs', 'patriots', 'cowboys', 'eagles']):
                    sports['Football'] += 1
                else:
                    two_team_games.append({
                        'id': game_id,
                        'home': team1,
                        'away': team2,
                        'tournament': tournament,
                        'positions': positions,
                        'lastUpdate': game_data.get('lastUpdate', 0)
                    })
    
    print(f"\nTournaments found ({len(tournaments)} unique):")
    # Show top tournaments by game count
    sorted_tournaments = sorted(tournaments.items(), key=lambda x: x[1], reverse=True)
    for tournament, count in sorted_tournaments[:20]:
        if tournament and 'esport' not in tournament.lower():
            print(f"  {tournament}: {count} games")
    
    print(f"\nSports breakdown:")
    for sport, count in sports.items():
        print(f"  {sport}: {count}")
    
    print(f"\nGames with 3 positions (likely soccer with Home/Draw/Away): {len(three_position_games)}")
    
    # Show recent three-position games
    three_position_games.sort(key=lambda x: x['lastUpdate'], reverse=True)
    
    print("\nRecent soccer-style games (3 positions):")
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    
    for i, game in enumerate(three_position_games[:10]):
        # Skip obvious non-soccer
        if 'esport' in game['home'].lower() or 'esport' in game['away'].lower():
            continue
            
        update_time = datetime.fromtimestamp(game['lastUpdate']/1000, tz=timezone.utc)
        hours_ago = (now_ms/1000 - game['lastUpdate']/1000) / 3600
        
        print(f"\n{i+1}. {game['home']} vs {game['away']}")
        print(f"   Tournament: {game['tournament']}")
        print(f"   Positions: {game['positions']}")
        print(f"   Last Update: {update_time} ({hours_ago:.1f} hours ago)")
        print(f"   ID: {game['id']}")
        
        # Check if positions make sense for soccer
        if game['positions'] == ['1', 'X', '2'] or game['positions'] == ['Home', 'Draw', 'Away']:
            print("   ✅ Standard soccer betting positions!")
            
    # Look for games by examining team names more carefully
    print("\n\nSearching for real soccer teams...")
    real_soccer_teams = [
        'manchester united', 'manchester city', 'liverpool', 'chelsea', 'arsenal',
        'real madrid', 'barcelona', 'atletico madrid', 'bayern munich', 'dortmund',
        'juventus', 'milan', 'inter', 'paris saint-germain', 'marseille'
    ]
    
    found_real_soccer = []
    
    for game_id, game_data in data.items():
        if isinstance(game_data, dict):
            teams = game_data.get('teams', [])
            if len(teams) == 2:
                team1 = teams[0].get('name', '').lower()
                team2 = teams[1].get('name', '').lower()
                
                if any(real_team in team1 or real_team in team2 for real_team in real_soccer_teams):
                    found_real_soccer.append({
                        'id': game_id,
                        'home': teams[0].get('name', ''),
                        'away': teams[1].get('name', ''),
                        'tournament': game_data.get('tournamentName', ''),
                        'isFinished': game_data.get('isGameFinished', False),
                        'lastUpdate': game_data.get('lastUpdate', 0)
                    })
    
    print(f"\nFound {len(found_real_soccer)} games with recognized soccer teams")
    
    # Show unfinished ones
    unfinished_real = [g for g in found_real_soccer if not g['isFinished']]
    print(f"Unfinished games with real teams: {len(unfinished_real)}")
    
    for game in unfinished_real[:5]:
        print(f"\n{game['home']} vs {game['away']}")
        print(f"  Tournament: {game['tournament']}")
        print(f"  ID: {game['id']}")