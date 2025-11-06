#!/usr/bin/env python3
"""Fetch real sports markets from Overtime API"""

import requests
import json
from datetime import datetime, timezone

def fetch_overtime_markets():
    """Fetch and structure markets from Overtime API"""
    
    response = requests.get(
        "https://api.overtime.io/overtime-v2/games-info",
        headers={'accept': 'application/json'}
    )
    
    if response.status_code != 200:
        return []
    
    data = response.json()
    markets = []
    
    # Define real sports keywords
    soccer_keywords = ['fc', 'united', 'city', 'athletic', 'real', 'barcelona', 'munich', 'juventus', 'milan', 'paris', 'chelsea', 'liverpool', 'arsenal']
    basketball_keywords = ['lakers', 'celtics', 'warriors', 'heat', 'bulls', 'knicks', 'nets', 'clippers']
    
    for game_id, game_data in data.items():
        if not isinstance(game_data, dict):
            continue
            
        teams = game_data.get('teams', [])
        is_finished = game_data.get('isGameFinished', False)
        tournament = game_data.get('tournamentName', '')
        positions = game_data.get('positionNames', [])
        
        # Skip finished games
        if is_finished or len(teams) != 2:
            continue
            
        # Get team names
        home_team = teams[0].get('name', '')
        away_team = teams[1].get('name', '')
        
        # Skip esports
        if any(word in (home_team + away_team + tournament).lower() for word in ['esport', 'gaming', 'dust2', 'esl', 'cct', 'blast']):
            continue
            
        # Determine sport
        combined_text = (home_team + away_team + tournament).lower()
        
        if any(keyword in combined_text for keyword in soccer_keywords):
            sport = 'Soccer'
        elif any(keyword in combined_text for keyword in basketball_keywords):
            sport = 'Basketball'
        elif 'nfl' in combined_text or 'football' in tournament.lower():
            sport = 'Football'
        elif 'nba' in tournament.lower():
            sport = 'Basketball'
        elif 'tennis' in tournament.lower() or 'atp' in tournament.lower():
            sport = 'Tennis'
        else:
            # Try to infer from positions
            if len(positions) == 3 and positions[1] in ['X', 'Draw']:
                sport = 'Soccer'
            else:
                sport = 'Unknown'
        
        # Create market structure
        market = {
            'market_id': game_id,
            'match_id': game_id,
            'source': 'overtime_api',
            'sport': sport,
            'league': tournament if tournament != 'N/A' else 'Unknown',
            'home_team': home_team,
            'away_team': away_team,
            'maturity_date': datetime.fromtimestamp(game_data.get('lastUpdate', 0)/1000, tz=timezone.utc),
            'is_finished': is_finished,
            'positions': positions,
            'odds': {}  # Would need separate endpoint for odds
        }
        
        # Generate mock odds for now (would need real odds endpoint)
        if len(positions) == 3:  # Soccer style
            market['odds'] = {
                'home': 2.50,
                'draw': 3.20,
                'away': 2.80
            }
        else:  # Two-way
            market['odds'] = {
                'home': 1.90,
                'away': 1.90
            }
        
        markets.append(market)
    
    # Filter to only upcoming games (not stale data)
    now = datetime.now(timezone.utc)
    upcoming_markets = [m for m in markets if m['maturity_date'] > now or 
                       (now - m['maturity_date']).total_seconds() < 7200]  # Within 2 hours
    
    # Sort by maturity date
    upcoming_markets.sort(key=lambda x: x['maturity_date'])
    
    return upcoming_markets


if __name__ == "__main__":
    print("Fetching real sports markets from Overtime...")
    
    markets = fetch_overtime_markets()
    
    print(f"\nFound {len(markets)} upcoming markets")
    
    # Group by sport
    by_sport = {}
    for market in markets:
        sport = market['sport']
        by_sport[sport] = by_sport.get(sport, 0) + 1
    
    print("\nMarkets by sport:")
    for sport, count in by_sport.items():
        print(f"  {sport}: {count}")
    
    # Show sample markets
    print("\nSample upcoming markets:")
    
    for sport in ['Soccer', 'Basketball', 'Football']:
        sport_markets = [m for m in markets if m['sport'] == sport]
        if sport_markets:
            print(f"\n{sport}:")
            for market in sport_markets[:3]:
                print(f"  {market['home_team']} vs {market['away_team']}")
                print(f"    League: {market['league']}")
                print(f"    Time: {market['maturity_date']}")
                print(f"    Odds: {market['odds']}")
    
    # Save markets for trading system
    if markets:
        with open('overtime_markets.json', 'w') as f:
            # Convert datetime objects to strings for JSON
            for market in markets:
                market['maturity_date'] = market['maturity_date'].isoformat()
            json.dump(markets, f, indent=2)
        print(f"\n✅ Saved {len(markets)} markets to overtime_markets.json")