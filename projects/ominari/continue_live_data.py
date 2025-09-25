#!/usr/bin/env python3
"""
Continue with live data - test the API to get real team names
Based on the successful 452 blockchain events we found
"""

import os
os.environ['PG_PORT'] = '5999'

import requests
import json
from datetime import datetime, timezone, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_overtime_api():
    """Test the Overtime API to get real data like we found before."""
    logger.info("🚀 Testing Overtime API for live data...")
    
    # Test the working endpoint from our previous success
    try:
        response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
        logger.info(f"API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            games = response.json()
            logger.info(f"✅ Found {len(games)} total games in API!")
            
            # Look for active/upcoming games
            active_games = []
            finished_games = 0
            future_games = 0
            
            for game_id, info in games.items():
                is_finished = info.get('isGameFinished', True)
                teams = info.get('teams', [])
                
                if is_finished:
                    finished_games += 1
                else:
                    future_games += 1
                    if len(teams) == 2:
                        home = next((t for t in teams if t.get('isHome')), {}).get('name', '')
                        away = next((t for t in teams if not t.get('isHome')), {}).get('name', '')
                        
                        # Skip championship/future markets
                        combined = f"{home} {away}".lower()
                        if not any(term in combined for term in ['winner', 'championship', 'mvp', 'future']):
                            active_games.append({
                                'id': game_id,
                                'home': home,
                                'away': away,
                                'tournament': info.get('tournamentName', 'Unknown'),
                                'status': info.get('gameStatus', 'Unknown'),
                                'sport': info.get('sport', 'Unknown')
                            })
            
            logger.info(f"📊 Games breakdown:")
            logger.info(f"  - Finished: {finished_games}")
            logger.info(f"  - Upcoming: {future_games}")
            logger.info(f"  - Active tradeable: {len(active_games)}")
            
            # Show sample active games
            logger.info(f"🎯 Sample active games:")
            for i, game in enumerate(active_games[:10]):
                logger.info(f"  {i+1}. {game['home']} vs {game['away']} ({game['tournament']}) - {game['sport']}")
                
            return active_games
            
        else:
            logger.error(f"❌ API returned status {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ API error: {e}")
        
    return []

def test_other_endpoints():
    """Test other potential endpoints."""
    logger.info("🔍 Testing other potential endpoints...")
    
    endpoints = [
        'https://api.overtime.io/overtime-v2/active-markets',
        'https://api.overtime.io/overtime-v2/live-markets', 
        'https://api.overtime.io/overtime-v2/markets',
        'https://overtimemarketsv2.xyz/overtime-v2/games-info',
        'https://v2.contracts.overtime.io/games'
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=10)
            logger.info(f"  {endpoint}: {response.status_code}")
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        logger.info(f"    → Dict with {len(data)} keys")
                    elif isinstance(data, list):
                        logger.info(f"    → List with {len(data)} items")
                    else:
                        logger.info(f"    → Data type: {type(data)}")
                except:
                    logger.info(f"    → Non-JSON response ({len(response.text)} chars)")
        except Exception as e:
            logger.debug(f"  {endpoint}: {e}")

def main():
    logger.info("🏁 Continuing live data investigation...")
    logger.info("Based on our previous success with 452 blockchain events")
    
    # Test the main API
    active_games = test_overtime_api()
    
    # Test other endpoints  
    test_other_endpoints()
    
    if active_games:
        logger.info(f"✅ SUCCESS: Found {len(active_games)} active tradeable games!")
        logger.info("🎯 Next steps:")
        logger.info("  1. These are real team names from live API")
        logger.info("  2. We can use these to replace sample data in dashboard")
        logger.info("  3. We can sync these to PostgreSQL database")
        logger.info("  4. Dashboard will show real upcoming games!")
    else:
        logger.warning("❌ No active games found - may need to investigate further")
        
    logger.info("✅ Live data investigation complete!")

if __name__ == "__main__":
    main()