#!/usr/bin/env python3
"""Test Overtime API to see what markets are available."""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_overtime_api():
    """Test Overtime API."""
    api_key = os.getenv("OVERTIME_API_KEY")
    if not api_key:
        print("No OVERTIME_API_KEY found in environment")
        return
        
    network_id = os.getenv("OVERTIME_NETWORK_ID", "10")  # 10 = Optimism
    url = f"https://api.overtime.io/overtime-v2/networks/{network_id}/markets"
    
    headers = {"x-api-key": api_key}
    
    print(f"Testing Overtime API at: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Count sports
            sports = {}
            total_markets = 0
            
            for sport, leagues in data.items():
                sport_count = 0
                for league_id, markets in leagues.items():
                    sport_count += len(markets)
                sports[sport] = sport_count
                total_markets += sport_count
            
            print(f"\nTotal markets: {total_markets}")
            print("\nMarkets by sport:")
            for sport, count in sorted(sports.items(), key=lambda x: x[1], reverse=True):
                print(f"  {sport}: {count}")
                
            # Show sample soccer market
            if 'Soccer' in data:
                for league_id, markets in data['Soccer'].items():
                    if markets:
                        market = markets[0]
                        print("\nSample Soccer market:")
                        print(f"  Game ID: {market.get('gameId')}")
                        print(f"  Teams: {market.get('homeTeam')} vs {market.get('awayTeam')}")
                        print(f"  League: {market.get('leagueName')}")
                        print(f"  Type: {market.get('type')}")
                        print(f"  Maturity: {market.get('maturityDate')}")
                        if 'odds' in market:
                            print(f"  Odds: {market['odds']}")
                        break
                        
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_overtime_api()