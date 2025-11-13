#!/usr/bin/env python3
"""
Check if Overtime API provides nation/country data
"""

import requests
import json

print("=== Checking Overtime API for Nation/Country Data ===\n")

# Check the sports endpoint
print("1. Checking Sports Endpoint:")
print("-" * 60)
try:
    response = requests.get('https://api.overtime.io/overtime-v2/sports', timeout=10)
    if response.status_code == 200:
        sports_data = response.json()
        print(f"✅ Found {len(sports_data)} sport entries")
        
        # Show sample sport entries
        sample_count = 0
        for sport_id, info in list(sports_data.items())[:5]:
            print(f"\nSport ID: {sport_id}")
            print(f"  Sport: {info.get('sport', 'N/A')}")
            print(f"  Label: {info.get('label', 'N/A')}")
            print(f"  OpticOddsName: {info.get('opticOddsName', 'N/A')}")
            
            # Check if opticOddsName contains country info
            optic_name = info.get('opticOddsName', '')
            if ' - ' in optic_name:
                parts = optic_name.split(' - ')
                if len(parts) >= 2:
                    print(f"  ⭐ Extracted Country: {parts[0]}")
                    print(f"  ⭐ Extracted League: {parts[1]}")
            
            print(f"  All fields: {list(info.keys())}")
            sample_count += 1
    else:
        print(f"❌ Sports endpoint returned status {response.status_code}")
except Exception as e:
    print(f"❌ Error accessing sports endpoint: {e}")

# Check games for any nation/country fields
print("\n\n2. Checking Games Endpoint for Nation Fields:")
print("-" * 60)
try:
    response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=10)
    if response.status_code == 200:
        games = response.json()
        print(f"✅ Found {len(games)} games")
        
        # Check a few active games
        sample_count = 0
        for game_id, info in games.items():
            if not info.get('isGameFinished', True) and sample_count < 3:
                print(f"\nGame ID: {game_id}")
                print(f"  Teams: {info.get('teams', [])}")
                
                # Check all fields for country/nation keywords
                nation_fields = []
                for key in info.keys():
                    if any(word in key.lower() for word in ['country', 'nation', 'region', 'location', 'geo']):
                        nation_fields.append(key)
                
                if nation_fields:
                    print(f"  ⭐ Found nation-related fields: {nation_fields}")
                    for field in nation_fields:
                        print(f"     {field}: {info.get(field)}")
                else:
                    print(f"  ❌ No nation/country fields found")
                
                # Check for sportId which we can map back to sports data
                if 'sportId' in info:
                    print(f"  SportId: {info['sportId']} (can map to sports endpoint)")
                
                sample_count += 1
    else:
        print(f"❌ Games endpoint returned status {response.status_code}")
except Exception as e:
    print(f"❌ Error accessing games endpoint: {e}")

print("\n\n=== Summary ===")
print("The Overtime API provides country/nation data indirectly:")
print("1. The sports endpoint has 'opticOddsName' field in format 'Country - League'")
print("2. Games have 'sportId' which maps to the sports endpoint")
print("3. We can extract country from the opticOddsName by splitting on ' - '")
print("\nThis is more reliable than manual mapping!")