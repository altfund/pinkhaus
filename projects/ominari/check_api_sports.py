#!/usr/bin/env python3
"""Check what sports are available in the API"""

import requests
import json
from collections import Counter

url = "https://api.overtime.io/overtime-v2/games-info"

print("🏟️  Checking available sports in Overtime API...\n")

response = requests.get(url, headers={'accept': 'application/json'})
data = response.json()

sports = Counter()
tournaments = Counter()
all_tags = []

print(f"Total markets: {len(data)}\n")

# Sample some markets
sample_count = 0
for market_id, market_data in data.items():
    if isinstance(market_data, dict):
        # Collect info
        sport = market_data.get('sport', 'Unknown')
        tournament = market_data.get('tournamentName', 'Unknown')
        tags = market_data.get('tags', [])
        is_finished = market_data.get('isGameFinished', True)
        
        if not is_finished:
            sports[sport] += 1
            tournaments[tournament] += 1
            all_tags.extend(tags)
            
            # Show sample
            if sample_count < 5:
                print(f"Market {sample_count + 1}:")
                print(f"  ID: {market_id[:20]}...")
                print(f"  Tournament: {tournament}")
                print(f"  Sport: {sport}")
                print(f"  Tags: {tags}")
                print(f"  Teams: {market_data.get('positionNames', [])}")
                print(f"  Odds: {market_data.get('odds', [])}")
                print()
                sample_count += 1

print("\n📊 Sport distribution (unfinished markets):")
for sport, count in sports.most_common():
    print(f"  {sport}: {count}")

print("\n🏆 Tournament distribution (top 10):")
for tournament, count in tournaments.most_common(10):
    print(f"  {tournament}: {count}")

print("\n🏷️  Unique tags:")
unique_tags = set(all_tags)
for tag in sorted(unique_tags)[:20]:
    print(f"  {tag}")

# Look for soccer specifically
print("\n⚽ Soccer/Football search:")
soccer_count = 0
for market_id, market_data in data.items():
    if isinstance(market_data, dict) and not market_data.get('isGameFinished', True):
        tournament = market_data.get('tournamentName', '').lower()
        sport = market_data.get('sport', '').lower()
        tags_str = ' '.join(str(t).lower() for t in market_data.get('tags', []))
        
        if 'soccer' in tournament or 'football' in tournament or 'soccer' in sport or 'football' in sport or 'soccer' in tags_str or 'football' in tags_str:
            soccer_count += 1
            if soccer_count <= 3:
                print(f"\nFound soccer market:")
                print(f"  Tournament: {market_data.get('tournamentName')}")
                print(f"  Teams: {market_data.get('positionNames', [])}")
                print(f"  Odds: {market_data.get('odds', [])}")

print(f"\nTotal soccer markets found: {soccer_count}")