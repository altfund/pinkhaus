#!/usr/bin/env python3
"""Find real soccer teams in cached markets"""

import json

# Load markets
with open('overtime_markets.json', 'r') as f:
    markets = json.load(f)

# Real soccer team indicators
soccer_indicators = [
    # English
    'manchester', 'liverpool', 'chelsea', 'arsenal', 'tottenham', 'leicester', 'everton',
    # Spanish
    'real madrid', 'barcelona', 'atletico', 'valencia', 'sevilla', 'villarreal',
    # German
    'bayern', 'dortmund', 'leipzig', 'leverkusen', 'frankfurt',
    # Italian
    'juventus', 'milan', 'inter', 'roma', 'napoli', 'lazio',
    # French
    'paris', 'psg', 'marseille', 'lyon', 'monaco',
    # Generic
    ' fc', ' cf', 'united', 'city', 'athletic', 'sporting'
]

real_soccer = []

for market in markets:
    home = market['home_team'].lower()
    away = market['away_team'].lower()
    
    # Check if it has real soccer team names
    if any(indicator in home or indicator in away for indicator in soccer_indicators):
        # Additional filtering - exclude obvious non-soccer
        if not any(word in home + away for word in ['grizzlies', 'heat', 'lakers', 'ufc', 'nba', 'nfl', 'canadiens', 'mammoth']):
            real_soccer.append(market)

print(f"Found {len(real_soccer)} real soccer matches")

# Group by league
leagues = {}
for market in real_soccer:
    league = market['league']
    if league not in leagues:
        leagues[league] = []
    leagues[league].append(market)

print("\nReal soccer matches by league:")
for league, matches in sorted(leagues.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
    print(f"\n{league}: {len(matches)} matches")
    # Show first few
    for match in matches[:3]:
        print(f"  - {match['home_team']} vs {match['away_team']}")

# Save real soccer markets
if real_soccer:
    with open('real_soccer_markets.json', 'w') as f:
        json.dump(real_soccer, f, indent=2)
    print(f"\n✅ Saved {len(real_soccer)} real soccer matches to real_soccer_markets.json")
    
# Show some with proper odds structure
print("\n\nChecking odds structure...")
for market in real_soccer[:5]:
    print(f"\n{market['home_team']} vs {market['away_team']}")
    print(f"  Odds: {market['odds']}")
    print(f"  Sport: {market['sport']}")