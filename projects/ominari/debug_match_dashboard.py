#!/usr/bin/env python3
"""Debug match dashboard data format."""

import requests
import json

def debug_dashboard():
    """Debug the match dashboard data."""
    print("=== DEBUGGING MATCH DASHBOARD DATA ===\n")
    
    # Get API data
    resp = requests.get("http://localhost:8888/api/dashboard/unified")
    data = resp.json()
    
    print("1. MARKETS:")
    markets = data.get('markets', [])
    for i, market in enumerate(markets[:2]):
        print(f"\nMarket {i}:")
        print(f"  ID: {market.get('id')}")
        print(f"  market_id: {market.get('market_id')}")
        print(f"  Match: {market.get('home_team')} vs {market.get('away_team')}")
        
    print("\n2. POSITIONS:")
    positions = data.get('positions', {})
    open_pos = positions.get('open', [])
    for i, pos in enumerate(open_pos[:2]):
        print(f"\nPosition {i}:")
        print(f"  market_id: {pos.get('market_id')}")
        print(f"  market_name: {pos.get('market_name')}")
        print(f"  outcome: {pos.get('outcome')}")
        
    # Check if IDs match
    print("\n3. ID MATCHING CHECK:")
    market_ids = [m.get('id') for m in markets]
    position_market_ids = [p.get('market_id') for p in open_pos]
    
    print(f"\nMarket IDs format: {market_ids[:2]}")
    print(f"Position market IDs format: {position_market_ids[:2]}")
    
    # Check for matches
    matches = []
    for pid in position_market_ids:
        if pid in market_ids:
            matches.append(pid)
            
    print(f"\nMatching IDs found: {len(matches)}")
    
if __name__ == "__main__":
    debug_dashboard()