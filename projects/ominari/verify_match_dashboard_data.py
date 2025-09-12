#!/usr/bin/env python3
"""Verify the match dashboard is showing data."""

import requests
import json

def verify_match_dashboard():
    """Check if match dashboard data is displaying correctly."""
    print("=== VERIFYING MATCH DASHBOARD DATA ===\n")
    
    try:
        # 1. Get API data
        print("1. Checking API data...")
        api_resp = requests.get("http://localhost:8888/api/dashboard/unified", timeout=10)
        if api_resp.status_code != 200:
            print(f"✗ API error: {api_resp.status_code}")
            return
            
        data = api_resp.json()
        markets = data.get('markets', [])
        positions = data.get('positions', {})
        
        print(f"✓ API Response:")
        print(f"  - Markets available: {len(markets)}")
        print(f"  - Open positions: {len(positions.get('open', []))}")
        print(f"  - Closed positions: {len(positions.get('closed', []))}")
        
        # 2. Show sample market data
        if markets:
            print("\n2. Sample Market Data:")
            for i, market in enumerate(markets[:3]):
                print(f"\n  Market {i+1}:")
                print(f"    - Match: {market.get('home_team')} vs {market.get('away_team')}")
                print(f"    - Status: {market.get('status', 'ACTIVE')}")
                print(f"    - Odds: H={market.get('home_odds')}, D={market.get('draw_odds')}, A={market.get('away_odds')}")
                signals = market.get('signals', {})
                print(f"    - Edges: H={signals.get('home_edge')}, D={signals.get('draw_edge')}, A={signals.get('away_edge')}")
                
        # 3. Show sample position data
        open_positions = positions.get('open', [])
        if open_positions:
            print("\n3. Sample Open Positions:")
            for i, pos in enumerate(open_positions[:3]):
                print(f"\n  Position {i+1}:")
                print(f"    - Market: {pos.get('market_name')}")
                print(f"    - Outcome: {pos.get('outcome')}")
                print(f"    - Stake: ${pos.get('stake', 0):.2f}")
                print(f"    - Odds: {pos.get('avg_odds', 0):.2f}")
                print(f"    - P&L: ${pos.get('pnl', 0):.2f}")
                
        # 4. Check HTML rendering
        print("\n4. Checking HTML rendering...")
        html_resp = requests.get("http://localhost:8888/unified", timeout=5)
        html = html_resp.text
        
        # Check for data placeholders
        checks = [
            ('Match Dashboard title', '⚽ Match Dashboard' in html),
            ('Table structure', 'id="matches-table"' in html),
            ('Data update function', 'updateMatchesAndPositions' in html),
            ('Filter functionality', 'filterMatches' in html),
        ]
        
        for check, passed in checks:
            print(f"  {'✓' if passed else '✗'} {check}")
            
        # 5. Debug info
        print("\n5. Debug Info:")
        print(f"  - Dashboard URL: http://localhost:8888/unified")
        print(f"  - API endpoint: http://localhost:8888/api/dashboard/unified")
        print(f"  - Data should auto-update every 10 seconds")
        
        print("\n✅ MATCH DASHBOARD VERIFICATION COMPLETE!")
        print("\nIf no data is showing:")
        print("1. Make sure paper trading is running")
        print("2. Check for JavaScript errors in browser console")
        print("3. Verify API endpoint is returning data")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    verify_match_dashboard()