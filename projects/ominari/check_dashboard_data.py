#!/usr/bin/env python3
"""Check if dashboard data is loading."""

import requests
import json

print("=== CHECKING DASHBOARD DATA ===\n")

try:
    # Check API
    resp = requests.get("http://localhost:8888/api/dashboard/unified", timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        markets = data.get('markets', [])
        positions = data.get('positions', {})
        
        print(f"✓ API Response OK")
        print(f"  - Markets: {len(markets)}")
        print(f"  - Open positions: {len(positions.get('open', []))}")
        print(f"  - Closed positions: {len(positions.get('closed', []))}")
        
        if markets:
            print("\nSample market:")
            m = markets[0]
            print(f"  - ID: {m.get('id')[:20]}...")
            print(f"  - Teams: {m.get('home_team')} vs {m.get('away_team')}")
    else:
        print(f"✗ API Error: {resp.status_code}")
        
    # Check page
    page_resp = requests.get("http://localhost:8888/unified")
    if page_resp.status_code == 200:
        print("\n✓ Dashboard page loads")
        print("\nTROUBLESHOOTING:")
        print("1. Open browser console (F12)")
        print("2. Look for JavaScript errors")
        print("3. Check 'Rendering X matches' message")
        print("4. Try hard refresh (Ctrl+Shift+R)")
    
except Exception as e:
    print(f"✗ Error: {e}")
    
print("\nDashboard URL: http://localhost:8888/unified")