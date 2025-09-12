#!/usr/bin/env python3
"""Force refresh the match dashboard."""

import time
import requests

print("=== REFRESHING MATCH DASHBOARD ===\n")

# Touch the file to update timestamp
import os
os.system("touch web_monitor.py")

print("✓ Updated web_monitor.py timestamp")
print("\n📊 Dashboard URL: http://localhost:8888/unified")
print("\nPlease:")
print("1. Open the dashboard in your browser")
print("2. Do a hard refresh (Ctrl+Shift+R or Cmd+Shift+R)")
print("3. Check the Developer Console (F12) for any errors")
print("\nThe table should now display data with:")
print("- Match rows with all position data inline")
print("- Color-coded edges and P&L")
print("- Filters working properly")

# Quick check
try:
    resp = requests.get("http://localhost:8888/api/dashboard/unified")
    data = resp.json()
    print(f"\n✓ API working: {len(data.get('markets', []))} markets, {len(data.get('positions', {}).get('open', []))} open positions")
except:
    print("\n✗ Could not reach API")