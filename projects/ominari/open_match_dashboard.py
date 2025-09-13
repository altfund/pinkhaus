#!/usr/bin/env python3
"""Open the match dashboard in browser and show status."""

import webbrowser
import requests
import time

def open_match_dashboard():
    """Open the match dashboard and verify it's working."""
    url = "http://localhost:8888/unified"
    
    print("=== OPENING MATCH DASHBOARD ===\n")
    
    # Check if server is running
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            print("✓ Dashboard server is running")
            print(f"✓ Opening dashboard at: {url}")
            
            # Get some data stats
            api_resp = requests.get("http://localhost:8888/api/dashboard/unified")
            if api_resp.status_code == 200:
                data = api_resp.json()
                markets = data.get('markets', [])
                positions = data.get('positions', {})
                
                print("\n📊 MATCH DASHBOARD DATA:")
                print(f"  - Active Markets: {len(markets)}")
                print(f"  - Open Positions: {len(positions.get('open', []))}")
                print(f"  - Closed Positions: {len(positions.get('closed', []))}")
                
                if markets:
                    print("\n⚽ Sample Matches:")
                    for market in markets[:3]:
                        print(f"  - {market.get('home_team')} vs {market.get('away_team')}")
                
            # Open in browser
            webbrowser.open(url)
            
            print("\n✅ Dashboard opened in browser!")
            print("\n🔄 Data updates every 30 seconds")
            print("📋 Use filters: All, Active Positions, Closed, Opportunities")
            print("📊 All position data shown inline in spreadsheet format")
            
        else:
            print(f"✗ Dashboard returned status code: {resp.status_code}")
            print("Make sure web_monitor.py is running")
            
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to dashboard")
        print("Please start the web monitor first:")
        print("  python web_monitor.py")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    open_match_dashboard()