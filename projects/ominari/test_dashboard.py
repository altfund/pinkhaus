#!/usr/bin/env python3
"""Test the dashboard connection"""

import requests

try:
    # Test basic connection
    r = requests.get('http://localhost:8888', timeout=5)
    print(f"✅ Dashboard is accessible: {r.status_code}")
    
    # Check if page loads
    if "Ominari Trading Dashboard" in r.text:
        print("✅ Dashboard HTML loaded correctly")
    else:
        print("❌ Dashboard HTML missing expected content")
        
    # Check if socket.io is in page
    if "socket.io" in r.text:
        print("✅ Socket.IO script found")
    else:
        print("❌ Socket.IO script missing")
        
    # Check key elements
    elements = ["portfolio-value", "matches-tbody", "activity-feed", "Live Markets"]
    for elem in elements:
        if elem in r.text:
            print(f"✅ Found element: {elem}")
        else:
            print(f"❌ Missing element: {elem}")
            
except Exception as e:
    print(f"❌ Error connecting to dashboard: {e}")