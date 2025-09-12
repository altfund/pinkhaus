#!/usr/bin/env python3
"""Simple dashboard test."""

import requests

print("Testing Unified Dashboard...")

# Test the API
print("\n1. API Test:")
response = requests.get("http://localhost:8888/api/dashboard/unified")
data = response.json()
print(f"   Status: {response.status_code}")
print(f"   Portfolio Value: ${data['portfolio']['total_value']:.2f}")
print(f"   ROI: {data['performance'].get('roi', 0):.1f}%")

# Test dashboard HTML
print("\n2. Dashboard HTML Test:")
response = requests.get("http://localhost:8888/unified")
html = response.text

# Check for key elements
checks = [
    ('JavaScript loadDashboardData', 'loadDashboardData()' in html),
    ('Portfolio value element', 'id="portfolio-value"' in html),
    ('Win rate element', 'id="win-rate"' in html),
    ('Update metrics function', 'updateMetrics(' in html),
    ('Console logging', 'console.log(' in html)
]

for name, found in checks:
    status = "✓" if found else "✗"
    print(f"   {status} {name}")

print("\n3. To debug in browser:")
print("   1. Open http://localhost:8888/unified")
print("   2. Press F12 for Developer Tools")
print("   3. Check Console tab for errors")
print("   4. Check Network tab to see if API is called")

print("\nThe dashboard should be working now!")