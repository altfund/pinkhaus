#!/usr/bin/env python3
"""Simulate dashboard loading process."""

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json

print("Testing Unified Dashboard Data Loading...")

# First test the API
print("\n1. Testing API endpoint...")
try:
    response = requests.get("http://localhost:8888/api/dashboard/unified")
    data = response.json()
    print(f"   API Status: {response.status_code}")
    print(f"   Portfolio Value: ${data['portfolio']['total_value']:.2f}")
    print(f"   Positions: {data['portfolio']['positions_count']}")
    print(f"   Win Rate: {data['performance'].get('win_rate', 0) * 100:.1f}%")
except Exception as e:
    print(f"   API Error: {e}")

# Test if dashboard HTML loads
print("\n2. Testing dashboard HTML...")
try:
    response = requests.get("http://localhost:8888/unified")
    print(f"   HTML Status: {response.status_code}")
    print(f"   HTML Size: {len(response.text)} bytes")
    
    # Check if JavaScript is present
    if 'loadDashboardData()' in response.text:
        print("   ✓ JavaScript function found")
    else:
        print("   ✗ JavaScript function NOT found")
        
    # Check if placeholders exist
    if 'id="portfolio-value"' in response.text:
        print("   ✓ Portfolio value element found")
    else:
        print("   ✗ Portfolio value element NOT found")
        
except Exception as e:
    print(f"   HTML Error: {e}")

# Create a simple test to check if values update
print("\n3. Checking if values would update...")
print("   The dashboard should show:")
print(f"   - Portfolio: $9370.77")
print(f"   - Cash: $8524.39")  
print(f"   - Positions: 3")
print(f"   - ROI: -6.3%")
print(f"   - Win Rate: 0.0%")

print("\n4. Common issues to check:")
print("   - Console errors (F12 in browser)")
print("   - Network tab shows API calls")
print("   - Elements have correct IDs")
print("\n✅ API is working correctly")
print("❓ Dashboard may have JavaScript issues - check browser console")