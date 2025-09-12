#!/usr/bin/env python3
"""Test the unified dashboard endpoint."""

import requests

try:
    # Test main endpoint
    print("Testing main dashboard...")
    response = requests.get("http://localhost:8888/", timeout=5)
    print(f"Main dashboard status: {response.status_code}")
    
    # Test unified endpoint  
    print("\nTesting unified dashboard...")
    response = requests.get("http://localhost:8888/unified", timeout=5)
    print(f"Unified dashboard status: {response.status_code}")
    
    if response.status_code == 200:
        # Check for key elements
        content = response.text
        print("\nChecking for key elements:")
        print(f"- Has metric cards: {'metric-card' in content}")
        print(f"- Has portfolio section: {'Portfolio Value' in content}")
        print(f"- Has markets table: {'markets-table' in content}")
        print(f"- Has activity feed: {'activity-feed' in content}")
        print(f"- Has grid layout: {'dashboard-grid' in content}")
        
        # Test unified API
        print("\nTesting unified API...")
        api_response = requests.get("http://localhost:8888/api/dashboard/unified", timeout=5)
        print(f"Unified API status: {api_response.status_code}")
        
        if api_response.status_code == 200:
            data = api_response.json()
            print(f"API response keys: {list(data.keys())}")
    else:
        print(f"\nError response: {response.text[:200]}")
        
except Exception as e:
    print(f"Error: {e}")