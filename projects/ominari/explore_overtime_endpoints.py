#!/usr/bin/env python3
"""
Explore other Overtime API endpoints for live markets
"""

import requests
import json

def explore_endpoints():
    """Try different Overtime API endpoints."""
    
    base_urls = [
        'https://api.overtime.io/overtime-v2/',
        'https://overtime.io/api/',
        'https://api.overtime.io/',
        'https://overtime.markets/api/',
    ]
    
    endpoints = [
        'games',
        'markets',
        'odds',
        'live-markets',
        'active-markets',
        'upcoming-games',
        'current-markets',
        'betting-markets',
        'game-markets',
        'live-odds',
        'sports-markets'
    ]
    
    print("🔍 Exploring Overtime API endpoints...")
    
    for base in base_urls:
        print(f"\n📡 Testing base URL: {base}")
        
        for endpoint in endpoints:
            try:
                url = f"{base}{endpoint}"
                print(f"  Trying: {url}")
                
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    print(f"  ✅ SUCCESS: {len(data) if isinstance(data, (list, dict)) else 'N/A'} items")
                    
                    # Show sample of successful responses
                    if isinstance(data, dict) and len(data) > 0:
                        sample_key = list(data.keys())[0]
                        print(f"     Sample key: {sample_key}")
                        print(f"     Sample data: {json.dumps(data[sample_key], indent=6)[:300]}...")
                    elif isinstance(data, list) and len(data) > 0:
                        print(f"     Sample item: {json.dumps(data[0], indent=6)[:300]}...")
                        
                elif response.status_code == 404:
                    print(f"  ❌ Not Found (404)")
                else:
                    print(f"  ❌ Error {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ Exception: {str(e)[:50]}...")
    
    # Also check known working endpoint for more paths
    print(f"\n🔍 Checking known endpoint for more paths...")
    try:
        response = requests.get('https://api.overtime.io/overtime-v2/', timeout=10)
        print(f"Base endpoint status: {response.status_code}")
        if response.status_code == 200:
            print("Base endpoint content:")
            print(response.text[:500])
    except Exception as e:
        print(f"Base endpoint error: {e}")

if __name__ == "__main__":
    explore_endpoints()