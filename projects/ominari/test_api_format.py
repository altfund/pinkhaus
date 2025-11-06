#!/usr/bin/env python3
"""Test API response format"""

import requests
import json

url = "https://api.overtime.io/overtime-v2/games-info"

print("🌐 Testing Overtime API format...\n")

response = requests.get(url, headers={'accept': 'application/json'})
print(f"Status: {response.status_code}")
print(f"Content Type: {response.headers.get('content-type')}")

# Get raw text
raw = response.text
print(f"\nRaw response (first 500 chars):")
print(raw[:500])

# Check if it's hex
if all(c in '0123456789abcdefABCDEF' for c in raw.strip()):
    print("\n✅ Response appears to be hex-encoded")
    
    # Try to decode
    try:
        decoded = bytes.fromhex(raw.strip()).decode('utf-8')
        print(f"\nDecoded (first 500 chars):")
        print(decoded[:500])
        
        # Try to parse as JSON
        data = json.loads(decoded)
        print(f"\n✅ Successfully parsed JSON. Type: {type(data)}")
        
        if isinstance(data, list):
            print(f"   List with {len(data)} items")
            if data:
                print(f"   First item type: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"   Keys: {list(data[0].keys())[:10]}")
        elif isinstance(data, dict):
            print(f"   Dict with keys: {list(data.keys())}")
            
    except Exception as e:
        print(f"\n❌ Failed to decode/parse: {e}")
else:
    print("\n❌ Response is not hex-encoded")
    
    # Try direct JSON parse
    try:
        data = response.json()
        print(f"\n✅ Direct JSON parse successful. Type: {type(data)}")
    except Exception as e:
        print(f"\n❌ JSON parse failed: {e}")