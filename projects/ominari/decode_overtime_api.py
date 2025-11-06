#!/usr/bin/env python3
"""Decode Overtime API hex response"""

import requests
import json

print("Fetching from Overtime API...")
response = requests.get(
    "https://api.overtime.io/overtime-v2/games-info", 
    headers={'accept': 'application/json'}
)

print(f"Status Code: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    
    # The response seems to be a dict with hex keys
    print(f"\nTotal keys in response: {len(data)}")
    
    # Get first few keys
    keys = list(data.keys())[:10]
    print(f"\nFirst 10 keys:")
    for key in keys:
        print(f"  {key}")
        
    # Check if these are game IDs by looking at the values
    print(f"\nChecking values for first key...")
    first_key = keys[0] if keys else None
    
    if first_key:
        value = data[first_key]
        print(f"Key: {first_key}")
        print(f"Value type: {type(value)}")
        print(f"Value: {value}")
        
        # Try to decode hex if it's a string
        if isinstance(first_key, str) and first_key.startswith('0x'):
            try:
                # Remove 0x prefix and decode
                hex_str = first_key[2:]
                decoded = bytes.fromhex(hex_str).decode('utf-8', errors='ignore')
                print(f"Decoded key: {decoded}")
            except:
                pass
    
    # Maybe the data structure has changed - let's save the full response
    with open('overtime_api_response.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nFull response saved to overtime_api_response.json")
    
    # Try alternative interpretation - maybe these are game IDs and we need to look them up differently
    print(f"\nThese appear to be game IDs. Total games: {len(data)}")