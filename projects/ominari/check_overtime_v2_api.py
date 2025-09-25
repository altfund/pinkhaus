#!/usr/bin/env python3
"""
Check Overtime V2 API endpoints more thoroughly
"""

import requests
import json

# Try V2 endpoints with network IDs
networks = {
    '10': 'Optimism',
    '42161': 'Arbitrum',
    '8453': 'Base'
}

print('🔍 CHECKING OVERTIME V2 API WITH NETWORK IDS')
print('=' * 60)

for network_id, network_name in networks.items():
    print(f'\n📌 Checking {network_name} (ID: {network_id})')
    
    endpoints = [
        f'https://api.overtime.io/overtime-v2/networks/{network_id}/markets',
        f'https://api.overtime.io/overtime-v2/networks/{network_id}/games',
        f'https://api.overtime.io/overtime-v2/networks/{network_id}/sports',
        f'https://api.thalesmarket.io/overtime-v2/networks/{network_id}/markets',
        f'https://api.thalesmarket.io/overtime-v2/networks/{network_id}/sports-markets',
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f'✅ {endpoint}')
                
                # Check structure
                if isinstance(data, list) and len(data) > 0:
                    print(f'   Found {len(data)} items')
                    # Check first item for sport fields
                    first = data[0]
                    if isinstance(first, dict):
                        sport_fields = [k for k in first.keys() if 'sport' in k.lower()]
                        if sport_fields:
                            print(f'   🎯 Sport fields: {sport_fields}')
                            for field in sport_fields:
                                print(f'      {field}: {first[field]}')
                        
                        # Show all fields of first item
                        print(f'   All fields: {list(first.keys())[:15]}')
                        
                elif isinstance(data, dict):
                    print(f'   Dict with {len(data)} keys')
            else:
                print(f'❌ {endpoint} - Status: {response.status_code}')
        except Exception as e:
            print(f'❌ {endpoint} - Error: {str(e)[:50]}')

# Try different base URLs
print('\n\n🌐 TRYING DIFFERENT BASE URLS:')
base_urls = [
    'https://api.thalesmarket.io/overtime-v2',
    'https://api.thalesmarket.io/overtime',
    'https://api.overtime.io/v2',
    'https://api.overtime.xyz',
]

for base in base_urls:
    print(f'\n📌 Base: {base}')
    for endpoint in ['/markets', '/games', '/sports-markets']:
        try:
            url = base + endpoint
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                print(f'✅ {endpoint} works!')
                data = response.json()
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    sport_fields = [k for k in data[0].keys() if 'sport' in k.lower()]
                    if sport_fields:
                        print(f'   🎯 Has sport fields: {sport_fields}')
        except:
            pass