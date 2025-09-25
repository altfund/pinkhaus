#!/usr/bin/env python3
"""
Check the markets endpoint that returned 401 (unauthorized)
Maybe it needs an API key or different format
"""

import requests
import json

print('🔍 CHECKING MARKETS ENDPOINTS')
print('=' * 60)

# The endpoint that returned 401
markets_endpoints = [
    'https://api.overtime.io/overtime-v2/networks/10/markets',
    'https://api.overtime.io/overtime-v2/networks/42161/markets',
    'https://api.overtime.io/overtime-v2/markets',
    'https://api.overtime.io/markets',
]

headers_variations = [
    {},  # No headers
    {'Accept': 'application/json'},
    {'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'},
]

for endpoint in markets_endpoints:
    print(f'\n📌 Testing: {endpoint}')
    
    for i, headers in enumerate(headers_variations):
        try:
            response = requests.get(endpoint, headers=headers, timeout=5)
            print(f'  Attempt {i+1}: Status {response.status_code}')
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    print(f'  ✅ Found {len(data)} markets!')
                    # Check first market
                    first = data[0]
                    if isinstance(first, dict):
                        print(f'  Fields: {list(first.keys())}')
                        # Look for sport fields
                        sport_fields = [k for k in first.keys() if 'sport' in k.lower()]
                        if sport_fields:
                            print(f'  🎯 Sport fields found: {sport_fields}')
                            for field in sport_fields:
                                print(f'     {field}: {first[field]}')
                elif isinstance(data, dict):
                    print(f'  Response is dict with keys: {list(data.keys())[:10]}')
            elif response.status_code == 401:
                print(f'  ❌ Unauthorized - may need API key')
                # Check response for hints
                try:
                    error_data = response.json()
                    print(f'  Error: {error_data}')
                except:
                    print(f'  Raw error: {response.text[:200]}')
        except Exception as e:
            print(f'  Error: {str(e)[:100]}')

# Check GraphQL endpoint
print('\n\n📊 CHECKING GRAPHQL ENDPOINT:')
graphql_url = 'https://api.thalesmarket.io/graphql'
query = '''
{
  sportMarkets(first: 5) {
    id
    address
    gameId
    sportId
    homeTeam
    awayTeam
    tags
  }
}
'''

try:
    response = requests.post(
        graphql_url,
        json={'query': query},
        headers={'Content-Type': 'application/json'}
    )
    print(f'GraphQL status: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2))
except Exception as e:
    print(f'GraphQL error: {e}')