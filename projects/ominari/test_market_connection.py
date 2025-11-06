#!/usr/bin/env python3
"""Test if we can connect blockchain and API markets"""

import os
import asyncio

# Set up database environment
os.environ['USE_POSTGRESQL'] = '1'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from unified_data_fetcher import UnifiedDataFetcher

async def main():
    fetcher = UnifiedDataFetcher()
    
    print("🔗 Testing market connections...\n")
    
    # Fetch and merge
    markets = await fetcher.fetch_all_markets()
    
    # Count by data source
    blockchain_only = 0
    api_only = 0
    both = 0
    
    for market in markets:
        sources = market.get('data_sources', [])
        if len(sources) > 1:
            both += 1
        elif 'blockchain' in sources:
            blockchain_only += 1
        elif 'api' in sources:
            api_only += 1
    
    print(f"📊 Market Source Analysis:")
    print(f"   Blockchain only: {blockchain_only}")
    print(f"   API only: {api_only}")
    print(f"   Both sources: {both}")
    
    if both > 0:
        print(f"\n✅ Successfully connected {both} markets!")
        print("\n📋 Sample connected markets:")
        
        connected = [m for m in markets if len(m.get('data_sources', [])) > 1]
        for market in connected[:5]:
            print(f"\n   {market['home_team']} vs {market['away_team']}")
            print(f"   Sources: {', '.join(market['data_sources'])}")
            print(f"   Blockchain ID: {market.get('blockchain_id', 'N/A')}")
            print(f"   API ID: {market.get('api_id', 'N/A')}")
            print(f"   Odds: {market['odds']}")
    else:
        print("\n❌ No markets successfully connected")
        print("\nDEBUG: Checking ID normalization...")
        
        # Test ID normalization
        test_ids = [
            'overtime_real_0x3230323531313033444441354644463000000000000000000000000000000000',
            'blockchain_v2_0x3230323531313033444441354644463000000000000000000000000000000000',
            'v2_0x3230323531313033444441354644463000000000000000000000000000000000'
        ]
        
        print("\nTest ID normalization:")
        for test_id in test_ids:
            normalized = fetcher._normalize_market_id(test_id)
            print(f"   {test_id[:50]}... -> {normalized[:50]}...")

if __name__ == "__main__":
    asyncio.run(main())