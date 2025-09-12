#!/usr/bin/env python3
"""Test GraphQL connection."""

import asyncio
from graphql_client import OvertimeGraphQLClient

async def test_graphql():
    """Test GraphQL connections."""
    print("Testing GraphQL connections...")
    
    # Test public GraphQL
    print("\n1. Testing Public GraphQL:")
    try:
        client = OvertimeGraphQLClient(network='optimism', use_local=False)
        markets = await client.get_markets(first=5)
        print(f"✅ Public GraphQL working! Found {len(markets)} markets")
        if markets:
            print(f"   Sample market: {markets[0].get('gameId', 'N/A')}")
    except Exception as e:
        print(f"❌ Public GraphQL failed: {e}")
    
    # Test local GraphQL
    print("\n2. Testing Local GraphQL:")
    try:
        client = OvertimeGraphQLClient(network='optimism', use_local=True)
        markets = await client.get_markets(first=5)
        print(f"✅ Local GraphQL working! Found {len(markets)} markets")
    except Exception as e:
        print(f"❌ Local GraphQL failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_graphql())