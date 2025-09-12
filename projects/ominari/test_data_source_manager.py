#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the new data source manager
"""

import asyncio
import logging
from datetime import datetime, timedelta
from data_source_manager import DataSourceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_data_sources():
    """Test all data sources and their fallback behavior."""
    
    print("\n" + "="*60)
    print("Testing Ominari Data Source Manager")
    print("="*60 + "\n")
    
    # Initialize manager
    manager = DataSourceManager()
    
    try:
        print("1. Initializing data sources...")
        await manager.initialize(network='optimism')
        
        # Get status
        print("\n2. Data source status:")
        status = await manager.get_status()
        print(f"   Initialized: {status['initialized']}")
        print("   Available sources:")
        for source in status['sources']:
            availability = '✅' if source['available'] else '❌'
            print(f"     - {source['name']:<30} (Priority: {source['priority']:<15}) {availability}")
        
        # Test market fetching
        print("\n3. Testing market data fetching...")
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now() + timedelta(days=7)
        
        try:
            markets = await manager.get_markets(
                start_date=start_date,
                end_date=end_date
            )
            print(f"   ✅ Successfully fetched {len(markets)} markets")
            
            if markets:
                # Show sample market
                sample = markets[0]
                print("\n   Sample market:")
                print(f"     Source: {sample.get('source')}")
                print(f"     Sport: {sample.get('sport')}")
                print(f"     Teams: {sample.get('home_team')} vs {sample.get('away_team')}")
                print(f"     Date: {sample.get('maturity_date')}")
                
        except Exception as e:
            print(f"   ❌ Error fetching markets: {e}")
        
        # Test specific sport
        print("\n4. Testing sport-specific query...")
        try:
            football_markets = await manager.get_markets(sport='football')
            print(f"   ✅ Found {len(football_markets)} football markets")
        except Exception as e:
            print(f"   ❌ Error fetching football markets: {e}")
        
        # Test API fallback
        print("\n5. Testing API fallback (forcing backup)...")
        try:
            api_markets = await manager.get_markets(use_backup=True)
            print(f"   ✅ API fallback returned {len(api_markets)} markets")
        except Exception as e:
            print(f"   ❌ API fallback failed: {e}")
        
        # Test odds fetching
        if markets and len(markets) > 0:
            print("\n6. Testing odds fetching...")
            market_ids = [m.get('source_id') for m in markets[:5] if m.get('source_id')]
            try:
                odds = await manager.get_odds(market_ids)
                print(f"   ✅ Fetched {len(odds)} odds records")
            except Exception as e:
                print(f"   ❌ Error fetching odds: {e}")
        
        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Fatal error during testing: {e}")
        import traceback
        traceback.print_exc()


async def test_graphql_connection():
    """Test GraphQL endpoints directly."""
    print("\n" + "="*60)
    print("Testing GraphQL Endpoints")
    print("="*60 + "\n")
    
    from graphql_client import OvertimeGraphQLClient
    
    networks = ['optimism', 'arbitrum']
    
    for network in networks:
        print(f"\nTesting {network.upper()} network:")
        
        # Test local endpoint
        try:
            local_client = OvertimeGraphQLClient(network=network, use_local=True)
            if await local_client.check_connection():
                print("  ✅ Local GraphQL endpoint accessible")
            else:
                print("  ❌ Local GraphQL endpoint not available")
        except Exception as e:
            print(f"  ❌ Local GraphQL error: {e}")
        
        # Test public endpoint
        try:
            public_client = OvertimeGraphQLClient(network=network, use_local=False)
            if await public_client.check_connection():
                print("  ✅ Public GraphQL endpoint accessible")
            else:
                print("  ❌ Public GraphQL endpoint not available")
        except Exception as e:
            print(f"  ❌ Public GraphQL error: {e}")


async def main():
    """Run all tests."""
    await test_graphql_connection()
    await test_data_sources()


if __name__ == "__main__":
    asyncio.run(main())