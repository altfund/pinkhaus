#!/usr/bin/env python3
"""Test finding connected markets between blockchain and API"""

import os
import asyncio

# Set up database environment
os.environ['USE_POSTGRESQL'] = '1'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from database_v2 import db_manager
from models import Market
from sqlalchemy import and_
from market_id_mapper import MarketIDMapper

print("🔍 Finding connected markets in database...\n")

# First, let's analyze what IDs we have
mapper = MarketIDMapper()

with db_manager.get_db_session() as db:
    # Get all soccer markets with their IDs
    all_markets = db.query(
        Market.source_id,
        Market.source,
        Market.home_team,
        Market.away_team,
        Market.maturity_date
    ).filter(
        Market.sport.ilike('%soccer%')
    ).all()
    
    print(f"Analyzing {len(all_markets)} soccer markets...")
    
    # Extract and group IDs
    market_ids = [(m.source_id, m.source) for m in all_markets]
    
    # Find groups of related IDs
    all_ids = [m[0] for m in market_ids]
    groups = mapper.group_related_ids(all_ids)
    
    print(f"\n✅ Found {len(groups)} groups of connected markets!\n")
    
    if groups:
        print("Sample connected market groups:")
        for i, group in enumerate(groups[:5]):
            print(f"\nGroup {i+1} ({len(group)} IDs):")
            
            # Find the markets for this group
            for market_id in group[:3]:
                market = next((m for m in all_markets if m.source_id == market_id), None)
                if market:
                    print(f"  - {market.source}: {market.home_team} vs {market.away_team}")
                    print(f"    ID: {market_id[:60]}...")
                    core_id = mapper.extract_core_id(market_id)
                    print(f"    Core: {core_id}")
    
    # Look for specific pattern matches
    print("\n\n🔗 Checking for blockchain/API pairs...")
    
    blockchain_markets = [(m.source_id, m) for m in all_markets if 'blockchain' in m.source or m.source.startswith('v2_')]
    api_markets = [(m.source_id, m) for m in all_markets if 'overtime' in m.source or 'api' in m.source]
    
    pairs_found = 0
    for bc_id, bc_market in blockchain_markets[:50]:
        bc_core = mapper.extract_core_id(bc_id)
        
        for api_id, api_market in api_markets:
            api_core = mapper.extract_core_id(api_id)
            
            if mapper.match_markets(bc_id, api_id):
                pairs_found += 1
                print(f"\n🎯 Found matching pair!")
                print(f"   Blockchain: {bc_market.source} - {bc_id[:50]}...")
                print(f"   API: {api_market.source} - {api_id[:50]}...")
                print(f"   Core ID: {bc_core}")
                print(f"   Match: {bc_market.home_team} vs {bc_market.away_team}")
                
                if pairs_found >= 5:
                    break
        
        if pairs_found >= 5:
            break
    
    if pairs_found == 0:
        print("\n❌ No blockchain/API pairs found with current matching logic")
        print("\nThis suggests the IDs are completely different between sources")
        print("We may need to match by team names + date instead")