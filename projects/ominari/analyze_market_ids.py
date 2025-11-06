#!/usr/bin/env python3
"""Analyze market ID formats to understand the disconnect"""

import os
from collections import Counter

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

print("🔍 Analyzing market ID formats...\n")

with db_manager.get_db_session() as db:
    # Get sample market IDs from each source
    sources = db.query(Market.source).distinct().all()
    
    for source in sources:
        source_name = source[0]
        
        # Get sample markets
        markets = db.query(Market.source_id).filter(
            Market.source == source_name
        ).limit(5).all()
        
        if markets:
            print(f"\n📊 Source: {source_name}")
            print(f"   Sample IDs:")
            
            for market in markets:
                id_val = market[0]
                print(f"     - {id_val[:80]}...")
                
                # Analyze ID format
                if id_val.startswith('0x'):
                    print(f"       Format: Hex address ({len(id_val)} chars)")
                elif id_val.startswith('overtime_real_'):
                    print(f"       Format: Overtime real prefix")
                elif '_' in id_val:
                    print(f"       Format: Underscore separated")
                else:
                    print(f"       Format: Other")
    
    print("\n\n🔗 Key Observations:")
    
    # Check overtime_v2 specifically
    overtime_v2_markets = db.query(Market).filter(
        and_(
            Market.source == 'overtime_v2',
            Market.sport.ilike('%soccer%')
        )
    ).limit(3).all()
    
    print("\n1. Overtime V2 Market IDs:")
    for market in overtime_v2_markets:
        print(f"   {market.home_team} vs {market.away_team}")
        print(f"   ID: {market.source_id}")
        print(f"   Expected blockchain format: 0x{market.source_id.replace('overtime_real_0x', '').replace('0' * 48, '')}")
        print()
    
    print("\n2. The Issue:")
    print("   - API markets use 'overtime_real_0x...' format")
    print("   - Blockchain markets should use '0x...' format")
    print("   - These are likely the SAME markets with different ID formats")
    print("   - Need to strip prefix and padding to match them")
    
    print("\n3. Solution:")
    print("   - Extract the actual hex ID from API format")
    print("   - Match it with blockchain contract addresses")
    print("   - This will connect API odds data with blockchain transaction capability")