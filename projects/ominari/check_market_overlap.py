#!/usr/bin/env python3
"""Check which markets exist in both blockchain and API sources"""

import os
from datetime import datetime, timezone, timedelta

# Set up database environment
os.environ['USE_POSTGRESQL'] = '1'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import and_, or_, func

print("🔍 Checking market overlap between blockchain and API...\n")

with db_manager.get_db_session() as db:
    # Define source groups
    blockchain_sources = ['blockchain_optimism_v2', 'blockchain_arbitrum_v2', 'blockchain_v2_optimism', 
                         'blockchain_optimism_v1', 'blockchain_live', 'blockchain_v2']
    api_sources = ['overtime_v2', 'overtime_soccer', 'overtime_v2_public', 'api_live', 'api_import']
    
    # Find markets that appear in both blockchain and API by matching team names and dates
    print("📊 Looking for markets in both blockchain and API...")
    
    # Get blockchain markets
    blockchain_markets = db.query(
        Market.home_team,
        Market.away_team,
        func.date(Market.maturity_date).label('match_date'),
        Market.source_id,
        Market.source
    ).filter(
        and_(
            Market.source.in_(blockchain_sources),
            Market.sport.ilike('%soccer%')
        )
    ).all()
    
    # Get API markets
    api_markets = db.query(
        Market.home_team,
        Market.away_team,
        func.date(Market.maturity_date).label('match_date'),
        Market.source_id,
        Market.source
    ).filter(
        and_(
            Market.source.in_(api_sources),
            Market.sport.ilike('%soccer%')
        )
    ).all()
    
    # Create lookup dictionaries
    blockchain_dict = {}
    for market in blockchain_markets:
        key = f"{market.home_team.lower().strip()}_{market.away_team.lower().strip()}_{market.match_date}"
        if key not in blockchain_dict:
            blockchain_dict[key] = []
        blockchain_dict[key].append({
            'source_id': market.source_id,
            'source': market.source
        })
    
    api_dict = {}
    for market in api_markets:
        key = f"{market.home_team.lower().strip()}_{market.away_team.lower().strip()}_{market.match_date}"
        if key not in api_dict:
            api_dict[key] = []
        api_dict[key].append({
            'source_id': market.source_id,
            'source': market.source
        })
    
    # Find overlaps
    overlaps = []
    for key in blockchain_dict:
        if key in api_dict:
            overlaps.append({
                'key': key,
                'blockchain': blockchain_dict[key],
                'api': api_dict[key]
            })
    
    print(f"\n✅ Found {len(overlaps)} markets that exist in both blockchain and API\n")
    
    if overlaps:
        print("Sample overlapping markets:")
        for i, overlap in enumerate(overlaps[:5]):
            parts = overlap['key'].split('_')
            home = parts[0]
            away = parts[1]
            date = parts[2]
            
            print(f"\n{i+1}. {home} vs {away} ({date})")
            print("   Blockchain sources:")
            for source in overlap['blockchain'][:2]:
                print(f"     - {source['source']}: {source['source_id'][:50]}...")
                
                # Get odds from blockchain
                odds = db.query(Odd).filter(
                    Odd.source_id == source['source_id']
                ).limit(3).all()
                if odds:
                    odds_str = ", ".join([f"{o.outcome}: {o.decimal_odds or o.american_odds}" for o in odds])
                    print(f"       Odds: {odds_str}")
            
            print("   API sources:")
            for source in overlap['api'][:2]:
                print(f"     - {source['source']}: {source['source_id'][:50]}...")
                
                # Get odds from API
                odds = db.query(Odd).filter(
                    Odd.source_id == source['source_id']
                ).limit(3).all()
                if odds:
                    odds_str = ", ".join([f"{o.outcome}: {o.decimal_odds or o.american_odds}" for o in odds])
                    print(f"       Odds: {odds_str}")
    else:
        print("❌ No overlapping markets found!")
        print("\nThis suggests that:")
        print("1. API and blockchain are using different market IDs")
        print("2. Team names might be formatted differently")
        print("3. The data might be from different time periods")
    
    # Check if we have any markets with blockchain addresses
    print("\n\n🔗 Checking for markets with blockchain addresses...")
    
    markets_with_addresses = db.query(Market).filter(
        and_(
            Market.sport.ilike('%soccer%'),
            or_(
                Market.source_id.like('0x%'),  # Ethereum-style addresses
                Market.metadata.like('%address%')  # Address in metadata
            )
        )
    ).limit(10).all()
    
    if markets_with_addresses:
        print(f"\nFound {len(markets_with_addresses)} markets with potential blockchain addresses:")
        for market in markets_with_addresses[:5]:
            print(f"\n   {market.home_team} vs {market.away_team}")
            print(f"   Source: {market.source}")
            print(f"   ID: {market.source_id[:60]}...")
            if market.metadata:
                print(f"   Metadata: {str(market.metadata)[:100]}...")