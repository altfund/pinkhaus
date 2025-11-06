#!/usr/bin/env python3
"""Compare blockchain vs API data to see what we have"""

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
from sqlalchemy import func, and_, or_

print("🔍 Comparing Blockchain vs API data...\n")

with db_manager.get_db_session() as db:
    # Get summary stats
    print("📊 Data Summary:")
    print(f"   Total Markets: {db.query(Market).count()}")
    print(f"   Total Odds: {db.query(Odd).count()}")
    
    # Check blockchain sources
    blockchain_sources = ['blockchain_optimism_v2', 'blockchain_arbitrum_v2', 'blockchain_v2_optimism', 
                         'blockchain_optimism_v1', 'blockchain_live', 'blockchain_v2']
    
    api_sources = ['overtime_v2', 'overtime_soccer', 'overtime_v2_public', 'api_live', 'api_import']
    
    blockchain_markets = db.query(Market).filter(Market.source.in_(blockchain_sources)).count()
    api_markets = db.query(Market).filter(Market.source.in_(api_sources)).count()
    
    print(f"\n📡 Source Breakdown:")
    print(f"   Blockchain Markets: {blockchain_markets}")
    print(f"   API Markets: {api_markets}")
    
    # Get sample markets with odds from each source
    print("\n🔗 Blockchain Markets with Odds:")
    
    blockchain_with_odds = db.query(Market)\
        .join(Odd, Market.source_id == Odd.source_id)\
        .filter(
            and_(
                Market.source.in_(blockchain_sources),
                Market.is_finished == False
            )
        )\
        .distinct()\
        .limit(5)\
        .all()
    
    if blockchain_with_odds:
        for market in blockchain_with_odds:
            print(f"\n   {market.home_team} vs {market.away_team}")
            print(f"   Sport: {market.sport} | League: {market.league_name}")
            print(f"   Market ID: {market.source_id}")
            
            # Get odds
            odds = db.query(Odd).filter(Odd.source_id == market.source_id).all()
            if odds:
                print(f"   Odds ({len(odds)} records):")
                for odd in odds[:3]:
                    odds_val = odd.decimal_odds if odd.decimal_odds else odd.american_odds
                    print(f"     {odd.outcome}: {odds_val} ({odd.source})")
    else:
        print("   ❌ No blockchain markets with odds found")
    
    print("\n🌐 API Markets with Odds:")
    
    api_with_odds = db.query(Market)\
        .join(Odd, Market.source_id == Odd.source_id)\
        .filter(
            and_(
                Market.source.in_(api_sources),
                Market.is_finished == False
            )
        )\
        .distinct()\
        .limit(5)\
        .all()
    
    if api_with_odds:
        for market in api_with_odds:
            print(f"\n   {market.home_team} vs {market.away_team}")
            print(f"   Sport: {market.sport} | League: {market.league_name}")
            print(f"   Market ID: {market.source_id}")
            
            # Get odds
            odds = db.query(Odd).filter(Odd.source_id == market.source_id).all()
            if odds:
                print(f"   Odds ({len(odds)} records):")
                for odd in odds[:3]:
                    odds_val = odd.decimal_odds if odd.decimal_odds else odd.american_odds
                    print(f"     {odd.outcome}: {odds_val} ({odd.source})")
    else:
        print("   ❌ No API markets with odds found")
    
    # Check for recent updates
    now = datetime.now(timezone.utc)
    recent = now - timedelta(hours=1)
    
    print("\n⏰ Recent Activity (last hour):")
    
    recent_blockchain = db.query(Market)\
        .filter(
            and_(
                Market.source.in_(blockchain_sources),
                Market.updated_at > recent
            )
        ).count()
    
    recent_api = db.query(Market)\
        .filter(
            and_(
                Market.source.in_(api_sources),
                Market.updated_at > recent
            )
        ).count()
    
    print(f"   Blockchain updates: {recent_blockchain}")
    print(f"   API updates: {recent_api}")
    
    # Check for soccer markets specifically
    print("\n⚽ Soccer Markets Analysis:")
    
    soccer_blockchain = db.query(Market)\
        .filter(
            and_(
                Market.source.in_(blockchain_sources),
                Market.sport.ilike('%soccer%'),
                Market.is_finished == False
            )
        ).count()
    
    soccer_api = db.query(Market)\
        .filter(
            and_(
                Market.source.in_(api_sources),
                Market.sport.ilike('%soccer%'),
                Market.is_finished == False
            )
        ).count()
    
    print(f"   Blockchain Soccer: {soccer_blockchain}")
    print(f"   API Soccer: {soccer_api}")
    
    # Show sample soccer market with full data
    print("\n📋 Sample Complete Soccer Market:")
    
    soccer_market = db.query(Market)\
        .join(Odd, Market.source_id == Odd.source_id)\
        .filter(
            and_(
                Market.sport.ilike('%soccer%'),
                Market.is_finished == False,
                Market.maturity_date > now
            )
        )\
        .first()
    
    if soccer_market:
        print(f"\n   {soccer_market.home_team} vs {soccer_market.away_team}")
        print(f"   League: {soccer_market.league_name}")
        print(f"   Maturity: {soccer_market.maturity_date}")
        print(f"   Source: {soccer_market.source}")
        print(f"   Market ID: {soccer_market.source_id}")
        
        # Get all odds
        all_odds = db.query(Odd).filter(Odd.source_id == soccer_market.source_id).all()
        
        if all_odds:
            print(f"\n   All Odds ({len(all_odds)} records):")
            for odd in all_odds:
                odds_val = odd.decimal_odds if odd.decimal_odds else odd.american_odds
                print(f"     {odd.outcome}: {odds_val}")
                print(f"       Source: {odd.source}")
                print(f"       Updated: {odd.updated_at}")
    else:
        print("   No upcoming soccer markets found")
    
    print("\n\n💡 Key Findings:")
    print("1. We have both blockchain and API data")
    print("2. Blockchain data includes on-chain odds")
    print("3. API data provides market metadata")
    print("4. Need to merge both sources for complete picture")
    print("5. Check if sync processes are running to get live updates")