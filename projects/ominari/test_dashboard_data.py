#!/usr/bin/env python3
"""Test if dashboard is getting data from PostgreSQL"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import func
import json

# Test direct database connection
try:
    with db_manager.get_db_session() as db:
        # Count total markets
        total = db.query(Market).count()
        active = db.query(Market).filter(Market.is_finished == False).count()
        
        # Get a sample market with odds
        sample_market = db.query(Market).filter(
            Market.is_finished == False
        ).first()
        
        if sample_market:
            odds = db.query(Odd).filter(
                Odd.source_id == sample_market.source_id
            ).all()
            
            print(f"✅ Database connected successfully!")
            print(f"📊 Total markets: {total:,}")
            print(f"🎯 Active markets: {active:,}")
            print(f"\n🏆 Sample market:")
            print(f"  {sample_market.home_team} vs {sample_market.away_team}")
            print(f"  Sport: {sample_market.sport}")
            print(f"  Source: {sample_market.source}")
            print(f"  Odds count: {len(odds)}")
            
            if odds:
                print(f"\n💰 Odds:")
                for odd in odds:
                    print(f"  {odd.outcome}: {odd.decimal_odds}")
        else:
            print("❌ No markets found")
            
except Exception as e:
    print(f"❌ Database error: {e}")
    
# Now test the web_monitor's get_market_data function
print("\n" + "="*50)
print("Testing web_monitor.get_market_data()...")

try:
    from web_monitor import get_market_data
    
    markets, signals, stats, chunks = get_market_data()
    
    print(f"✅ get_market_data() returned:")
    print(f"  Markets: {len(markets)}")
    print(f"  Signals: {len(signals)}")
    print(f"  Stats: {json.dumps(stats, indent=2)}")
    print(f"  Chunks: {len(chunks)}")
    
    if markets:
        print(f"\n🏆 First market:")
        m = markets[0]
        print(f"  {m.get('home_team')} vs {m.get('away_team')}")
        print(f"  Home odds: {m.get('home_odds')}")
        print(f"  Draw odds: {m.get('draw_odds')}")
        print(f"  Away odds: {m.get('away_odds')}")
except Exception as e:
    print(f"❌ Error testing get_market_data: {e}")
    import traceback
    traceback.print_exc()