#!/usr/bin/env python3
"""Test web_monitor's actual database connection"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import text

# Test if web_monitor is really using PostgreSQL
try:
    print("🔍 Testing web_monitor's database connection...")
    print(f"📊 Database URL: {db_manager.db_url}")
    
    with db_manager.get_db_session() as db:
        # Check if we're really connected to PostgreSQL
        result = db.execute(text("SELECT version()"))
        version = result.scalar()
        print(f"\n✅ Connected to: {version}")
        
        # Count active markets
        active = db.query(Market).filter(Market.is_finished == False).count()
        print(f"\n🎯 Active markets: {active:,}")
        
        # Test the exact query web_monitor uses
        soccer_markets = db.query(Market).filter(
            Market.is_finished == False,
            Market.sport == 'Soccer'
        ).order_by(Market.maturity_date).limit(50).all()
        
        other_markets = db.query(Market).filter(
            Market.is_finished == False,
            Market.sport != 'Soccer'  
        ).order_by(Market.maturity_date).limit(50).all()
        
        print(f"\n⚽ Soccer markets: {len(soccer_markets)}")
        print(f"🏈 Other markets: {len(other_markets)}")
        print(f"📊 Total returned by get_market_data query: {len(soccer_markets) + len(other_markets)}")
        
        if soccer_markets:
            print(f"\n🏆 First soccer market:")
            m = soccer_markets[0]
            print(f"  {m.home_team} vs {m.away_team}")
            print(f"  Sport: {m.sport}")
            print(f"  Source: {m.source}")
            
            # Check odds
            odds = db.query(Odd).filter(
                Odd.source_id == m.source_id
            ).order_by(Odd.updated_at.desc()).limit(3).all()
            print(f"  Odds count: {len(odds)}")
            
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()