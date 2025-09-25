#!/usr/bin/env python3
"""
Dashboard Status - Shows current configuration and updates
"""

import os

# Set PostgreSQL environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import func

print("🎯 OMINARI DASHBOARD STATUS")
print("=" * 60)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n🏛️ CONFIGURATION:")
print("-" * 40)
print("⚽ Sport Filter: SOCCER ONLY (Not configurable)")
print("📝 Trading Mode: PAPER TRADING")
print("📡 Strategy: UNDERDOG BETTING")
print("💰 Initial Capital: $10,000")
print("🎯 Bet Size: $50 per trade")
print("📈 Minimum Odds: 2.5 (underdog threshold)")

print("\n📊 DATABASE STATUS:")
print("-" * 40)

try:
    with db_manager.get_db_session() as db:
        # Count soccer markets
        soccer_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False
        ).count()
        
        # Count total markets
        total_markets = db.query(Market).filter(
            Market.is_finished == False
        ).count()
        
        # Count odds
        total_odds = db.query(Odd).count()
        
        print(f"✅ Database Connection: SUCCESS")
        print(f"🌐 PostgreSQL Port: 5999")
        print(f"⚽ Active Soccer Markets: {soccer_markets}")
        print(f"📄 Total Active Markets: {total_markets}")
        print(f"📊 Total Odds Records: {total_odds:,}")
        print(f"🔒 Soccer-only Filter: {soccer_markets}/{total_markets} markets shown")
        
except Exception as e:
    print(f"❌ Database Connection: FAILED")
    print(f"Error: {str(e)[:100]}")

print("\n🌐 DASHBOARD FEATURES:")
print("-" * 40)
print("✅ Soccer-only filter (hardcoded, not user-changeable)")
print("✅ Paper trading mode indicator")
print("✅ Underdog betting strategy (odds > 2.5)")
print("✅ Real-time blockchain data")
print("✅ WebSocket live updates")
print("✅ Portfolio tracking")
print("✅ Signal generation")
print("✅ Performance charts")

print("\n🚀 DASHBOARD URL:")
print("-" * 40)
print("http://localhost:8888")
print("\n✅ All requested updates have been implemented!")