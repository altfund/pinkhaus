#!/usr/bin/env python3
"""Debug WebSocket data emission"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from web_monitor import get_market_data
import json

print("🔍 Testing get_market_data() function...")

try:
    markets, signals, stats, chunks = get_market_data()
    
    print(f"\n📊 Function returned:")
    print(f"  Markets: {len(markets)}")
    print(f"  Signals: {len(signals)}")
    print(f"  Stats: {json.dumps(stats, indent=2)}")
    print(f"  Chunks: {len(chunks)}")
    
    if markets:
        print(f"\n🏆 First 3 markets:")
        for i, m in enumerate(markets[:3]):
            print(f"\n  Market {i+1}:")
            print(f"    Teams: {m.get('home_team')} vs {m.get('away_team')}")
            print(f"    Sport: {m.get('sport')}")
            print(f"    Time until: {m.get('time_until')}")
            print(f"    Odds: Home={m.get('home_odds')}, Draw={m.get('draw_odds')}, Away={m.get('away_odds')}")
            print(f"    Status: {m.get('status')} (color: {m.get('status_color')})")
            print(f"    Has odds: {'Yes' if m.get('home_odds') and m.get('away_odds') else 'No'}")
            
        print(f"\n📦 WebSocket payload size: {len(json.dumps({'markets': markets, 'signals': signals, 'stats': stats, 'chunks': chunks}))} bytes")
    else:
        print("\n❌ No markets returned!")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()