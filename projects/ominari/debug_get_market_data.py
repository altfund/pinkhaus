#!/usr/bin/env python3
"""Direct test of get_market_data function"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

# First test database connection
from database_v2 import db_manager
from models import Market

try:
    with db_manager.get_db_session() as db:
        count = db.query(Market).filter(Market.is_finished == False).count()
        print(f"📈 Active markets in database: {count}")
except Exception as e:
    print(f"❌ Database error: {e}")

# Now test get_market_data
import sys
sys.path.insert(0, '.')  # Make sure we import from current directory

# Mock the imports that require flask
import importlib
import types

# Create mock modules
flask_socketio = types.ModuleType('flask_socketio')
flask_socketio.SocketIO = lambda *args, **kwargs: None
flask_socketio.emit = lambda *args, **kwargs: None
sys.modules['flask_socketio'] = flask_socketio

flask = types.ModuleType('flask')
flask.Flask = lambda *args, **kwargs: None
flask.render_template_string = lambda *args, **kwargs: None
flask.jsonify = lambda x: x
flask.request = type('obj', (object,), {'args': {}})
sys.modules['flask'] = flask

# Mock other dependencies
for mod in ['evaluate_open_markets', 'simple_daily_change', 'calculate_max_drawdown', 'paper_trading_models_v2', 'simple_paper_trading']:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

# Now import the actual function
from web_monitor import get_market_data

print("\n🔍 Testing get_market_data()...")
try:
    markets, signals, stats, chunks = get_market_data()
    print(f"📊 Returned: {len(markets)} markets")
    print(f"📊 Stats: {stats}")
    
    if markets:
        print(f"\n🏆 First market:")
        m = markets[0]
        print(f"  Teams: {m.get('home_team')} vs {m.get('away_team')}")
        print(f"  Odds: H={m.get('home_odds')} D={m.get('draw_odds')} A={m.get('away_odds')}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()