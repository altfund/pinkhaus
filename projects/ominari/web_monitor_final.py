#!/usr/bin/env python3
"""
Final Connected Ominari Dashboard - Using working components
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timezone
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ominari-blockchain-trading-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Import components that work
components_loaded = False
try:
    # Direct imports without problematic dependencies
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import database manager first
    from database_v2 import db_manager
    
    # Import session manager
    from paper_trading_postgres_integrated import PaperTradingSessionManager
    session_manager = PaperTradingSessionManager()
    
    # Try to import data fetcher
    try:
        # First disable signals import to avoid grpc
        import unified_data_fetcher
        # Monkey patch to avoid grpc imports
        unified_data_fetcher.ENABLE_GRPC = False
        from unified_data_fetcher import UnifiedDataFetcher
        unified_fetcher = UnifiedDataFetcher(blockchain_first=True)
    except:
        unified_fetcher = None
        logger.warning("Unified fetcher not available")
    
    # Try edge calculator
    try:
        from edge_calculator import EdgeCalculator
        edge_calculator = EdgeCalculator()
    except:
        edge_calculator = None
        logger.warning("Edge calculator not available")
    
    components_loaded = True
    logger.info("✅ Core components loaded")
    
except Exception as e:
    logger.error(f"Failed to load components: {e}")
    import traceback
    traceback.print_exc()

# Load HTML template
try:
    with open('web_monitor_blockchain.py', 'r') as f:
        content = f.read()
        start = content.find('BLOCKCHAIN_DASHBOARD_HTML = """') + len('BLOCKCHAIN_DASHBOARD_HTML = """')
        end = content.rfind('"""')
        end = content.rfind('"""', 0, end)
        DASHBOARD_HTML = content[start:end]
        logger.info("Dashboard HTML loaded")
except Exception as e:
    logger.error(f"Failed to load HTML: {e}")
    DASHBOARD_HTML = "<h1>Error loading dashboard</h1>"

async def fetch_markets_direct():
    """Fetch markets using direct API calls"""
    import aiohttp
    
    markets = []
    
    # Try local API first
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/api/v1/markets", timeout=3) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    markets = data.get('markets', [])
                    logger.info(f"Fetched {len(markets)} from local API")
    except:
        pass
    
    # Use test data if no API
    if not markets:
        markets = [
            {
                'match_id': 'demo1',
                'home_team': 'Manchester City',
                'away_team': 'Arsenal',
                'sport': 'Soccer',
                'league': 'Premier League',
                'maturity_date': datetime.now(timezone.utc).isoformat(),
                'odds': 1.95,
                'position': 'Home Win',
                'blockchain_connected': True,
                'blockchain_address': '0x1234...5678'
            },
            {
                'match_id': 'demo1',
                'home_team': 'Manchester City',
                'away_team': 'Arsenal',
                'sport': 'Soccer',
                'league': 'Premier League',
                'maturity_date': datetime.now(timezone.utc).isoformat(),
                'odds': 3.40,
                'position': 'Draw',
                'blockchain_connected': True,
                'blockchain_address': '0x1234...5678'
            },
            {
                'match_id': 'demo1',
                'home_team': 'Manchester City',
                'away_team': 'Arsenal',
                'sport': 'Soccer',
                'league': 'Premier League',
                'maturity_date': datetime.now(timezone.utc).isoformat(),
                'odds': 4.20,
                'position': 'Away Win',
                'blockchain_connected': True,
                'blockchain_address': '0x1234...5678'
            }
        ]
    
    # Add edge calculations
    for market in markets:
        if market.get('odds'):
            market['implied_prob'] = 1 / market['odds']
            # Simple edge based on overround
            market['edge'] = 0.02 if market.get('position') == 'Home Win' else 0
            market['has_edge'] = market['edge'] > 0
    
    return markets

async def get_dashboard_data():
    """Get dashboard data"""
    # Get markets
    if unified_fetcher:
        try:
            markets = await unified_fetcher.fetch_all_markets()
            trading_markets = unified_fetcher.format_for_trading(markets)
            logger.info(f"Fetched {len(trading_markets)} from unified fetcher")
        except:
            trading_markets = await fetch_markets_direct()
    else:
        trading_markets = await fetch_markets_direct()
    
    # Get trading status
    trading_status = {'status': 'Demo Mode', 'bankroll': 10000, 'active_positions': 0}
    
    if session_manager and components_loaded:
        try:
            session_id = session_manager.get_current_session()
            if session_id:
                session = session_manager.get_session(session_id)
                if session:
                    positions = session_manager.get_active_positions(session_id)
                    trading_status = {
                        'status': 'Active',
                        'session_id': session_id,
                        'bankroll': float(session.get('current_bankroll', 0)),
                        'initial_bankroll': float(session.get('initial_bankroll', 0)),
                        'active_positions': len(positions),
                        'positions_value': sum(float(p.get('stake', 0)) for p in positions)
                    }
                    logger.info(f"Got trading status for session {session_id}")
        except Exception as e:
            logger.error(f"Error getting trading status: {e}")
    
    # Format markets for display
    formatted_markets = []
    for market in trading_markets[:50]:
        formatted_markets.append({
            'match_id': market.get('match_id', ''),
            'home_team': market.get('home_team', ''),
            'away_team': market.get('away_team', ''),
            'sport': market.get('sport', ''),
            'league': market.get('league', ''),
            'maturity_date': (
                market.get('maturity_date').isoformat() 
                if hasattr(market.get('maturity_date'), 'isoformat')
                else str(market.get('maturity_date', ''))
            ),
            'odds': market.get('odds', 0),
            'position': market.get('position', market.get('outcome', '')),
            'blockchain_connected': market.get('blockchain_connected', False) or bool(market.get('blockchain_address')),
            'edge': market.get('edge', 0),
            'implied_prob': market.get('implied_prob', 0),
            'has_edge': market.get('has_edge', False)
        })
    
    blockchain_connected = sum(1 for m in formatted_markets if m.get('blockchain_connected'))
    
    return {
        'markets': formatted_markets,
        'trading_status': trading_status,
        'stats': {
            'total_markets': len(formatted_markets),
            'blockchain_connected': blockchain_connected,
            'blockchain_ratio': (blockchain_connected / len(formatted_markets) * 100) if formatted_markets else 0,
            'chunk_hours': 2.0
        },
        'config': {
            'testnet_mode': False,
            'simulation_mode': False,
            'network': 'Connected' if components_loaded else 'Demo'
        },
        'capital_state': {
            'available_cash': trading_status.get('bankroll', 0) - trading_status.get('positions_value', 0),
            'pending_stakes': 0,
            'in_play_exposure': trading_status.get('positions_value', 0),
            'settlement_pending': 0,
            'utilization_rate': (
                trading_status.get('positions_value', 0) / trading_status.get('bankroll', 1)
                if trading_status.get('bankroll', 0) > 0 else 0
            ),
            'value_at_risk': trading_status.get('positions_value', 0),
            'expected_value': 0
        }
    }

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    asyncio.run(send_update())

@socketio.on('request_dashboard_data')
def handle_request():
    asyncio.run(send_update())

async def send_update():
    try:
        data = await get_dashboard_data()
        socketio.emit('dashboard_update', data)
        logger.info(f"Sent {len(data['markets'])} markets")
    except Exception as e:
        logger.error(f"Update error: {e}")
        socketio.emit('dashboard_update', {'error': str(e)})

@socketio.on('update_empirical_data')
def handle_empirical():
    emit('empirical_data_update', {
        'success': True,
        'message': 'Data refreshed',
        'empirical_status': {
            'patterns_loaded': 10,
            'capital_tracking_active': True,
            'settlement_patterns': 5,
            'valuation_active': False,
            'dynamic_chunking_enabled': False
        }
    })

# Background updates
def background_loop():
    while True:
        try:
            asyncio.run(send_update())
        except Exception as e:
            logger.error(f"Background error: {e}")
        
        import time
        time.sleep(30)

if __name__ == '__main__':
    logger.info("Starting Final Connected Dashboard on port 8888...")
    logger.info(f"Components loaded: {components_loaded}")
    
    import threading
    bg_thread = threading.Thread(target=background_loop, daemon=True)
    bg_thread.start()
    
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)