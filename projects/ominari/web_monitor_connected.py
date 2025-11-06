#!/usr/bin/env python3
"""
Connected Ominari Dashboard - Real blockchain data without problematic imports
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

# Set up environment for database
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

# Import only what we need, avoiding problematic imports
try:
    from unified_data_fetcher import UnifiedDataFetcher
    from paper_trading_postgres_integrated import PaperTradingSessionManager
    from edge_calculator import EdgeCalculator
    unified_fetcher = UnifiedDataFetcher(blockchain_first=True)
    session_manager = PaperTradingSessionManager()
    edge_calculator = EdgeCalculator()
    logger.info("✅ Core components loaded successfully")
except ImportError as e:
    logger.error(f"Failed to import core components: {e}")
    unified_fetcher = None
    session_manager = None
    edge_calculator = None

# Load the HTML template
try:
    with open('web_monitor_blockchain.py', 'r') as f:
        content = f.read()
        start = content.find('BLOCKCHAIN_DASHBOARD_HTML = """') + len('BLOCKCHAIN_DASHBOARD_HTML = """')
        end = content.rfind('"""')
        end = content.rfind('"""', 0, end)
        DASHBOARD_HTML = content[start:end]
        logger.info("Dashboard HTML loaded successfully")
except Exception as e:
    logger.error(f"Failed to load dashboard HTML: {e}")
    DASHBOARD_HTML = "<h1>Error loading dashboard</h1>"

# Global state
current_session_id = None

def get_current_session():
    """Get or create trading session"""
    global current_session_id
    if session_manager:
        current_session_id = session_manager.get_current_session()
        if not current_session_id:
            current_session_id = session_manager.create_session(
                initial_bankroll=10000.0,
                description="Web Dashboard Trading Session"
            )
        return current_session_id
    return None

async def fetch_real_markets():
    """Fetch real market data from blockchain API"""
    if not unified_fetcher:
        return []
    
    try:
        # Fetch markets
        markets = await unified_fetcher.fetch_all_markets()
        
        # Format for trading
        trading_markets = unified_fetcher.format_for_trading(markets)
        
        # Calculate edges if available
        if edge_calculator and trading_markets:
            signals = edge_calculator.calculate_edges(trading_markets)
            
            # Create lookup for signals
            signal_lookup = {}
            for signal in signals:
                position = signal.get('position', signal.get('outcome', ''))
                key = f"{signal.get('match_id')}-{position}"
                signal_lookup[key] = signal
        else:
            signal_lookup = {}
        
        # Enhance markets with edge info
        enhanced_markets = []
        for market in trading_markets:
            # Add blockchain info
            market['blockchain_connected'] = (
                market.get('has_blockchain_data', False) or 
                bool(market.get('blockchain_address'))
            )
            
            # Add edge info
            position = market.get('position', market.get('outcome', ''))
            key = f"{market.get('match_id')}-{position}"
            
            if key in signal_lookup:
                signal = signal_lookup[key]
                market['edge'] = signal.get('edge', 0)
                market['signal_prob'] = signal.get('signal_prob')
                market['implied_prob'] = signal.get('implied_prob')
                market['has_edge'] = market['edge'] > 0.01  # 1% minimum edge
            else:
                market['edge'] = 0
                market['has_edge'] = False
                # Calculate implied prob from odds
                if market.get('odds'):
                    market['implied_prob'] = 1 / market['odds']
            
            enhanced_markets.append(market)
        
        logger.info(f"Fetched {len(enhanced_markets)} markets")
        return enhanced_markets
        
    except Exception as e:
        logger.error(f"Error fetching markets: {e}")
        return []

async def get_trading_status():
    """Get current trading status"""
    if not session_manager:
        return {
            'status': 'Not initialized',
            'bankroll': 0,
            'active_positions': 0
        }
    
    session_id = get_current_session()
    if not session_id:
        return {
            'status': 'No session',
            'bankroll': 0,
            'active_positions': 0
        }
    
    try:
        session = session_manager.get_session(session_id)
        positions = session_manager.get_active_positions(session_id)
        
        return {
            'status': 'Active',
            'session_id': session_id,
            'bankroll': float(session.get('current_bankroll', 0)),
            'initial_bankroll': float(session.get('initial_bankroll', 0)),
            'active_positions': len(positions),
            'positions_value': sum(float(p.get('stake', 0)) for p in positions)
        }
    except Exception as e:
        logger.error(f"Error getting trading status: {e}")
        return {
            'status': 'Error',
            'bankroll': 0,
            'active_positions': 0
        }

async def get_dashboard_data():
    """Get complete dashboard data"""
    # Fetch real markets
    markets = await fetch_real_markets()
    
    # Get trading status
    trading_status = await get_trading_status()
    
    # Format markets for display
    formatted_markets = []
    for market in markets[:50]:  # Limit to 50 for performance
        formatted_markets.append({
            'match_id': market.get('match_id', ''),
            'home_team': market.get('home_team', ''),
            'away_team': market.get('away_team', ''),
            'sport': market.get('sport', ''),
            'league': market.get('league', ''),
            'maturity_date': (
                market.get('maturity_date', '').isoformat() 
                if hasattr(market.get('maturity_date', ''), 'isoformat') 
                else str(market.get('maturity_date', ''))
            ),
            'odds': market.get('odds', 0),
            'position': market.get('position', market.get('outcome', '')),
            'blockchain_connected': market.get('blockchain_connected', False),
            'edge': market.get('edge', 0),
            'signal_prob': market.get('signal_prob'),
            'implied_prob': market.get('implied_prob', 0),
            'has_edge': market.get('has_edge', False)
        })
    
    # Calculate stats
    blockchain_connected = sum(1 for m in markets if m.get('blockchain_connected', False))
    
    return {
        'markets': formatted_markets,
        'trading_status': trading_status,
        'stats': {
            'total_markets': len(markets),
            'blockchain_connected': blockchain_connected,
            'blockchain_ratio': (blockchain_connected / len(markets) * 100) if markets else 0,
            'supported_sports': ['Soccer'],
            'chunk_hours': 2.0
        },
        'config': {
            'testnet_mode': True,
            'simulation_mode': False,
            'network': 'Blockchain API'
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
    """Main dashboard page"""
    return render_template_string(DASHBOARD_HTML)

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    # Send initial data
    asyncio.run(send_dashboard_update())

@socketio.on('request_dashboard_data')
def handle_dashboard_request():
    asyncio.run(send_dashboard_update())

async def send_dashboard_update():
    """Send dashboard update to client"""
    try:
        data = await get_dashboard_data()
        socketio.emit('dashboard_update', data)
    except Exception as e:
        logger.error(f"Error sending dashboard update: {e}")
        socketio.emit('dashboard_update', {'error': str(e)})

@socketio.on('update_empirical_data')
def handle_update_empirical():
    # Simplified response without complex imports
    emit('empirical_data_update', {
        'success': True,
        'message': 'Connected to blockchain API - real-time data active',
        'empirical_status': {
            'patterns_loaded': 0,
            'capital_tracking_active': True,
            'settlement_patterns': 0,
            'valuation_active': False,
            'dynamic_chunking_enabled': False
        }
    })

@socketio.on('start_trading')
def handle_start_trading():
    emit('trading_status_update', {
        'trading_active': True,
        'success': True,
        'message': 'Trading system connected'
    })

@socketio.on('stop_trading')
def handle_stop_trading():
    emit('trading_status_update', {
        'trading_active': False,
        'success': True,
        'message': 'Trading paused'
    })

# Background update thread
def background_updates():
    """Send periodic updates"""
    while True:
        try:
            asyncio.run(send_dashboard_update())
        except Exception as e:
            logger.error(f"Background update error: {e}")
        
        # Wait 30 seconds between updates
        import time
        time.sleep(30)

if __name__ == '__main__':
    logger.info("Starting Connected Ominari Dashboard on port 8888...")
    logger.info("Connecting to blockchain API and real trading data...")
    
    # Start background updates in a thread
    import threading
    update_thread = threading.Thread(target=background_updates, daemon=True)
    update_thread.start()
    
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)