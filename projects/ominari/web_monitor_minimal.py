#!/usr/bin/env python3
"""
Minimal Ominari Dashboard - Just the enhanced UI without complex imports
"""

import os
import logging
from datetime import datetime, timezone
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ominari-blockchain-trading-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Load the HTML template from web_monitor_blockchain.py
try:
    with open('web_monitor_blockchain.py', 'r') as f:
        content = f.read()
        # Extract the HTML template
        start = content.find('BLOCKCHAIN_DASHBOARD_HTML = """') + len('BLOCKCHAIN_DASHBOARD_HTML = """')
        end = content.rfind('"""')
        end = content.rfind('"""', 0, end)  # Find the second to last occurrence
        DASHBOARD_HTML = content[start:end]
        logger.info("Dashboard HTML loaded successfully")
except Exception as e:
    logger.error(f"Failed to load dashboard HTML: {e}")
    DASHBOARD_HTML = "<h1>Error loading dashboard</h1>"

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template_string(DASHBOARD_HTML)

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('dashboard_update', get_demo_data())

@socketio.on('request_dashboard_data')
def handle_dashboard_request():
    emit('dashboard_update', get_demo_data())

def get_demo_data():
    """Get demo data for display"""
    current_time = datetime.now(timezone.utc)
    future_time = current_time.replace(hour=(current_time.hour + 2) % 24)
    
    return {
        'markets': [
            # Game 1 - Multiple markets
            {
                'match_id': '1',
                'home_team': 'Manchester United',
                'away_team': 'Liverpool',
                'sport': 'Soccer',
                'league': 'Premier League',
                'maturity_date': future_time.isoformat(),
                'odds': 2.15,
                'position': 'Home Win',
                'blockchain_connected': True,
                'edge': 0.0235,
                'signal_prob': 0.485,
                'implied_prob': 0.465
            },
            {
                'match_id': '1',
                'home_team': 'Manchester United',
                'away_team': 'Liverpool',
                'sport': 'Soccer',
                'league': 'Premier League',
                'maturity_date': future_time.isoformat(),
                'odds': 3.20,
                'position': 'Draw',
                'blockchain_connected': True,
                'edge': 0,
                'signal_prob': None,
                'implied_prob': 0.313
            },
            {
                'match_id': '1',
                'home_team': 'Manchester United',
                'away_team': 'Liverpool',
                'sport': 'Soccer',
                'league': 'Premier League',
                'maturity_date': future_time.isoformat(),
                'odds': 3.45,
                'position': 'Away Win',
                'blockchain_connected': True,
                'edge': -0.012,
                'signal_prob': 0.278,
                'implied_prob': 0.290
            },
            # Game 2
            {
                'match_id': '2',
                'home_team': 'Real Madrid',
                'away_team': 'Barcelona',
                'sport': 'Soccer',
                'league': 'La Liga',
                'maturity_date': future_time.replace(hour=(future_time.hour + 2) % 24).isoformat(),
                'odds': 1.85,
                'position': 'Home Win',
                'blockchain_connected': False,
                'edge': 0,
                'signal_prob': None,
                'implied_prob': 0.541
            },
            {
                'match_id': '2',
                'home_team': 'Real Madrid',
                'away_team': 'Barcelona',
                'sport': 'Soccer',
                'league': 'La Liga',
                'maturity_date': future_time.replace(hour=(future_time.hour + 2) % 24).isoformat(),
                'odds': 3.80,
                'position': 'Draw',
                'blockchain_connected': False,
                'edge': 0,
                'signal_prob': None,
                'implied_prob': 0.263
            },
            {
                'match_id': '2',
                'home_team': 'Real Madrid',
                'away_team': 'Barcelona',
                'sport': 'Soccer',
                'league': 'La Liga',
                'maturity_date': future_time.replace(hour=(future_time.hour + 2) % 24).isoformat(),
                'odds': 4.20,
                'position': 'Away Win',
                'blockchain_connected': False,
                'edge': 0,
                'signal_prob': None,
                'implied_prob': 0.238
            }
        ],
        'trading_status': {
            'status': 'Demo Mode',
            'bankroll': 10000,
            'active_positions': 3,
            'valuation_engine_active': True
        },
        'stats': {
            'total_markets': 6,
            'blockchain_connected': 3,
            'blockchain_ratio': 50.0,
            'chunk_hours': 2.0,
            'dynamic_chunking_enabled': True
        },
        'config': {
            'testnet_mode': True,
            'simulation_mode': True,
            'network': 'Demo'
        },
        'capital_state': {
            'available_cash': 7500,
            'pending_stakes': 500,
            'in_play_exposure': 1500,
            'settlement_pending': 500,
            'utilization_rate': 0.25,
            'value_at_risk': 1500,
            'expected_value': 150
        },
        'dynamic_chunking_status': {
            'enabled': True,
            'chunk_manager': {
                'empirical_data_loaded': 42
            },
            'capital_tracker': {
                'active': True
            },
            'settlement_analyzer': {
                'patterns_loaded': 15
            }
        }
    }

@socketio.on('update_empirical_data')
def handle_update_empirical():
    emit('empirical_data_update', {
        'success': True,
        'message': 'Demo mode - empirical data update simulated',
        'empirical_status': {
            'patterns_loaded': 42,
            'capital_tracking_active': True,
            'settlement_patterns': 15,
            'valuation_active': True,
            'dynamic_chunking_enabled': True
        }
    })

@socketio.on('start_trading')
def handle_start_trading():
    emit('trading_status_update', {
        'trading_active': True,
        'success': True,
        'message': 'Demo trading started'
    })

@socketio.on('stop_trading')
def handle_stop_trading():
    emit('trading_status_update', {
        'trading_active': False,
        'success': True,
        'message': 'Demo trading stopped'
    })

if __name__ == '__main__':
    logger.info("Starting Minimal Enhanced Dashboard on port 8888...")
    logger.info("This version shows the enhanced UI with demo data")
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)