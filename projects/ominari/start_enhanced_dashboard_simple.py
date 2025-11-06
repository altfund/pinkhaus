#!/usr/bin/env python3
"""
Minimal startup script for the enhanced dashboard
Avoids complex imports to get the server running
"""

import os
import sys
import logging
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
import json
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

# Load the HTML template from web_monitor_blockchain.py
try:
    with open('web_monitor_blockchain.py', 'r') as f:
        content = f.read()
        # Extract the HTML template
        start = content.find('BLOCKCHAIN_DASHBOARD_HTML = """') + len('BLOCKCHAIN_DASHBOARD_HTML = """')
        end = content.rfind('"""')
        DASHBOARD_HTML = content[start:end]
except Exception as e:
    logger.error(f"Failed to load dashboard HTML: {e}")
    DASHBOARD_HTML = "<h1>Error loading dashboard</h1>"

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ominari-blockchain-trading-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/dashboard')
def api_dashboard():
    """API endpoint for dashboard data"""
    # Return mock data for now
    return jsonify({
        'markets': [
            {
                'match_id': '1',
                'home_team': 'Manchester United',
                'away_team': 'Liverpool',
                'sport': 'Soccer',
                'league': 'Premier League',
                'maturity_date': datetime.now(timezone.utc).isoformat(),
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
                'maturity_date': datetime.now(timezone.utc).isoformat(),
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
                'maturity_date': datetime.now(timezone.utc).isoformat(),
                'odds': 3.45,
                'position': 'Away Win',
                'blockchain_connected': True,
                'edge': -0.012,
                'signal_prob': 0.278,
                'implied_prob': 0.290
            }
        ],
        'trading_status': {
            'status': 'Active',
            'bankroll': 10000,
            'active_positions': 3,
            'valuation_engine_active': True
        },
        'stats': {
            'total_markets': 15,
            'blockchain_connected': 12,
            'blockchain_ratio': 80.0,
            'chunk_hours': 2.0,
            'dynamic_chunking_enabled': True
        },
        'config': {
            'testnet_mode': True,
            'simulation_mode': True,
            'network': 'Testnet'
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
    })

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('connected', {'data': 'Connected to enhanced dashboard'})

@socketio.on('request_dashboard_data')
def handle_dashboard_request():
    # Send the dashboard data
    socketio.emit('dashboard_update', {
        'markets': [],  # Will be populated by api_dashboard
        'trading_status': {
            'status': 'Demo Mode',
            'bankroll': 10000,
            'active_positions': 0
        },
        'stats': {
            'total_markets': 0,
            'blockchain_connected': 0,
            'blockchain_ratio': 0
        }
    })

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

if __name__ == '__main__':
    logger.info("Starting Enhanced Ominari Dashboard on port 8888...")
    logger.info("This is a demo version with mock data")
    socketio.run(app, host='0.0.0.0', port=8888, debug=False)