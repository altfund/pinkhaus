#!/usr/bin/env python3
"""
Enhanced Blockchain-Integrated Ominari Web Monitor
Real-time dashboard with dynamic chunking and capital flow visualization
"""

import os
import asyncio
import threading
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
from decimal import Decimal
from typing import Dict, List, Optional, Any

# Set up environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

# Import blockchain trading components
from unified_data_fetcher import UnifiedDataFetcher
from integrated_blockchain_trading import IntegratedBlockchainTrading
from paper_trading_postgres_integrated import PaperTradingSessionManager
from edge_calculator import EdgeCalculator
from testnet_config_simple import setup_testnet_environment, TESTNET_CONFIG

# Import dynamic chunking components
from capital_exposure_tracker import CapitalExposureTracker
from settlement_analyzer import SettlementAnalyzer
from realtime_valuation_engine import RealTimeValuationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON encoder
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if hasattr(obj, '__float__'):
            return float(obj)
        return super().default(obj)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ominari-blockchain-trading-2024'
app.json_encoder = CustomJSONEncoder
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', json=json)

# Global components
unified_fetcher = None
trading_system = None
session_manager = None
edge_calculator = None

# Trading state
trading_active = False
trading_thread = None
current_cycle_data = {}

def initialize_components():
    """Initialize all blockchain trading components"""
    global unified_fetcher, trading_system, session_manager, edge_calculator
    
    try:
        # Setup testnet environment
        setup_testnet_environment()
        
        # Initialize components
        unified_fetcher = UnifiedDataFetcher(blockchain_first=True)
        session_manager = PaperTradingSessionManager()
        edge_calculator = EdgeCalculator()
        
        # Get or create session
        session_id = session_manager.get_current_session()
        if not session_id:
            session_id = session_manager.create_session(
                initial_bankroll=10000.0,
                description="Blockchain Web Trading Session"
            )
        
        # Initialize trading system
        trading_system = IntegratedBlockchainTrading(session_id)
        
        logger.info("✅ All blockchain components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        return False

async def get_blockchain_market_data():
    """Get market data using blockchain-first approach"""
    try:
        if not unified_fetcher:
            return []
        
        # Fetch markets using unified data fetcher
        markets = await unified_fetcher.fetch_all_markets()
        
        # Format for trading
        trading_markets = unified_fetcher.format_for_trading(markets)
        
        # Add blockchain connectivity info
        for market in trading_markets:
            market['blockchain_connected'] = market.get('has_blockchain_data', False)
            market['data_sources'] = market.get('data_sources', [])
            market['blockchain_address'] = market.get('blockchain_address', '')
        
        return trading_markets
        
    except Exception as e:
        logger.error(f"Error fetching blockchain market data: {e}")
        return []

async def get_trading_system_status():
    """Get current trading system status and metrics including dynamic chunking"""
    try:
        if not trading_system or not session_manager:
            return {
                'status': 'not_initialized',
                'session_id': None,
                'bankroll': 0,
                'blockchain_markets': 0,
                'active_positions': 0
            }
        
        # Get session info
        session_id = session_manager.get_current_session()
        session = session_manager.get_session(session_id) if session_id else None
        
        # Get market count
        markets = await get_blockchain_market_data()
        blockchain_markets = sum(1 for m in markets if m.get('blockchain_connected', False))
        
        # Get positions
        positions = session_manager.get_positions(session_id) if session_id else []
        active_positions = len([p for p in positions if p.get('status') in ['pending', 'open']])
        
        # Get comprehensive system status
        system_status = trading_system.get_system_status()
        
        return {
            'status': 'active' if trading_active else 'ready',
            'session_id': session_id,
            'bankroll': session['current_bankroll'] if session else 0,
            'blockchain_markets': blockchain_markets,
            'total_markets': len(markets),
            'active_positions': active_positions,
            'testnet_mode': TESTNET_CONFIG['use_testnet'],
            'simulation_mode': TESTNET_CONFIG['simulate_blockchain_calls'],
            'network': TESTNET_CONFIG['default_network'],
            'current_cycle': current_cycle_data,
            'dynamic_chunking': system_status.get('dynamic_chunking', {}),
            'capital_state': system_status.get('capital_state', {})
        }
        
    except Exception as e:
        logger.error(f"Error getting trading system status: {e}")
        return {'status': 'error', 'error': str(e)}

async def get_capital_flow_data():
    """Get capital flow visualization data"""
    try:
        if not trading_system:
            return {
                'capital_state': {},
                'chunk_exposure': {},
                'settlement_schedule': []
            }
        
        # Get session ID
        session_id = session_manager.get_current_session() if session_manager else None
        if not session_id:
            return {
                'capital_state': {},
                'chunk_exposure': {},
                'settlement_schedule': []
            }
        
        # Initialize capital tracker if needed
        capital_tracker = CapitalExposureTracker()
        
        try:
            # Get real-time metrics
            metrics = capital_tracker.get_real_time_metrics(session_id)
            
            return {
                'capital_state': metrics.get('capital_state', {}),
                'chunk_exposure': metrics.get('chunk_exposure', {}),
                'risk_metrics': metrics.get('risk_metrics', {}),
                'limits': metrics.get('limits', {})
            }
        except Exception as e:
            logger.error(f"Error getting capital flow data: {e}")
            return {
                'capital_state': {},
                'chunk_exposure': {},
                'settlement_schedule': []
            }
    
    except Exception as e:
        logger.error(f"Error getting capital flow data: {e}")
        return {
            'capital_state': {},
            'chunk_exposure': {},
            'settlement_schedule': []
        }

def format_market_for_display(market):
    """Format market data for web display"""
    return {
        'market_id': market.get('market_id', ''),
        'home_team': market.get('home_team', ''),
        'away_team': market.get('away_team', ''),
        'sport': market.get('sport', ''),
        'league': market.get('league', ''),
        'maturity_date': market.get('maturity_date', '').isoformat() if hasattr(market.get('maturity_date', ''), 'isoformat') else str(market.get('maturity_date', '')),
        'odds': market.get('odds', 0),
        'position': market.get('position', ''),
        'blockchain_connected': market.get('blockchain_connected', False),
        'blockchain_address': market.get('blockchain_address', '')[:20] + '...' if market.get('blockchain_address', '') else '',
        'data_sources': market.get('data_sources', []),
        'has_edge': False  # Will be calculated with signals
    }

async def run_trading_cycle():
    """Run a single trading cycle and update global state with enhanced metrics"""
    global current_cycle_data
    
    try:
        if not trading_system:
            return
        
        logger.info("🔄 Starting trading cycle from web interface")
        
        current_cycle_data = {
            'status': 'running',
            'start_time': datetime.now(timezone.utc).isoformat(),
            'markets_analyzed': 0,
            'trades_executed': 0,
            'stage': 'fetching_markets',
            'time_chunks': [],
            'current_chunk': None,
            'chunking_method': 'unknown',
            'capital_metrics': {}
        }
        
        # Emit status update
        socketio.emit('trading_cycle_update', current_cycle_data)
        
        # Run trading cycle
        result = await trading_system.run_trading_cycle(simulate=True)
        
        # Extract chunk information
        chunks_analyzed = result.get('chunks_analyzed', 0)
        chunking_method = result.get('chunking_method', 'unknown')
        time_horizon = result.get('time_horizon_hours', 0)
        
        current_cycle_data.update({
            'status': 'completed',
            'end_time': datetime.now(timezone.utc).isoformat(),
            'success': result.get('success', False),
            'markets_analyzed': result.get('markets_analyzed', 0),
            'trades_executed': result.get('trades_executed', 0),
            'chunks_analyzed': chunks_analyzed,
            'chunking_method': chunking_method,
            'time_horizon_hours': time_horizon,
            'capital_metrics': result.get('capital_metrics', {}),
            'error': result.get('error') if not result.get('success') else None
        })
        
        # Emit completion
        socketio.emit('trading_cycle_update', current_cycle_data)
        socketio.emit('dashboard_update', await get_dashboard_data())
        
        logger.info(f"✅ Trading cycle completed: {result.get('trades_executed', 0)} trades executed")
        
    except Exception as e:
        logger.error(f"❌ Trading cycle error: {e}")
        current_cycle_data.update({
            'status': 'error',
            'error': str(e),
            'end_time': datetime.now(timezone.utc).isoformat()
        })
        socketio.emit('trading_cycle_update', current_cycle_data)

async def continuous_trading_loop():
    """Continuous trading loop"""
    global trading_active
    
    logger.info("🚀 Starting continuous trading loop")
    
    while trading_active:
        try:
            await run_trading_cycle()
            
            if trading_active:  # Check if still active after cycle
                logger.info("⏳ Waiting 60 seconds until next cycle...")
                await asyncio.sleep(60)
                
        except Exception as e:
            logger.error(f"Error in continuous trading loop: {e}")
            await asyncio.sleep(30)  # Wait before retrying
    
    logger.info("⏹️ Continuous trading loop stopped")

def start_continuous_trading():
    """Start continuous trading in background thread"""
    global trading_active, trading_thread
    
    if trading_active:
        return False
    
    trading_active = True
    
    def run_async_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(continuous_trading_loop())
        loop.close()
    
    trading_thread = threading.Thread(target=run_async_loop, daemon=True)
    trading_thread.start()
    
    return True

def stop_continuous_trading():
    """Stop continuous trading"""
    global trading_active
    trading_active = False
    return True

async def get_dashboard_data():
    """Get complete dashboard data with enhanced capital metrics"""
    try:
        # Get market data
        markets = await get_blockchain_market_data()
        
        # Get trading system status
        trading_status = await get_trading_system_status()
        
        # Format markets for display
        formatted_markets = [format_market_for_display(m) for m in markets[:50]]  # Limit to 50 for web display
        
        # Calculate statistics
        blockchain_connected = sum(1 for m in markets if m.get('blockchain_connected', False))
        
        # Enhanced stats with dynamic chunking info
        stats = {
            'total_markets': len(markets),
            'blockchain_connected': blockchain_connected,
            'blockchain_ratio': (blockchain_connected / len(markets) * 100) if markets else 0,
            'supported_sports': TESTNET_CONFIG['supported_sports'],
            'supported_outcomes': TESTNET_CONFIG['supported_outcomes'],
            'chunk_hours': TESTNET_CONFIG.get('chunk_hours', 2.0),
            'max_hours_ahead': TESTNET_CONFIG.get('max_hours_ahead', 6.0),
            'chunk_size': TESTNET_CONFIG['chunk_size'],  # Legacy field
            'dynamic_chunking_enabled': trading_status.get('dynamic_chunking', {}).get('enabled', False)
        }
        
        # Add capital flow data if available
        capital_flow_data = await get_capital_flow_data()
        
        return {
            'markets': formatted_markets,
            'trading_status': trading_status,
            'stats': stats,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'config': {
                'testnet_mode': TESTNET_CONFIG['use_testnet'],
                'simulation_mode': TESTNET_CONFIG['simulate_blockchain_calls'],
                'network': TESTNET_CONFIG['default_network']
            },
            'capital_flow': capital_flow_data,
            'dynamic_chunking_status': trading_status.get('dynamic_chunking', {}),
            'capital_state': trading_status.get('capital_state', {})
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        return {'error': str(e)}

# Web routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template_string(ENHANCED_DASHBOARD_HTML)

@app.route('/api/dashboard')
async def api_dashboard():
    """API endpoint for dashboard data"""
    data = await get_dashboard_data()
    return jsonify(data)

@app.route('/api/start_trading', methods=['POST'])
def api_start_trading():
    """Start continuous trading"""
    success = start_continuous_trading()
    return jsonify({'success': success, 'trading_active': trading_active})

@app.route('/api/stop_trading', methods=['POST'])
def api_stop_trading():
    """Stop continuous trading"""
    success = stop_continuous_trading()
    return jsonify({'success': success, 'trading_active': trading_active})

@app.route('/api/single_cycle', methods=['POST'])
async def api_single_cycle():
    """Run a single trading cycle"""
    if trading_active:
        return jsonify({'success': False, 'error': 'Continuous trading is active'})
    
    # Run single cycle in background
    async def run_cycle():
        await run_trading_cycle()
    
    asyncio.create_task(run_cycle())
    return jsonify({'success': True, 'message': 'Trading cycle started'})

@app.route('/api/update_empirical_data', methods=['POST'])
async def api_update_empirical_data():
    """Update empirical settlement data"""
    if trading_system:
        success = await trading_system.update_empirical_data_and_restart()
        return jsonify({'success': success})
    return jsonify({'success': False, 'error': 'Trading system not initialized'})

# WebSocket events
@socketio.on('connect')
def handle_connect():
    logger.info(f"Web client connected: {request.sid}")
    
@socketio.on('request_dashboard_data')
async def handle_request_dashboard_data():
    """Send current dashboard data"""
    data = await get_dashboard_data()
    emit('dashboard_update', data)

@socketio.on('start_trading')
def handle_start_trading():
    """Start continuous trading via WebSocket"""
    success = start_continuous_trading()
    emit('trading_status_update', {'trading_active': trading_active, 'success': success})

@socketio.on('stop_trading')
def handle_stop_trading():
    """Stop continuous trading via WebSocket"""
    success = stop_continuous_trading()
    emit('trading_status_update', {'trading_active': trading_active, 'success': success})

@socketio.on('update_empirical_data')
async def handle_update_empirical_data():
    """Update empirical settlement data"""
    if trading_system:
        success = await trading_system.update_empirical_data_and_restart()
        emit('empirical_data_update', {'success': success})

# Background updates
def background_updates():
    """Background thread for periodic dashboard updates"""
    while True:
        try:
            async def update_dashboard():
                data = await get_dashboard_data()
                socketio.emit('dashboard_update', data)
            
            # Run async function in background
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(update_dashboard())
            loop.close()
            
        except Exception as e:
            logger.error(f"Background update error: {e}")
        
        time.sleep(30)  # Update every 30 seconds

# Enhanced Dashboard HTML template
ENHANCED_DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Ominari Blockchain Trading - Dynamic Chunking</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Consolas', 'Monaco', monospace;
            background: #0a0a0a;
            color: #00ff00;
            overflow-x: hidden;
        }
        
        .header {
            background: #111;
            padding: 15px 20px;
            border-bottom: 2px solid #00ff00;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .status-bar {
            background: #0f0f0f;
            padding: 10px 20px;
            border-bottom: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .main-container {
            display: grid;
            grid-template-columns: 1fr 450px;
            gap: 20px;
            padding: 20px;
            height: calc(100vh - 120px);
        }
        
        .markets-section {
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 15px;
            overflow-y: auto;
        }
        
        .controls-section {
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 15px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            overflow-y: auto;
        }
        
        .market-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 10px;
            transition: all 0.3s;
        }
        
        .market-card.blockchain-connected {
            border-left: 3px solid #00ff00;
        }
        
        .market-card:hover {
            border-color: #00ff00;
            transform: translateY(-2px);
        }
        
        .market-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .blockchain-indicator {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            font-size: 12px;
        }
        
        .connected { color: #00ff00; }
        .not-connected { color: #666; }
        
        .button {
            background: #1a1a1a;
            border: 1px solid #00ff00;
            color: #00ff00;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-family: inherit;
            transition: all 0.3s;
        }
        
        .button:hover {
            background: #00ff00;
            color: #000;
        }
        
        .button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .button.danger {
            border-color: #ff4444;
            color: #ff4444;
        }
        
        .button.danger:hover {
            background: #ff4444;
            color: #000;
        }
        
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .stat-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 10px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #00ff00;
        }
        
        .stat-label {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }
        
        .capital-flow-section {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 15px;
        }
        
        .capital-bar {
            display: flex;
            height: 40px;
            border-radius: 5px;
            overflow: hidden;
            margin: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.5);
        }
        
        .capital-segment {
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            color: #000;
            transition: all 0.3s;
            position: relative;
        }
        
        .capital-available { background: #00ff00; }
        .capital-pending { background: #ffaa00; }
        .capital-in-play { background: #0088ff; }
        .capital-settlement { background: #ff88ff; }
        
        .capital-legend {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-top: 10px;
            font-size: 12px;
            gap: 10px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .legend-color {
            width: 15px;
            height: 15px;
            border-radius: 3px;
        }
        
        .chunk-timeline {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
            max-height: 150px;
            overflow-y: auto;
        }
        
        .chunk-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px;
            margin-bottom: 5px;
            background: #222;
            border-radius: 3px;
            border-left: 3px solid #00ff00;
        }
        
        .chunk-info {
            font-size: 12px;
        }
        
        .dynamic-indicator {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: bold;
            margin-left: 10px;
        }
        
        .dynamic-enabled {
            background: #00ff00;
            color: #000;
        }
        
        .dynamic-disabled {
            background: #666;
            color: #fff;
        }
        
        .cycle-progress {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 15px;
        }
        
        .progress-bar {
            background: #333;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            background: #00ff00;
            height: 100%;
            transition: width 0.3s;
            border-radius: 10px;
        }
        
        .log-section {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 10px;
            height: 200px;
            overflow-y: auto;
            font-size: 12px;
        }
        
        .log-entry {
            margin-bottom: 5px;
            padding: 2px 0;
        }
        
        .log-timestamp {
            color: #666;
        }
        
        .positive { color: #00ff00; }
        .negative { color: #ff4444; }
        .warning { color: #ffaa00; }
        
        .empirical-data-section {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
            font-size: 12px;
        }
        
        .empirical-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }
        
        .empirical-stat {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .empirical-value {
            color: #00ff00;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🔗 Ominari Blockchain Trading</h1>
            <div>Dynamic Chunking & Real-time Capital Management</div>
        </div>
        <div id="connection-status">⚡ Connecting...</div>
    </div>
    
    <div class="status-bar">
        <div id="trading-status">🔄 Ready</div>
        <div id="system-stats">📊 Loading...</div>
        <div id="network-info">🧪 Testnet</div>
    </div>
    
    <div class="main-container">
        <div class="markets-section">
            <h2>📈 Live Markets</h2>
            <div id="markets-container">Loading markets...</div>
        </div>
        
        <div class="controls-section">
            <div>
                <h3>🎮 Trading Controls</h3>
                <div style="display: flex; gap: 10px; margin-top: 10px;">
                    <button id="start-trading" class="button">Start Continuous</button>
                    <button id="stop-trading" class="button danger" disabled>Stop Trading</button>
                </div>
                <button id="single-cycle" class="button" style="width: 100%; margin-top: 10px;">Run Single Cycle</button>
                <button id="update-empirical" class="button" style="width: 100%; margin-top: 5px;">Update Empirical Data</button>
            </div>
            
            <div>
                <h3>📊 System Stats</h3>
                <div class="stat-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="total-markets">0</div>
                        <div class="stat-label">Total Markets</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="blockchain-markets">0</div>
                        <div class="stat-label">Blockchain Connected</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="bankroll">$0</div>
                        <div class="stat-label">Bankroll</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="active-positions">0</div>
                        <div class="stat-label">Active Positions</div>
                    </div>
                </div>
            </div>
            
            <div class="capital-flow-section" id="capital-flow-container">
                <h4>💰 Capital Flow <span id="dynamic-chunking-indicator" class="dynamic-indicator dynamic-disabled">Fixed Chunks</span></h4>
                <div class="capital-bar" id="capital-bar">
                    <div class="capital-segment capital-available" style="width: 100%;">Available</div>
                </div>
                <div class="capital-legend">
                    <div class="legend-item">
                        <div class="legend-color capital-available"></div>
                        <span>Available: $<span id="capital-available">0</span></span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color capital-pending"></div>
                        <span>Pending: $<span id="capital-pending">0</span></span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color capital-in-play"></div>
                        <span>In-Play: $<span id="capital-in-play">0</span></span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color capital-settlement"></div>
                        <span>Settlement: $<span id="capital-settlement">0</span></span>
                    </div>
                </div>
                <div style="margin-top: 10px; font-size: 12px; color: #888;">
                    <div>Utilization: <span id="capital-utilization" style="color: #00ff00;">0%</span></div>
                    <div>Value at Risk: $<span id="value-at-risk" style="color: #ffaa00;">0</span></div>
                    <div>Expected Value: $<span id="expected-value" style="color: #00ff00;">0</span></div>
                </div>
            </div>
            
            <div class="cycle-progress">
                <h3>🔄 Trading Cycle</h3>
                <div id="cycle-status">Idle</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="cycle-progress" style="width: 0%"></div>
                </div>
                <div id="cycle-details">Ready to start</div>
                <div class="chunk-timeline" id="chunk-timeline" style="display: none;">
                    <h4>📦 Time Chunks</h4>
                    <div id="chunk-list"></div>
                </div>
            </div>
            
            <div class="empirical-data-section" id="empirical-data-section" style="display: none;">
                <h4>📊 Empirical Data</h4>
                <div class="empirical-stats">
                    <div class="empirical-stat">
                        <span>Patterns Loaded:</span>
                        <span id="patterns-loaded" class="empirical-value">0</span>
                    </div>
                    <div class="empirical-stat">
                        <span>Capital Tracking:</span>
                        <span id="capital-tracking" class="empirical-value">0</span>
                    </div>
                    <div class="empirical-stat">
                        <span>Settlement Analysis:</span>
                        <span id="settlement-analysis" class="empirical-value">0</span>
                    </div>
                    <div class="empirical-stat">
                        <span>Real-time Valuation:</span>
                        <span id="realtime-valuation" class="empirical-value">Inactive</span>
                    </div>
                </div>
            </div>
            
            <div>
                <h3>📝 Activity Log</h3>
                <div class="log-section" id="activity-log">
                    <div class="log-entry">
                        <span class="log-timestamp">[Ready]</span> System initialized
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Initialize socket connection
        const socket = io();
        let isConnected = false;
        let tradingActive = false;
        
        // Connection handlers
        socket.on('connect', function() {
            isConnected = true;
            document.getElementById('connection-status').textContent = '✅ Connected';
            socket.emit('request_dashboard_data');
            addLogEntry('Connected to server');
        });
        
        socket.on('disconnect', function() {
            isConnected = false;
            document.getElementById('connection-status').textContent = '❌ Disconnected';
            addLogEntry('Disconnected from server', 'warning');
        });
        
        // Data handlers
        socket.on('dashboard_update', function(data) {
            updateDashboard(data);
        });
        
        socket.on('trading_cycle_update', function(data) {
            updateTradingCycle(data);
        });
        
        socket.on('trading_status_update', function(data) {
            tradingActive = data.trading_active;
            updateTradingControls();
        });
        
        socket.on('empirical_data_update', function(data) {
            if (data.success) {
                addLogEntry('Empirical data updated successfully', 'positive');
            } else {
                addLogEntry('Failed to update empirical data', 'warning');
            }
        });
        
        // Control handlers
        document.getElementById('start-trading').onclick = function() {
            socket.emit('start_trading');
            addLogEntry('Starting continuous trading...');
        };
        
        document.getElementById('stop-trading').onclick = function() {
            socket.emit('stop_trading');
            addLogEntry('Stopping trading...');
        };
        
        document.getElementById('single-cycle').onclick = function() {
            fetch('/api/single_cycle', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        addLogEntry('Single cycle started');
                    } else {
                        addLogEntry('Failed to start cycle: ' + data.error, 'warning');
                    }
                });
        };
        
        document.getElementById('update-empirical').onclick = function() {
            socket.emit('update_empirical_data');
            addLogEntry('Updating empirical data...');
        };
        
        function updateDashboard(data) {
            if (data.error) {
                addLogEntry('Dashboard error: ' + data.error, 'warning');
                return;
            }
            
            // Update system stats
            document.getElementById('total-markets').textContent = data.stats?.total_markets || 0;
            document.getElementById('blockchain-markets').textContent = data.stats?.blockchain_connected || 0;
            document.getElementById('bankroll').textContent = '$' + (data.trading_status?.bankroll || 0).toLocaleString();
            document.getElementById('active-positions').textContent = data.trading_status?.active_positions || 0;
            
            // Update status indicators
            document.getElementById('trading-status').textContent = '🔄 ' + (data.trading_status?.status || 'Unknown');
            
            const chunkingMethod = data.stats?.dynamic_chunking_enabled ? 'Dynamic' : 'Fixed';
            document.getElementById('system-stats').textContent = 
                `📊 ${data.stats?.blockchain_ratio?.toFixed(1) || 0}% blockchain | ⏰ ${chunkingMethod} chunks`;
            document.getElementById('network-info').textContent = 
                `🧪 ${data.config?.network || 'Unknown'} ${data.config?.testnet_mode ? '(Testnet)' : '(Mainnet)'}`;
            
            // Update capital flow visualization
            updateCapitalFlow(data.capital_state || {});
            
            // Update dynamic chunking indicator
            const dynamicIndicator = document.getElementById('dynamic-chunking-indicator');
            if (data.stats?.dynamic_chunking_enabled) {
                dynamicIndicator.className = 'dynamic-indicator dynamic-enabled';
                dynamicIndicator.textContent = 'Dynamic Chunks';
                
                // Show empirical data section
                document.getElementById('empirical-data-section').style.display = 'block';
                updateEmpiricalData(data.dynamic_chunking_status || {});
            } else {
                dynamicIndicator.className = 'dynamic-indicator dynamic-disabled';
                dynamicIndicator.textContent = 'Fixed Chunks';
                document.getElementById('empirical-data-section').style.display = 'none';
            }
            
            // Update markets
            updateMarkets(data.markets || []);
        }
        
        function updateMarkets(markets) {
            const container = document.getElementById('markets-container');
            
            if (markets.length === 0) {
                container.innerHTML = '<div>No markets available</div>';
                return;
            }
            
            container.innerHTML = markets.slice(0, 20).map(market => `
                <div class="market-card ${market.blockchain_connected ? 'blockchain-connected' : ''}">
                    <div class="market-header">
                        <div><strong>${market.home_team} vs ${market.away_team}</strong></div>
                        <div class="blockchain-indicator ${market.blockchain_connected ? 'connected' : 'not-connected'}">
                            ${market.blockchain_connected ? '🔗' : '⚫'} 
                            ${market.blockchain_connected ? 'Blockchain' : 'API Only'}
                        </div>
                    </div>
                    <div style="font-size: 12px; color: #888;">
                        ${market.sport} • ${market.position} @ ${market.odds}
                        ${market.blockchain_address ? `<br>📍 ${market.blockchain_address}` : ''}
                    </div>
                </div>
            `).join('');
        }
        
        function updateTradingCycle(data) {
            const statusElement = document.getElementById('cycle-status');
            const progressElement = document.getElementById('cycle-progress');
            const detailsElement = document.getElementById('cycle-details');
            const chunkTimeline = document.getElementById('chunk-timeline');
            
            if (data.status === 'running') {
                statusElement.textContent = 'Running...';
                progressElement.style.width = '50%';
                let details = `Stage: ${data.stage || 'processing'}`;
                if (data.current_chunk) {
                    details += ` | Time chunk: ${data.current_chunk}`;
                }
                detailsElement.textContent = details;
                addLogEntry(`Trading cycle started - ${data.stage || 'processing'}`);
            } else if (data.status === 'completed') {
                statusElement.textContent = data.success ? 'Completed ✅' : 'Failed ❌';
                progressElement.style.width = '100%';
                
                let summary = `${data.markets_analyzed || 0} markets, ${data.trades_executed || 0} trades`;
                if (data.chunks_analyzed > 0) {
                    summary += ` across ${data.chunks_analyzed} ${data.chunking_method} chunks`;
                    summary += ` (${data.time_horizon_hours || 0}hr horizon)`;
                }
                detailsElement.textContent = summary;
                
                // Update capital metrics if available
                if (data.capital_metrics && Object.keys(data.capital_metrics).length > 0) {
                    const cm = data.capital_metrics;
                    addLogEntry(
                        `Capital: Available $${cm.available_cash?.toFixed(0) || 0}, ` +
                        `Utilization ${(cm.utilization_rate * 100).toFixed(1)}%, ` +
                        `VaR $${cm.value_at_risk?.toFixed(0) || 0}`,
                        'info'
                    );
                }
                
                addLogEntry(`Cycle completed: ${data.trades_executed || 0} trades executed`, 
                    data.success ? 'positive' : 'warning');
                
                // Show chunk timeline if chunks were analyzed
                if (data.chunks_analyzed > 0) {
                    chunkTimeline.style.display = 'block';
                    updateChunkTimeline(data.time_chunks || []);
                }
                
                // Reset after a few seconds
                setTimeout(() => {
                    statusElement.textContent = 'Ready';
                    progressElement.style.width = '0%';
                    detailsElement.textContent = 'Ready for next cycle';
                    chunkTimeline.style.display = 'none';
                }, 5000);
            } else if (data.status === 'error') {
                statusElement.textContent = 'Error ❌';
                progressElement.style.width = '0%';
                detailsElement.textContent = data.error || 'Unknown error';
                addLogEntry(`Cycle error: ${data.error}`, 'warning');
            }
        }
        
        function updateTradingControls() {
            document.getElementById('start-trading').disabled = tradingActive;
            document.getElementById('stop-trading').disabled = !tradingActive;
            document.getElementById('single-cycle').disabled = tradingActive;
        }
        
        function updateCapitalFlow(capitalState) {
            if (!capitalState || Object.keys(capitalState).length === 0) {
                // Default state if no data
                document.getElementById('capital-bar').innerHTML = 
                    '<div class="capital-segment capital-available" style="width: 100%;">Available</div>';
                document.getElementById('capital-available').textContent = '0';
                document.getElementById('capital-pending').textContent = '0';
                document.getElementById('capital-in-play').textContent = '0';
                document.getElementById('capital-settlement').textContent = '0';
                document.getElementById('capital-utilization').textContent = '0%';
                document.getElementById('value-at-risk').textContent = '0';
                document.getElementById('expected-value').textContent = '0';
                return;
            }
            
            // Calculate percentages
            const total = capitalState.available_cash + 
                         (capitalState.pending_stakes || 0) + 
                         (capitalState.in_play_exposure || 0) + 
                         (capitalState.settlement_pending || 0);
            
            if (total === 0) return;
            
            const availablePct = (capitalState.available_cash / total * 100).toFixed(1);
            const pendingPct = ((capitalState.pending_stakes || 0) / total * 100).toFixed(1);
            const inPlayPct = ((capitalState.in_play_exposure || 0) / total * 100).toFixed(1);
            const settlementPct = ((capitalState.settlement_pending || 0) / total * 100).toFixed(1);
            
            // Update capital bar
            let barHTML = '';
            if (parseFloat(availablePct) > 5) {
                barHTML += `<div class="capital-segment capital-available" style="width: ${availablePct}%;">` +
                          `${parseFloat(availablePct) > 15 ? 'Available' : ''}</div>`;
            }
            if (parseFloat(pendingPct) > 5) {
                barHTML += `<div class="capital-segment capital-pending" style="width: ${pendingPct}%;">` +
                          `${parseFloat(pendingPct) > 15 ? 'Pending' : ''}</div>`;
            }
            if (parseFloat(inPlayPct) > 5) {
                barHTML += `<div class="capital-segment capital-in-play" style="width: ${inPlayPct}%;">` +
                          `${parseFloat(inPlayPct) > 15 ? 'In-Play' : ''}</div>`;
            }
            if (parseFloat(settlementPct) > 5) {
                barHTML += `<div class="capital-segment capital-settlement" style="width: ${settlementPct}%;">` +
                          `${parseFloat(settlementPct) > 15 ? 'Settlement' : ''}</div>`;
            }
            
            document.getElementById('capital-bar').innerHTML = barHTML || 
                '<div class="capital-segment capital-available" style="width: 100%;">Available</div>';
            
            // Update values
            document.getElementById('capital-available').textContent = capitalState.available_cash?.toFixed(0) || '0';
            document.getElementById('capital-pending').textContent = capitalState.pending_stakes?.toFixed(0) || '0';
            document.getElementById('capital-in-play').textContent = capitalState.in_play_exposure?.toFixed(0) || '0';
            document.getElementById('capital-settlement').textContent = capitalState.settlement_pending?.toFixed(0) || '0';
            document.getElementById('capital-utilization').textContent = 
                ((capitalState.utilization_rate || 0) * 100).toFixed(1) + '%';
            document.getElementById('value-at-risk').textContent = capitalState.value_at_risk?.toFixed(0) || '0';
            document.getElementById('expected-value').textContent = capitalState.expected_value?.toFixed(0) || '0';
        }
        
        function updateChunkTimeline(chunks) {
            const chunkList = document.getElementById('chunk-list');
            if (!chunks || chunks.length === 0) {
                chunkList.innerHTML = '<div>No chunks processed</div>';
                return;
            }
            
            chunkList.innerHTML = chunks.map((chunk, idx) => `
                <div class="chunk-item">
                    <div class="chunk-info">
                        <strong>Chunk ${idx + 1}</strong>: ${chunk.label || chunk.chunk_id || 'Unknown'}<br>
                        Markets: ${chunk.match_count || 0} | Stakes: $${chunk.total_stakes?.toFixed(0) || 0}
                    </div>
                    <div style="text-align: right; font-size: 11px; color: #888;">
                        ${chunk.start_time || 'N/A'}
                    </div>
                </div>
            `).join('');
        }
        
        function updateEmpiricalData(dynamicStatus) {
            if (!dynamicStatus || !dynamicStatus.enabled) return;
            
            document.getElementById('patterns-loaded').textContent = 
                dynamicStatus.chunk_manager?.empirical_data_loaded || 0;
            
            document.getElementById('capital-tracking').textContent = 
                `${dynamicStatus.capital_tracker?.positions_tracked || 0} pos`;
            
            document.getElementById('settlement-analysis').textContent = 
                `${dynamicStatus.settlement_analyzer?.league_stats_available || 0} leagues`;
            
            document.getElementById('realtime-valuation').textContent = 
                dynamicStatus.valuation_engine?.active ? 
                `Active (${dynamicStatus.valuation_engine.live_odds_tracked || 0} odds)` : 'Inactive';
        }
        
        function addLogEntry(message, type = 'info') {
            const logContainer = document.getElementById('activity-log');
            const timestamp = new Date().toLocaleTimeString();
            const entryClass = type === 'positive' ? 'positive' : 
                              type === 'warning' ? 'warning' : 
                              type === 'negative' ? 'negative' : '';
            
            const entry = document.createElement('div');
            entry.className = `log-entry ${entryClass}`;
            entry.innerHTML = `<span class="log-timestamp">[${timestamp}]</span> ${message}`;
            
            logContainer.appendChild(entry);
            logContainer.scrollTop = logContainer.scrollHeight;
            
            // Keep only last 50 entries
            while (logContainer.children.length > 50) {
                logContainer.removeChild(logContainer.firstChild);
            }
        }
        
        // Request initial data
        setInterval(() => {
            if (isConnected) {
                socket.emit('request_dashboard_data');
            }
        }, 30000);
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    print("🚀 Starting Enhanced Blockchain-Integrated Web Monitor")
    print("=" * 60)
    
    # Initialize components
    if not initialize_components():
        print("❌ Failed to initialize components")
        exit(1)
    
    print("✅ Components initialized successfully")
    print(f"📊 Configuration:")
    print(f"  Network: {TESTNET_CONFIG['default_network']}")
    print(f"  Testnet: {TESTNET_CONFIG['use_testnet']}")
    print(f"  Simulation: {TESTNET_CONFIG['simulate_blockchain_calls']}")
    print(f"  Dynamic Chunking: {os.getenv('USE_DYNAMIC_CHUNKING', 'true')}")
    
    # Start background updates
    update_thread = threading.Thread(target=background_updates, daemon=True)
    update_thread.start()
    
    print(f"\n🌐 Starting web server on http://localhost:8889")
    print(f"💡 Enhanced Features:")
    print(f"  • Dynamic chunking with empirical data")
    print(f"  • Real-time capital flow visualization")
    print(f"  • Value at Risk (VaR) monitoring")
    print(f"  • Settlement pattern analysis")
    print(f"  • Live position valuation tracking")
    
    # Run the app
    socketio.run(app, host='0.0.0.0', port=8889, debug=False, allow_unsafe_werkzeug=True)