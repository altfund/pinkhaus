#!/usr/bin/env python3
"""
Ominari Web Monitor - Good Dashboard with SQLite
Beautiful UI from original web_monitor.py with working SQLite data
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import func, desc
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ominari-trading-system-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Beautiful UI template from original web_monitor.py
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🎯 Ominari Trading Dashboard</title>
    <meta charset="utf-8">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Consolas', 'Monaco', monospace;
            background: #0a0a0a;
            color: #00ff00;
            padding: 0;
            margin: 0;
            overflow-x: hidden;
        }
        
        /* Header */
        .header {
            background: #111;
            padding: 15px 20px;
            border-bottom: 2px solid #00ff00;
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 60px;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
        }
        
        .header-title {
            font-size: 1.5em;
            font-weight: bold;
            color: #00ffff;
        }
        
        .header-time {
            color: #888;
        }
        
        .stats-bar {
            background: #1a1a1a;
            padding: 10px 20px;
            position: fixed;
            top: 60px;
            left: 0;
            right: 0;
            z-index: 999;
            display: flex;
            justify-content: space-around;
            border-bottom: 1px solid #333;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #00ffff;
        }
        
        .stat-label {
            font-size: 0.8em;
            color: #666;
        }
        
        /* Main layout */
        .main-container {
            margin-top: 110px;
            padding: 20px;
            display: grid;
            grid-template-columns: 1fr 1fr 400px;
            gap: 20px;
            max-width: 1800px;
            margin-left: auto;
            margin-right: auto;
        }
        
        /* Sections */
        .section {
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 20px;
            display: flex;
            flex-direction: column;
        }
        
        .section-title {
            font-size: 1.2em;
            color: #00ffff;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
            flex-shrink: 0;
        }
        
        .section-content {
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
        }
        
        /* Market table */
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th {
            background: #222;
            padding: 8px 6px;
            text-align: center;
            color: #00ffff;
            border-bottom: 2px solid #444;
            font-size: 0.9em;
        }
        
        td {
            padding: 8px 6px;
            border-bottom: 1px solid #222;
            text-align: center;
        }
        
        tr:hover {
            background: #1a1a1a;
        }
        
        .market-teams {
            font-weight: bold;
            color: #fff;
            font-size: 0.85em;
        }
        
        .odds-cell {
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 4px;
            margin: 1px;
            font-size: 0.9em;
        }
        
        .odds-value {
            color: #00ff00;
            font-weight: bold;
        }
        
        .status-live { color: #ff6666; }
        .status-scheduled { color: #888; }
        .status-starting { color: #ffff00; }
        
        /* Sport distribution */
        .sport-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #222;
        }
        
        .sport-name {
            color: #fff;
        }
        
        .sport-count {
            color: #00ff00;
            font-weight: bold;
        }
        
        /* Loading */
        .loading {
            text-align: center;
            padding: 50px;
            color: #666;
        }
        
        .spinner {
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 4px solid #333;
            border-radius: 50%;
            border-top-color: #00ff00;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #111;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #333;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #444;
        }
        
        /* Responsive */
        @media (max-width: 1400px) {
            .main-container {
                grid-template-columns: 1fr 1fr;
            }
            
            .stats-section {
                grid-column: 1 / -1;
            }
        }
        
        @media (max-width: 900px) {
            .main-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
    <div class="header">
        <div class="header-title">🎯 Ominari Trading Dashboard</div>
        <div class="header-time" id="current-time"></div>
    </div>
    
    <div class="stats-bar">
        <div class="stat-item">
            <div class="stat-value" id="total-markets">0</div>
            <div class="stat-label">Total Markets</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" id="api-markets">0</div>
            <div class="stat-label">API Markets</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" id="blockchain-markets">0</div>
            <div class="stat-label">Blockchain</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" id="active-markets">0</div>
            <div class="stat-label">Active</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" id="markets-with-odds">0</div>
            <div class="stat-label">With Odds</div>
        </div>
    </div>
    
    <div class="main-container">
        <div class="section">
            <div class="section-title">🎲 Live Markets</div>
            <div class="section-content" id="markets-container">
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Loading markets...</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">⛓️ Blockchain Markets</div>
            <div class="section-content" id="blockchain-container">
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Loading blockchain data...</p>
                </div>
            </div>
        </div>
        
        <div class="section stats-section">
            <div class="section-title">📊 Statistics</div>
            <div class="section-content">
                <div style="margin-bottom: 20px;">
                    <h4 style="color: #00ffff; margin-bottom: 10px;">Sport Distribution</h4>
                    <div id="sport-distribution"></div>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <h4 style="color: #00ffff; margin-bottom: 10px;">Data Sources</h4>
                    <div id="source-stats"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        // Update time
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = now.toUTCString();
        }
        setInterval(updateTime, 1000);
        updateTime();
        
        // Socket event handlers
        socket.on('connect', () => {
            console.log('Connected to server');
            socket.emit('request_all_data');
        });
        
        socket.on('stats', (data) => {
            document.getElementById('total-markets').textContent = data.total_markets.toLocaleString();
            document.getElementById('api-markets').textContent = data.api_markets.toLocaleString();
            document.getElementById('blockchain-markets').textContent = data.blockchain_markets.toLocaleString();
            document.getElementById('active-markets').textContent = data.active_markets.toLocaleString();
            document.getElementById('markets-with-odds').textContent = data.markets_with_odds.toLocaleString();
        });
        
        socket.on('markets_update', (data) => {
            renderMarketsTable(data.markets, 'markets-container');
        });
        
        socket.on('blockchain_update', (data) => {
            renderMarketsTable(data.markets, 'blockchain-container', true);
        });
        
        socket.on('sport_distribution', (data) => {
            const container = document.getElementById('sport-distribution');
            container.innerHTML = '';
            
            data.forEach(item => {
                const div = document.createElement('div');
                div.className = 'sport-item';
                div.innerHTML = `
                    <span class="sport-name">${item.sport}</span>
                    <span class="sport-count">${item.count.toLocaleString()}</span>
                `;
                container.appendChild(div);
            });
        });
        
        socket.on('source_stats', (data) => {
            const container = document.getElementById('source-stats');
            container.innerHTML = '';
            
            data.forEach(item => {
                const div = document.createElement('div');
                div.className = 'sport-item';
                const sourceClass = item.source.includes('blockchain') ? 'style="color: #ff9900"' : 'style="color: #00ccff"';
                div.innerHTML = `
                    <span ${sourceClass}>${item.source}</span>
                    <span class="sport-count">${item.count.toLocaleString()}</span>
                `;
                container.appendChild(div);
            });
        });
        
        function renderMarketsTable(markets, containerId, isBlockchain = false) {
            const container = document.getElementById(containerId);
            
            if (markets.length === 0) {
                container.innerHTML = '<p style="text-align: center; color: #666;">No markets available</p>';
                return;
            }
            
            let html = `
                <table>
                    <thead>
                        <tr>
                            <th>Match</th>
                            <th>Sport</th>
                            <th>Home</th>
                            <th>Draw</th>
                            <th>Away</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            markets.slice(0, 25).forEach(market => {
                const statusClass = getStatusClass(market.status);
                html += `
                    <tr>
                        <td class="market-teams">${market.home_team} vs ${market.away_team}</td>
                        <td style="font-size: 0.8em; color: #888;">${market.sport}</td>
                        <td class="odds-cell">${formatOdds(market.home_odds)}</td>
                        <td class="odds-cell">${formatOdds(market.draw_odds)}</td>
                        <td class="odds-cell">${formatOdds(market.away_odds)}</td>
                        <td><span class="${statusClass}">${market.status || 'Active'}</span></td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            container.innerHTML = html;
        }
        
        function formatOdds(odds) {
            return odds ? `<span class="odds-value">${odds.toFixed(2)}</span>` : '-';
        }
        
        function getStatusClass(status) {
            if (!status) return '';
            const s = status.toLowerCase();
            if (s.includes('live') || s.includes('play')) return 'status-live';
            if (s.includes('soon') || s.includes('imminent')) return 'status-starting';
            return 'status-scheduled';
        }
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            socket.emit('request_all_data');
        }, 30000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('request_all_data')
def handle_request_all_data():
    """Send all data to client"""
    emit('stats', get_stats())
    emit('markets_update', get_markets())
    emit('blockchain_update', get_blockchain_markets())
    emit('sport_distribution', get_sport_distribution())
    emit('source_stats', get_source_stats())

def get_stats():
    """Get overall statistics"""
    try:
        with db_manager.get_db_session() as db:
            # Total markets
            total = db.query(Market).count()
            
            # API markets
            api_markets = db.query(Market).filter(Market.source.like('%overtime%')).count()
            
            # Blockchain markets
            blockchain_markets = db.query(Market).filter(Market.source.like('%blockchain%')).count()
            
            # Active markets
            active = db.query(Market).filter(
                Market.is_finished == False
            ).count()
            
            # Markets with odds
            with_odds = db.query(func.count(func.distinct(Odd.source_id))).scalar() or 0
            
            return {
                'total_markets': total,
                'api_markets': api_markets,
                'blockchain_markets': blockchain_markets,
                'active_markets': active,
                'markets_with_odds': with_odds
            }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            'total_markets': 0,
            'api_markets': 0,
            'blockchain_markets': 0,
            'active_markets': 0,
            'markets_with_odds': 0
        }

def get_markets():
    """Get live markets from API"""
    try:
        with db_manager.get_db_session() as db:
            markets = db.query(Market).filter(
                Market.source.like('%overtime%')
            ).filter(
                Market.is_finished == False
            ).order_by(Market.maturity_date).limit(50).all()
            
            market_list = []
            for market in markets:
                market_data = {
                    'source_id': market.source_id,
                    'home_team': market.home_team,
                    'away_team': market.away_team,
                    'sport': market.sport,
                    'source': market.source,
                    'maturity_date': market.maturity_date.isoformat() if market.maturity_date else None,
                    'league_name': market.league_name,
                    'status': 'Active'
                }
                
                # Get odds
                odds = db.query(Odd).filter(
                    Odd.source_id == market.source_id
                ).order_by(Odd.position).all()
                
                # Parse odds by outcome
                market_data['home_odds'] = None
                market_data['draw_odds'] = None
                market_data['away_odds'] = None
                
                for odd in odds:
                    outcome = str(odd.outcome).lower() if odd.outcome else ''
                    if 'home' in outcome or outcome == 'option_1':
                        market_data['home_odds'] = odd.decimal_odds
                    elif 'away' in outcome or outcome == 'option_2':
                        market_data['away_odds'] = odd.decimal_odds
                    elif 'draw' in outcome or 'tie' in outcome or outcome == 'option_3':
                        market_data['draw_odds'] = odd.decimal_odds
                
                market_list.append(market_data)
            
            return {'markets': market_list}
    except Exception as e:
        logger.error(f"Error getting markets: {e}")
        return {'markets': []}

def get_blockchain_markets():
    """Get blockchain markets"""
    try:
        with db_manager.get_db_session() as db:
            markets = db.query(Market).filter(
                Market.source.like('%blockchain%')
            ).order_by(desc(Market.updated_at)).limit(30).all()
            
            market_list = []
            for market in markets:
                market_data = {
                    'source_id': market.source_id,
                    'home_team': market.home_team,
                    'away_team': market.away_team,
                    'sport': market.sport,
                    'source': market.source,
                    'updated_at': market.updated_at.isoformat() if market.updated_at else None,
                    'status': 'Blockchain'
                }
                
                # Get odds
                odds = db.query(Odd).filter(
                    Odd.source_id == market.source_id
                ).order_by(Odd.position).all()
                
                # Parse odds by outcome
                market_data['home_odds'] = None
                market_data['draw_odds'] = None
                market_data['away_odds'] = None
                
                for odd in odds:
                    outcome = str(odd.outcome).lower() if odd.outcome else ''
                    if 'home' in outcome or outcome == 'option_1':
                        market_data['home_odds'] = odd.decimal_odds
                    elif 'away' in outcome or outcome == 'option_2':
                        market_data['away_odds'] = odd.decimal_odds
                    elif 'draw' in outcome or 'tie' in outcome or outcome == 'option_3':
                        market_data['draw_odds'] = odd.decimal_odds
                
                market_list.append(market_data)
            
            return {'markets': market_list}
    except Exception as e:
        logger.error(f"Error getting blockchain markets: {e}")
        return {'markets': []}

def get_sport_distribution():
    """Get sport distribution"""
    try:
        with db_manager.get_db_session() as db:
            results = db.query(
                Market.sport,
                func.count(Market.source_id).label('count')
            ).group_by(Market.sport).order_by(desc('count')).limit(15).all()
            
            return [{'sport': r.sport, 'count': r.count} for r in results]
    except Exception as e:
        logger.error(f"Error getting sport distribution: {e}")
        return []

def get_source_stats():
    """Get source statistics"""
    try:
        with db_manager.get_db_session() as db:
            results = db.query(
                Market.source,
                func.count(Market.source_id).label('count')
            ).group_by(Market.source).order_by(desc('count')).all()
            
            return [{'source': r.source, 'count': r.count} for r in results]
    except Exception as e:
        logger.error(f"Error getting source stats: {e}")
        return []

def background_updates():
    """Background thread for periodic updates"""
    while True:
        try:
            with app.app_context():
                socketio.emit('stats', get_stats())
        except Exception as e:
            logger.error(f"Background update error: {e}")
        time.sleep(30)

if __name__ == '__main__':
    # Start background thread
    bg_thread = threading.Thread(target=background_updates, daemon=True)
    bg_thread.start()
    
    # Start server
    logger.info("🎯 Starting Ominari Trading Dashboard")
    logger.info("📊 Beautiful UI with all your data at: http://localhost:8888")
    logger.info("✨ Shows API markets, blockchain data, and real-time stats")
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)