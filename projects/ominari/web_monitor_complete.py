#!/usr/bin/env python3
"""
Complete Ominari Web Monitor - Shows all API and Blockchain data
Uses SQLite database directly
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import sqlite3
from contextlib import contextmanager
import threading
import time
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ominari-trading-system-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

DB_PATH = 'sport_odds.db'

# Database helper
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

COMPLETE_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Ominari Trading Dashboard - Complete View</title>
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
        
        /* Markets */
        .market-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 10px;
            transition: all 0.3s;
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
        
        .market-teams {
            font-size: 1.1em;
            font-weight: bold;
            color: #fff;
        }
        
        .market-info {
            text-align: right;
        }
        
        .market-sport {
            display: inline-block;
            padding: 2px 8px;
            background: #222;
            border: 1px solid #444;
            border-radius: 3px;
            font-size: 0.8em;
            margin-bottom: 3px;
        }
        
        .market-source {
            font-size: 0.8em;
            color: #666;
        }
        
        .source-api {
            color: #00ccff;
        }
        
        .source-blockchain {
            color: #ff9900;
        }
        
        .odds-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 10px 0;
        }
        
        .odd-box {
            background: #222;
            border: 1px solid #444;
            padding: 10px;
            text-align: center;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .odd-box:hover {
            border-color: #00ff00;
            transform: scale(1.05);
        }
        
        .odd-label {
            font-size: 0.8em;
            color: #888;
            margin-bottom: 5px;
        }
        
        .odd-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #00ff00;
        }
        
        .odd-change {
            font-size: 0.8em;
            margin-top: 2px;
        }
        
        .odd-up {
            color: #00ff00;
        }
        
        .odd-down {
            color: #ff4444;
        }
        
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
        
        /* Messages */
        .message {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            font-size: 0.9em;
        }
        
        .message-info {
            background: #1a3a1a;
            border: 1px solid #2a5a2a;
            color: #00ff00;
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
        <div class="header-title">🎯 Ominari Complete Dashboard</div>
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
                
                <div id="system-messages"></div>
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
            const container = document.getElementById('markets-container');
            container.innerHTML = '';
            
            if (data.markets.length === 0) {
                container.innerHTML = '<p style="text-align: center; color: #666;">No active markets</p>';
                return;
            }
            
            data.markets.forEach(market => {
                const card = createMarketCard(market);
                container.appendChild(card);
            });
        });
        
        socket.on('blockchain_update', (data) => {
            const container = document.getElementById('blockchain-container');
            container.innerHTML = '';
            
            if (data.markets.length === 0) {
                container.innerHTML = '<p style="text-align: center; color: #666;">No blockchain markets</p>';
                return;
            }
            
            data.markets.forEach(market => {
                const card = createMarketCard(market, true);
                container.appendChild(card);
            });
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
                const sourceClass = item.source.includes('blockchain') ? 'source-blockchain' : 'source-api';
                div.innerHTML = `
                    <span class="${sourceClass}">${item.source}</span>
                    <span class="sport-count">${item.count.toLocaleString()}</span>
                `;
                container.appendChild(div);
            });
        });
        
        function createMarketCard(market, isBlockchain = false) {
            const card = document.createElement('div');
            card.className = 'market-card';
            
            const sourceClass = isBlockchain ? 'source-blockchain' : 'source-api';
            const sourceIcon = isBlockchain ? '⛓️' : '📡';
            
            let oddsHtml = '';
            if (market.odds && market.odds.length > 0) {
                oddsHtml = '<div class="odds-row">';
                market.odds.forEach(odd => {
                    oddsHtml += `
                        <div class="odd-box">
                            <div class="odd-label">${odd.outcome.toUpperCase()}</div>
                            <div class="odd-value">${odd.decimal_odds.toFixed(3)}</div>
                        </div>
                    `;
                });
                oddsHtml += '</div>';
            }
            
            card.innerHTML = `
                <div class="market-header">
                    <div class="market-teams">${market.home_team} vs ${market.away_team}</div>
                    <div class="market-info">
                        <div class="market-sport">${market.sport}</div>
                        <div class="market-source ${sourceClass}">${sourceIcon} ${market.source}</div>
                    </div>
                </div>
                ${oddsHtml}
            `;
            
            return card;
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
    return render_template_string(COMPLETE_DASHBOARD)

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
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Total markets
        total = cursor.execute("SELECT COUNT(*) FROM market").fetchone()[0]
        
        # API markets
        api_markets = cursor.execute(
            "SELECT COUNT(*) FROM market WHERE source = 'api_live_real'"
        ).fetchone()[0]
        
        # Blockchain markets
        blockchain_markets = cursor.execute(
            "SELECT COUNT(*) FROM market WHERE source LIKE '%blockchain%'"
        ).fetchone()[0]
        
        # Active markets
        active = cursor.execute("""
            SELECT COUNT(*) FROM market 
            WHERE is_finished = 0 
            AND (maturity_date > datetime('now') OR maturity_date IS NULL)
        """).fetchone()[0]
        
        # Markets with odds
        with_odds = cursor.execute(
            "SELECT COUNT(DISTINCT source_id) FROM odd"
        ).fetchone()[0]
        
        return {
            'total_markets': total,
            'api_markets': api_markets,
            'blockchain_markets': blockchain_markets,
            'active_markets': active,
            'markets_with_odds': with_odds
        }

def get_markets():
    """Get live markets from API"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        markets = cursor.execute("""
            SELECT m.source_id, m.home_team, m.away_team, m.sport, 
                   m.source, m.maturity_date, m.league_name
            FROM market m
            WHERE m.source = 'api_live_real'
            AND m.is_finished = 0
            AND (m.maturity_date > datetime('now') OR m.maturity_date IS NULL)
            ORDER BY m.maturity_date
            LIMIT 50
        """).fetchall()
        
        market_list = []
        for row in markets:
            market = {
                'source_id': row['source_id'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'sport': row['sport'],
                'source': row['source'],
                'maturity_date': row['maturity_date'],
                'league_name': row['league_name']
            }
            
            # Get odds
            odds = cursor.execute("""
                SELECT outcome, decimal_odds, american_odds
                FROM odd
                WHERE source_id = ?
                ORDER BY position
            """, (row['source_id'],)).fetchall()
            
            if odds:
                market['odds'] = [
                    {
                        'outcome': o['outcome'],
                        'decimal_odds': o['decimal_odds'],
                        'american_odds': o['american_odds']
                    }
                    for o in odds
                ]
            
            market_list.append(market)
        
        return {'markets': market_list}

def get_blockchain_markets():
    """Get blockchain markets"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        markets = cursor.execute("""
            SELECT m.source_id, m.home_team, m.away_team, m.sport, 
                   m.source, m.updated_at
            FROM market m
            WHERE m.source LIKE '%blockchain%'
            ORDER BY m.updated_at DESC
            LIMIT 30
        """).fetchall()
        
        market_list = []
        for row in markets:
            market = {
                'source_id': row['source_id'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'sport': row['sport'],
                'source': row['source'],
                'updated_at': row['updated_at']
            }
            
            # Get odds
            odds = cursor.execute("""
                SELECT outcome, decimal_odds
                FROM odd
                WHERE source_id = ?
                ORDER BY position
            """, (row['source_id'],)).fetchall()
            
            if odds:
                market['odds'] = [
                    {
                        'outcome': o['outcome'],
                        'decimal_odds': o['decimal_odds']
                    }
                    for o in odds
                ]
            
            market_list.append(market)
        
        return {'markets': market_list}

def get_sport_distribution():
    """Get sport distribution"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        results = cursor.execute("""
            SELECT sport, COUNT(*) as count
            FROM market
            GROUP BY sport
            ORDER BY count DESC
            LIMIT 15
        """).fetchall()
        
        return [{'sport': r['sport'], 'count': r['count']} for r in results]

def get_source_stats():
    """Get source statistics"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        results = cursor.execute("""
            SELECT source, COUNT(*) as count
            FROM market
            GROUP BY source
            ORDER BY count DESC
        """).fetchall()
        
        return [{'source': r['source'], 'count': r['count']} for r in results]

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
    logger.info("Starting Complete Ominari Dashboard on http://localhost:8888")
    logger.info("Dashboard shows all API and Blockchain data")
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)