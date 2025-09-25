#!/usr/bin/env python3
"""
Enhanced Ominari Web Monitor - Comprehensive Dashboard
Shows API markets, blockchain data, sport distributions, and real-time odds
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import sqlite3
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ominari-trading-system-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

DB_PATH = "sport_odds.db"

ENHANCED_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Ominari Comprehensive Dashboard</title>
    <meta charset="utf-8">
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
        }
        
        .header-title {
            font-size: 1.5em;
            font-weight: bold;
            color: #00ffff;
        }
        
        .header-stats {
            display: flex;
            gap: 30px;
            align-items: center;
        }
        
        .header-stat {
            text-align: center;
        }
        
        .header-stat-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #fff;
        }
        
        .header-stat-label {
            font-size: 0.8em;
            color: #888;
        }
        
        /* Tabs */
        .tabs {
            background: #1a1a1a;
            padding: 10px 20px;
            border-bottom: 1px solid #333;
            position: sticky;
            top: 60px;
            z-index: 900;
        }
        
        .tab {
            display: inline-block;
            padding: 10px 20px;
            margin-right: 10px;
            background: #222;
            border: 1px solid #444;
            border-radius: 5px 5px 0 0;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .tab:hover {
            background: #333;
            border-color: #00ff00;
        }
        
        .tab.active {
            background: #333;
            border-color: #00ff00;
            color: #00ffff;
            border-bottom: 1px solid #333;
        }
        
        /* Main container */
        .main-container {
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        /* Grid layouts */
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        
        /* Cards */
        .stat-card {
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }
        
        .stat-card-value {
            font-size: 2em;
            font-weight: bold;
            color: #00ffff;
            margin-bottom: 10px;
        }
        
        .stat-card-label {
            font-size: 0.9em;
            color: #888;
        }
        
        .section {
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .section-title {
            font-size: 1.3em;
            color: #00ffff;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }
        
        /* Market cards */
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
            align-items: flex-start;
            margin-bottom: 10px;
        }
        
        .market-teams {
            font-size: 1.1em;
            font-weight: bold;
            color: #fff;
            flex: 1;
        }
        
        .market-meta {
            text-align: right;
        }
        
        .market-sport {
            display: inline-block;
            padding: 2px 8px;
            background: #222;
            border: 1px solid #444;
            border-radius: 3px;
            font-size: 0.8em;
            color: #00ff00;
            margin-bottom: 5px;
        }
        
        .market-source {
            font-size: 0.8em;
            color: #666;
        }
        
        .blockchain-source {
            color: #ff9900;
        }
        
        .api-source {
            color: #00ccff;
        }
        
        .odds-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 10px;
        }
        
        .odd-box {
            background: #222;
            border: 1px solid #444;
            padding: 10px;
            text-align: center;
            border-radius: 4px;
            transition: all 0.2s;
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
        
        .odd-american {
            font-size: 0.9em;
            color: #666;
        }
        
        /* Sport distribution */
        .sport-bar {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .sport-name {
            width: 120px;
            color: #fff;
        }
        
        .sport-bar-fill {
            height: 20px;
            background: #00ff00;
            border-radius: 3px;
            margin: 0 10px;
            transition: width 0.5s ease;
        }
        
        .sport-count {
            color: #888;
        }
        
        /* Tables */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        
        .data-table th {
            background: #222;
            padding: 10px;
            text-align: left;
            color: #00ffff;
            border-bottom: 2px solid #444;
        }
        
        .data-table td {
            padding: 10px;
            border-bottom: 1px solid #222;
        }
        
        .data-table tr:hover {
            background: #1a1a1a;
        }
        
        /* Charts container */
        .chart-container {
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            height: 400px;
        }
        
        /* Loading */
        .loading {
            text-align: center;
            padding: 50px;
            color: #666;
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
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">🎯 Ominari Comprehensive Trading Dashboard</div>
        <div class="header-stats">
            <div class="header-stat">
                <div class="header-stat-value" id="total-markets">0</div>
                <div class="header-stat-label">Total Markets</div>
            </div>
            <div class="header-stat">
                <div class="header-stat-value" id="api-markets">0</div>
                <div class="header-stat-label">API Markets</div>
            </div>
            <div class="header-stat">
                <div class="header-stat-value" id="blockchain-markets">0</div>
                <div class="header-stat-label">Blockchain</div>
            </div>
            <div class="header-stat">
                <div class="header-stat-value" id="current-time">--:--:--</div>
                <div class="header-stat-label">UTC Time</div>
            </div>
        </div>
    </div>
    
    <div class="tabs">
        <div class="tab active" onclick="switchTab('overview')">📊 Overview</div>
        <div class="tab" onclick="switchTab('markets')">🎲 Live Markets</div>
        <div class="tab" onclick="switchTab('blockchain')">⛓️ Blockchain Data</div>
        <div class="tab" onclick="switchTab('sports')">🏆 Sports Analysis</div>
        <div class="tab" onclick="switchTab('odds')">📈 Odds Comparison</div>
    </div>
    
    <div class="main-container">
        <!-- Overview Tab -->
        <div id="overview-tab" class="tab-content active">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-card-value" id="active-markets">0</div>
                    <div class="stat-card-label">Active Markets</div>
                </div>
                <div class="stat-card">
                    <div class="stat-card-value" id="markets-with-odds">0</div>
                    <div class="stat-card-label">Markets with Odds</div>
                </div>
                <div class="stat-card">
                    <div class="stat-card-value" id="unique-sports">0</div>
                    <div class="stat-card-label">Sports Covered</div>
                </div>
                <div class="stat-card">
                    <div class="stat-card-value" id="data-sources">0</div>
                    <div class="stat-card-label">Data Sources</div>
                </div>
            </div>
            
            <div class="dashboard-grid">
                <div class="section">
                    <div class="section-title">🏆 Sport Distribution</div>
                    <div id="sport-distribution" class="section-content">
                        <div class="loading">Loading sport data...</div>
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">📡 Data Sources</div>
                    <div id="source-breakdown" class="section-content">
                        <div class="loading">Loading source data...</div>
                    </div>
                </div>
            </div>
            
            <div class="chart-container">
                <canvas id="sport-chart"></canvas>
            </div>
        </div>
        
        <!-- Markets Tab -->
        <div id="markets-tab" class="tab-content">
            <div class="section">
                <div class="section-title">🎲 Live Markets (Next 24 Hours)</div>
                <div id="live-markets" class="section-content">
                    <div class="loading">Loading markets...</div>
                </div>
            </div>
        </div>
        
        <!-- Blockchain Tab -->
        <div id="blockchain-tab" class="tab-content">
            <div class="section">
                <div class="section-title">⛓️ Blockchain Markets</div>
                <div id="blockchain-markets-list" class="section-content">
                    <div class="loading">Loading blockchain data...</div>
                </div>
            </div>
            
            <div class="dashboard-grid">
                <div class="section">
                    <div class="section-title">📊 Chain Distribution</div>
                    <div id="chain-distribution" class="section-content">
                        <div class="loading">Loading chain data...</div>
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">💰 Recent Blockchain Odds</div>
                    <div id="recent-blockchain-odds" class="section-content">
                        <div class="loading">Loading odds...</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Sports Tab -->
        <div id="sports-tab" class="tab-content">
            <div class="section">
                <div class="section-title">🏆 Sports Analysis</div>
                <table class="data-table" id="sports-table">
                    <thead>
                        <tr>
                            <th>Sport</th>
                            <th>Total Markets</th>
                            <th>API Markets</th>
                            <th>Blockchain Markets</th>
                            <th>Avg Odds</th>
                        </tr>
                    </thead>
                    <tbody id="sports-table-body">
                        <tr><td colspan="5" style="text-align:center">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Odds Tab -->
        <div id="odds-tab" class="tab-content">
            <div class="section">
                <div class="section-title">📈 Odds Comparison</div>
                <div id="odds-comparison" class="section-content">
                    <div class="loading">Loading odds comparison...</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        let currentTab = 'overview';
        
        // Tab switching
        function switchTab(tab) {
            currentTab = tab;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
            
            document.querySelector(`.tab:nth-child(${getTabIndex(tab)})`).classList.add('active');
            document.getElementById(tab + '-tab').classList.add('active');
            
            // Refresh data for the selected tab
            socket.emit('request_' + tab + '_data');
        }
        
        function getTabIndex(tab) {
            const tabs = ['overview', 'markets', 'blockchain', 'sports', 'odds'];
            return tabs.indexOf(tab) + 1;
        }
        
        // Socket event handlers
        socket.on('connect', () => {
            console.log('Connected to server');
            socket.emit('request_initial_data');
        });
        
        socket.on('stats_update', (data) => {
            document.getElementById('total-markets').textContent = data.total_markets.toLocaleString();
            document.getElementById('api-markets').textContent = data.api_markets.toLocaleString();
            document.getElementById('blockchain-markets').textContent = data.blockchain_markets.toLocaleString();
            document.getElementById('active-markets').textContent = data.active_markets.toLocaleString();
            document.getElementById('markets-with-odds').textContent = data.markets_with_odds.toLocaleString();
            document.getElementById('unique-sports').textContent = data.unique_sports;
            document.getElementById('data-sources').textContent = data.data_sources;
        });
        
        socket.on('sport_distribution', (data) => {
            const container = document.getElementById('sport-distribution');
            container.innerHTML = '';
            
            const maxCount = Math.max(...data.map(s => s.count));
            
            data.forEach(sport => {
                const percentage = (sport.count / maxCount) * 100;
                const bar = document.createElement('div');
                bar.className = 'sport-bar';
                bar.innerHTML = `
                    <div class="sport-name">${sport.sport}</div>
                    <div class="sport-bar-fill" style="width: ${percentage}%"></div>
                    <div class="sport-count">${sport.count.toLocaleString()}</div>
                `;
                container.appendChild(bar);
            });
            
            // Update chart
            updateSportChart(data);
        });
        
        socket.on('live_markets', (markets) => {
            const container = document.getElementById('live-markets');
            container.innerHTML = '';
            
            if (markets.length === 0) {
                container.innerHTML = '<div class="loading">No live markets found</div>';
                return;
            }
            
            markets.forEach(market => {
                const card = createMarketCard(market);
                container.appendChild(card);
            });
        });
        
        socket.on('blockchain_markets', (markets) => {
            const container = document.getElementById('blockchain-markets-list');
            container.innerHTML = '';
            
            markets.forEach(market => {
                const card = createMarketCard(market, true);
                container.appendChild(card);
            });
        });
        
        socket.on('sports_analysis', (data) => {
            const tbody = document.getElementById('sports-table-body');
            tbody.innerHTML = '';
            
            data.forEach(sport => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${sport.sport}</td>
                    <td>${sport.total.toLocaleString()}</td>
                    <td>${sport.api_markets.toLocaleString()}</td>
                    <td>${sport.blockchain_markets.toLocaleString()}</td>
                    <td>${sport.avg_odds ? sport.avg_odds.toFixed(3) : 'N/A'}</td>
                `;
                tbody.appendChild(row);
            });
        });
        
        function createMarketCard(market, isBlockchain = false) {
            const card = document.createElement('div');
            card.className = 'market-card';
            
            const sourceClass = isBlockchain ? 'blockchain-source' : 'api-source';
            const sourceIcon = isBlockchain ? '⛓️' : '📡';
            
            card.innerHTML = `
                <div class="market-header">
                    <div class="market-teams">${market.home_team} vs ${market.away_team}</div>
                    <div class="market-meta">
                        <div class="market-sport">${market.sport}</div>
                        <div class="market-source ${sourceClass}">${sourceIcon} ${market.source}</div>
                    </div>
                </div>
                ${market.odds ? createOddsHtml(market.odds) : '<div style="color:#666">No odds available</div>'}
            `;
            
            return card;
        }
        
        function createOddsHtml(odds) {
            return `
                <div class="odds-container">
                    ${odds.map(odd => `
                        <div class="odd-box">
                            <div class="odd-label">${odd.outcome.toUpperCase()}</div>
                            <div class="odd-value">${odd.decimal_odds.toFixed(3)}</div>
                            <div class="odd-american">${odd.american_odds > 0 ? '+' : ''}${odd.american_odds}</div>
                        </div>
                    `).join('')}
                </div>
            `;
        }
        
        // Sport chart
        let sportChart;
        
        function updateSportChart(data) {
            const ctx = document.getElementById('sport-chart').getContext('2d');
            
            if (sportChart) {
                sportChart.destroy();
            }
            
            sportChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: data.map(s => s.sport),
                    datasets: [{
                        data: data.map(s => s.count),
                        backgroundColor: [
                            '#00ff00', '#00ffff', '#ff9900', '#ff00ff',
                            '#ffff00', '#00ff99', '#9900ff', '#ff0099',
                            '#99ff00', '#0099ff', '#ff9999', '#99ff99'
                        ],
                        borderColor: '#222',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                color: '#fff',
                                font: {
                                    family: 'Consolas, Monaco, monospace'
                                }
                            }
                        },
                        title: {
                            display: true,
                            text: 'Market Distribution by Sport',
                            color: '#00ffff',
                            font: {
                                size: 16,
                                family: 'Consolas, Monaco, monospace'
                            }
                        }
                    }
                }
            });
        }
        
        // Update time
        setInterval(() => {
            const now = new Date();
            document.getElementById('current-time').textContent = 
                now.toUTCString().split(' ')[4];
        }, 1000);
        
        // Auto-refresh data every 30 seconds
        setInterval(() => {
            socket.emit('request_' + currentTab + '_data');
        }, 30000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main dashboard route"""
    return render_template_string(ENHANCED_DASHBOARD)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connection_status', {'status': 'connected'})

@socketio.on('request_initial_data')
def handle_initial_data():
    """Send initial data to client"""
    send_stats_update()
    send_sport_distribution()
    send_live_markets()

@socketio.on('request_overview_data')
def handle_overview_data():
    """Send overview data"""
    send_stats_update()
    send_sport_distribution()

@socketio.on('request_markets_data')
def handle_markets_data():
    """Send markets data"""
    send_live_markets()

@socketio.on('request_blockchain_data')
def handle_blockchain_data():
    """Send blockchain data"""
    send_blockchain_markets()

@socketio.on('request_sports_data')
def handle_sports_data():
    """Send sports analysis data"""
    send_sports_analysis()

@socketio.on('request_odds_data')
def handle_odds_data():
    """Send odds comparison data"""
    send_odds_comparison()

def send_stats_update():
    """Send overall statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get statistics
    stats = {}
    
    # Total markets
    cursor.execute("SELECT COUNT(*) FROM market")
    stats['total_markets'] = cursor.fetchone()[0]
    
    # API markets
    cursor.execute("SELECT COUNT(*) FROM market WHERE source = 'api_live_real'")
    stats['api_markets'] = cursor.fetchone()[0]
    
    # Blockchain markets
    cursor.execute("SELECT COUNT(*) FROM market WHERE source LIKE '%blockchain%'")
    stats['blockchain_markets'] = cursor.fetchone()[0]
    
    # Active markets
    cursor.execute("""
        SELECT COUNT(*) FROM market 
        WHERE is_finished = 0 
        AND (maturity_date > datetime('now') OR maturity_date IS NULL)
    """)
    stats['active_markets'] = cursor.fetchone()[0]
    
    # Markets with odds
    cursor.execute("SELECT COUNT(DISTINCT source_id) FROM odd")
    stats['markets_with_odds'] = cursor.fetchone()[0]
    
    # Unique sports
    cursor.execute("SELECT COUNT(DISTINCT sport) FROM market")
    stats['unique_sports'] = cursor.fetchone()[0]
    
    # Data sources
    cursor.execute("SELECT COUNT(DISTINCT source) FROM market")
    stats['data_sources'] = cursor.fetchone()[0]
    
    conn.close()
    
    emit('stats_update', stats, broadcast=True)

def send_sport_distribution():
    """Send sport distribution data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT sport, COUNT(*) as count
        FROM market
        GROUP BY sport
        ORDER BY count DESC
        LIMIT 12
    """)
    
    distribution = [{'sport': row[0], 'count': row[1]} for row in cursor.fetchall()]
    conn.close()
    
    emit('sport_distribution', distribution)

def send_live_markets():
    """Send live markets data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get markets for next 24 hours
    cursor.execute("""
        SELECT m.source_id, m.home_team, m.away_team, m.sport, m.source,
               m.maturity_date, m.league_name
        FROM market m
        WHERE m.is_finished = 0
        AND m.maturity_date > datetime('now')
        AND m.maturity_date < datetime('now', '+24 hours')
        ORDER BY m.maturity_date
        LIMIT 50
    """)
    
    markets = []
    for row in cursor.fetchall():
        market = {
            'source_id': row[0],
            'home_team': row[1],
            'away_team': row[2],
            'sport': row[3],
            'source': row[4],
            'maturity_date': row[5],
            'league_name': row[6]
        }
        
        # Get odds
        cursor.execute("""
            SELECT outcome, decimal_odds, american_odds
            FROM odd
            WHERE source_id = ?
            ORDER BY position
        """, (row[0],))
        
        odds = []
        for odd_row in cursor.fetchall():
            odds.append({
                'outcome': odd_row[0],
                'decimal_odds': odd_row[1],
                'american_odds': odd_row[2]
            })
        
        if odds:
            market['odds'] = odds
        
        markets.append(market)
    
    conn.close()
    emit('live_markets', markets)

def send_blockchain_markets():
    """Send blockchain markets data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT m.source_id, m.home_team, m.away_team, m.sport, m.source
        FROM market m
        WHERE m.source LIKE '%blockchain%'
        AND m.is_finished = 0
        ORDER BY m.updated_at DESC
        LIMIT 20
    """)
    
    markets = []
    for row in cursor.fetchall():
        market = {
            'source_id': row[0],
            'home_team': row[1],
            'away_team': row[2],
            'sport': row[3],
            'source': row[4]
        }
        
        # Get odds
        cursor.execute("""
            SELECT outcome, decimal_odds, american_odds
            FROM odd
            WHERE source_id = ?
            ORDER BY position
        """, (row[0],))
        
        odds = []
        for odd_row in cursor.fetchall():
            odds.append({
                'outcome': odd_row[0],
                'decimal_odds': odd_row[1],
                'american_odds': odd_row[2]
            })
        
        if odds:
            market['odds'] = odds
        
        markets.append(market)
    
    conn.close()
    emit('blockchain_markets', markets)

def send_sports_analysis():
    """Send sports analysis data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            m.sport,
            COUNT(DISTINCT m.source_id) as total,
            COUNT(DISTINCT CASE WHEN m.source = 'api_live_real' THEN m.source_id END) as api_markets,
            COUNT(DISTINCT CASE WHEN m.source LIKE '%blockchain%' THEN m.source_id END) as blockchain_markets,
            AVG(o.decimal_odds) as avg_odds
        FROM market m
        LEFT JOIN odd o ON m.source_id = o.source_id
        GROUP BY m.sport
        ORDER BY total DESC
    """)
    
    analysis = []
    for row in cursor.fetchall():
        analysis.append({
            'sport': row[0],
            'total': row[1],
            'api_markets': row[2],
            'blockchain_markets': row[3],
            'avg_odds': row[4]
        })
    
    conn.close()
    emit('sports_analysis', analysis)

def send_odds_comparison():
    """Send odds comparison data"""
    # This would compare odds between different sources
    # For now, sending a placeholder
    emit('odds_comparison', {'message': 'Odds comparison coming soon'})

def background_updates():
    """Background thread for periodic updates"""
    while True:
        try:
            with app.app_context():
                socketio.emit('stats_update', get_stats())
        except Exception as e:
            logger.error(f"Background update error: {e}")
        time.sleep(30)

def get_stats():
    """Helper to get stats (used by background thread)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    stats = {}
    cursor.execute("SELECT COUNT(*) FROM market")
    stats['total_markets'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM market WHERE source = 'api_live_real'")
    stats['api_markets'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM market WHERE source LIKE '%blockchain%'")
    stats['blockchain_markets'] = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) FROM market 
        WHERE is_finished = 0 
        AND (maturity_date > datetime('now') OR maturity_date IS NULL)
    """)
    stats['active_markets'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT source_id) FROM odd")
    stats['markets_with_odds'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT sport) FROM market")
    stats['unique_sports'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT source) FROM market")
    stats['data_sources'] = cursor.fetchone()[0]
    
    conn.close()
    return stats

if __name__ == '__main__':
    # Start background thread
    bg_thread = threading.Thread(target=background_updates, daemon=True)
    bg_thread.start()
    
    # Start server
    logger.info("Starting Enhanced Ominari Dashboard on http://localhost:8888")
    socketio.run(app, host='0.0.0.0', port=8888, debug=False)