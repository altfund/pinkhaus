#!/usr/bin/env python3
"""
Fixed Ominari Web Monitor - No expanding windows
Shows markets, signals, and stats without problematic charts
"""

import logging
from datetime import datetime, timezone
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO
from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import desc
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ominari-trading-system-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

FIXED_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Ominari Trading System</title>
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
        .header {
            background: #111;
            padding: 15px 20px;
            border-bottom: 2px solid #00ff00;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
        }
        .main-container {
            margin-top: 80px;
            padding: 20px;
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            max-width: 1600px;
            margin-left: auto;
            margin-right: auto;
        }
        .section {
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 20px;
        }
        .section-title {
            font-size: 1.2em;
            color: #00ffff;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }
        
        /* Markets */
        .markets-container {
            max-height: calc(100vh - 140px);
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
        .market-time {
            color: #888;
            font-size: 0.9em;
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
            background: #333;
            border-color: #00ff00;
        }
        .odd-label {
            font-size: 0.8em;
            color: #888;
            margin-bottom: 4px;
        }
        .odd-value {
            font-size: 1.3em;
            font-weight: bold;
            color: #00ffff;
        }
        .signal-box {
            background: #1a2a1a;
            border: 1px solid #2a4a2a;
            padding: 10px;
            margin-top: 10px;
            border-radius: 4px;
        }
        
        /* Stats */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        .stat-box {
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 20px;
            text-align: center;
            border-radius: 5px;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .stat-label {
            color: #888;
            font-size: 0.9em;
        }
        .profit { color: #00ff00; }
        .loss { color: #ff4444; }
        
        /* System info */
        .info-panel {
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .info-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #222;
        }
        .info-item:last-child {
            border-bottom: none;
        }
        .info-label {
            color: #888;
        }
        .info-value {
            color: #00ff00;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #111;
        }
        ::-webkit-scrollbar-thumb {
            background: #333;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #444;
        }
        
        /* WebSocket indicator */
        .ws-status {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            font-size: 0.9em;
        }
        .ws-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #ff4444;
        }
        .ws-indicator.connected {
            background: #00ff00;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="display: inline; margin-right: 20px;">🎯 Ominari Trading System</h1>
            <span class="ws-status">
                <span class="ws-indicator" id="ws-indicator"></span>
                <span id="ws-status">Connecting...</span>
            </span>
        </div>
        <div>
            <span id="current-time"></span>
        </div>
    </div>
    
    <div class="main-container">
        <!-- Markets Section -->
        <div class="section">
            <div class="section-title">⚽ Active Soccer Markets ({{ market_count }})</div>
            <div class="markets-container" id="markets-container">
                <!-- Populated by JavaScript -->
            </div>
        </div>
        
        <!-- Stats & Info Section -->
        <div>
            <!-- Account Stats -->
            <div class="section">
                <div class="section-title">📊 System Statistics</div>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value" id="total-markets">0</div>
                        <div class="stat-label">Active Markets</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="soccer-markets">0</div>
                        <div class="stat-label">Soccer Matches</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value profit">$10,000</div>
                        <div class="stat-label">Paper Balance</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="avg-odds">0.00</div>
                        <div class="stat-label">Avg Odds</div>
                    </div>
                </div>
                
                <div class="info-panel">
                    <div class="info-item">
                        <span class="info-label">Configuration</span>
                        <span class="info-value">V1 Mode</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Data Source</span>
                        <span class="info-value">Overtime API</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Update Frequency</span>
                        <span class="info-value">5 minutes</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Last Update</span>
                        <span class="info-value" id="last-update">-</span>
                    </div>
                </div>
            </div>
            
            <!-- Recent Activity -->
            <div class="section" style="margin-top: 20px;">
                <div class="section-title">📜 Recent Activity</div>
                <div id="activity-log" style="max-height: 300px; overflow-y: auto;">
                    <!-- Populated by JavaScript -->
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            updateTime();
            setInterval(updateTime, 1000);
            loadData();
            
            // Socket.IO handlers
            socket.on('connect', function() {
                document.getElementById('ws-indicator').classList.add('connected');
                document.getElementById('ws-status').textContent = 'Connected';
            });
            
            socket.on('disconnect', function() {
                document.getElementById('ws-indicator').classList.remove('connected');
                document.getElementById('ws-status').textContent = 'Disconnected';
            });
            
            socket.on('update', function(data) {
                updateDashboard(data);
            });
        });
        
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = 
                now.toLocaleString('en-US', { 
                    dateStyle: 'short', 
                    timeStyle: 'medium',
                    hour12: false 
                });
        }
        
        function loadData() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => updateDashboard(data));
        }
        
        function updateDashboard(data) {
            // Update stats
            if (data.stats) {
                document.getElementById('total-markets').textContent = data.stats.total_active || 0;
                document.getElementById('soccer-markets').textContent = data.stats.soccer_count || 0;
                document.getElementById('avg-odds').textContent = 
                    (data.stats.avg_odds || 0).toFixed(2);
            }
            
            // Update markets
            if (data.markets) {
                updateMarkets(data.markets);
            }
            
            // Update activity
            if (data.activity) {
                updateActivity(data.activity);
            }
            
            // Update last update time
            document.getElementById('last-update').textContent = 
                new Date().toLocaleTimeString('en-US', { hour12: false });
        }
        
        function updateMarkets(markets) {
            const container = document.getElementById('markets-container');
            container.innerHTML = markets.map(market => `
                <div class="market-card">
                    <div class="market-header">
                        <div class="market-teams">${market.home_team} vs ${market.away_team}</div>
                        <div class="market-time">${market.time_until}</div>
                    </div>
                    
                    <div class="odds-row">
                        <div class="odd-box">
                            <div class="odd-label">Home</div>
                            <div class="odd-value">${market.home_odds?.toFixed(2) || '-'}</div>
                        </div>
                        <div class="odd-box">
                            <div class="odd-label">Draw</div>
                            <div class="odd-value">${market.draw_odds?.toFixed(2) || '-'}</div>
                        </div>
                        <div class="odd-box">
                            <div class="odd-label">Away</div>
                            <div class="odd-value">${market.away_odds?.toFixed(2) || '-'}</div>
                        </div>
                    </div>
                    
                    ${market.signal ? `
                        <div class="signal-box">
                            📡 ${market.signal}
                        </div>
                    ` : ''}
                </div>
            `).join('');
        }
        
        function updateActivity(activity) {
            const container = document.getElementById('activity-log');
            container.innerHTML = activity.map(item => `
                <div style="padding: 8px; border-bottom: 1px solid #222;">
                    <span style="color: #888; font-size: 0.8em;">${item.time}</span>
                    <div>${item.message}</div>
                </div>
            `).join('');
        }
        
        // Auto-refresh every 30 seconds
        setInterval(loadData, 30000);
    </script>
</body>
</html>
"""

def get_market_data():
    """Get market data with proper stats."""
    markets = []
    stats = {
        'total_active': 0,
        'soccer_count': 0,
        'avg_odds': 0
    }
    
    try:
        with db_manager.get_db_session() as db:
            # Get active soccer markets
            active_markets = db.query(Market).filter(
                Market.sport == 'Soccer',
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).order_by(Market.maturity_date).limit(100).all()
            
            stats['soccer_count'] = len(active_markets)
            all_odds = []
            
            for market in active_markets:
                # Get latest odds
                odds = db.query(Odd).filter(
                    Odd.source_id == market.source_id
                ).order_by(desc(Odd.updated_at)).limit(3).all()
                
                home_odds = None
                draw_odds = None  
                away_odds = None
                
                for odd in odds:
                    if odd.outcome == 'option_1' and odd.decimal_odds:
                        home_odds = odd.decimal_odds
                        all_odds.append(odd.decimal_odds)
                    elif odd.outcome == 'option_3' and odd.decimal_odds:
                        draw_odds = odd.decimal_odds
                        all_odds.append(odd.decimal_odds)
                    elif odd.outcome == 'option_2' and odd.decimal_odds:
                        away_odds = odd.decimal_odds
                        all_odds.append(odd.decimal_odds)
                
                # Calculate time until
                time_until = "Started"
                if market.maturity_date:
                    delta = market.maturity_date - datetime.now(timezone.utc)
                    if delta.total_seconds() > 0:
                        hours = int(delta.total_seconds() // 3600)
                        mins = int((delta.total_seconds() % 3600) // 60)
                        if hours > 24:
                            time_until = f"{hours // 24}d {hours % 24}h"
                        elif hours > 0:
                            time_until = f"{hours}h {mins}m"
                        else:
                            time_until = f"{mins}m"
                
                # Simple signal
                signal = None
                if home_odds and draw_odds and away_odds:
                    total = (1/home_odds + 1/draw_odds + 1/away_odds)
                    if total > 1.05:
                        margin = (total - 1) * 100
                        signal = f"Margin: {margin:.1f}%"
                
                markets.append({
                    'home_team': market.home_team,
                    'away_team': market.away_team,
                    'time_until': time_until,
                    'home_odds': home_odds,
                    'draw_odds': draw_odds,
                    'away_odds': away_odds,
                    'signal': signal
                })
            
            # Calculate stats
            stats['total_active'] = db.query(Market).filter(
                Market.is_finished == False
            ).count()
            
            if all_odds:
                stats['avg_odds'] = sum(all_odds) / len(all_odds)
                
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
    
    return markets, stats

def get_activity():
    """Get recent activity messages."""
    return [
        {'time': datetime.now().strftime("%H:%M"), 'message': 'System running normally'},
        {'time': datetime.now().strftime("%H:%M"), 'message': 'Data updated from Overtime API'}
    ]

@app.route('/')
def index():
    """Main dashboard."""
    markets, stats = get_market_data()
    return render_template_string(FIXED_HTML, 
        market_count=stats['soccer_count']
    )

@app.route('/api/status')
def api_status():
    """API endpoint for dashboard data."""
    markets, stats = get_market_data()
    return jsonify({
        'markets': markets,
        'stats': stats,
        'activity': get_activity(),
        'timestamp': datetime.now().isoformat()
    })

# Background updater
def background_updates():
    """Send updates to connected clients."""
    while True:
        time.sleep(30)  # Update every 30 seconds
        try:
            markets, stats = get_market_data()
            socketio.emit('update', {
                'markets': markets[:20],  # Top 20 for updates
                'stats': stats,
                'activity': get_activity()
            })
        except Exception as e:
            logger.error(f"Background update error: {e}")

# Start background thread
update_thread = threading.Thread(target=background_updates, daemon=True)
update_thread.start()

if __name__ == '__main__':
    print("🚀 Fixed Ominari Monitor Starting...")
    print("📊 Dashboard: http://localhost:8891")
    print("✨ No expanding windows!")
    
    socketio.run(app, host='0.0.0.0', port=8891, debug=False, allow_unsafe_werkzeug=True)