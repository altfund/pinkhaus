#!/usr/bin/env python3
"""
Enhanced Ominari Web Monitor - Combines best of both dashboards
Shows market data, signals, positions, trades, and performance
"""

import logging
from datetime import datetime, timezone
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import pandas as pd
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

# Global state
recent_logs = []
log_lock = threading.Lock()

ENHANCED_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Ominari Trading System - Enhanced Monitor</title>
    <meta charset="utf-8">
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
            padding: 10px 20px;
            border-bottom: 2px solid #00ff00;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .ticker-section {
            background: #0f0f0f;
            padding: 10px;
            border-bottom: 1px solid #333;
            overflow: hidden;
            height: 40px;
        }
        .ticker {
            display: flex;
            animation: scroll-left 30s linear infinite;
            white-space: nowrap;
        }
        .ticker-item {
            padding: 0 20px;
            border-right: 1px solid #333;
        }
        @keyframes scroll-left {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        
        .main-container {
            display: grid;
            grid-template-columns: 1fr 1fr 400px;
            gap: 20px;
            padding: 20px;
            height: calc(100vh - 100px);
            overflow: hidden;
        }
        
        /* Market Section */
        .markets-section {
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 15px;
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
        }
        .market-time {
            color: #888;
            font-size: 0.9em;
        }
        .market-odds {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 10px 0;
        }
        .odd-box {
            background: #222;
            border: 1px solid #444;
            padding: 8px;
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
            font-size: 1.2em;
            font-weight: bold;
            color: #00ffff;
        }
        .market-signal {
            background: #1a2a1a;
            border: 1px solid #2a4a2a;
            padding: 10px;
            margin-top: 10px;
            border-radius: 4px;
        }
        .signal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .signal-strength {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .signal-bar {
            width: 100px;
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
        }
        .signal-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff4444, #ffaa00, #00ff00);
            transition: width 0.3s;
        }
        .signal-recommendation {
            margin-top: 8px;
            padding: 5px 10px;
            background: rgba(0, 255, 0, 0.1);
            border: 1px solid rgba(0, 255, 0, 0.3);
            border-radius: 4px;
            font-size: 0.9em;
        }
        
        /* Positions & Performance Section */
        .positions-section {
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 15px;
            overflow-y: auto;
        }
        .position-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 5px;
            padding: 12px;
            margin-bottom: 10px;
        }
        .position-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .position-pnl {
            font-size: 1.2em;
            font-weight: bold;
        }
        .profit { color: #00ff00; }
        .loss { color: #ff4444; }
        
        /* Right Panel */
        .right-panel {
            display: flex;
            flex-direction: column;
            gap: 20px;
            height: 100%;
            overflow: hidden;
        }
        
        /* Account Stats */
        .account-card {
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 15px;
        }
        .stat-box {
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .stat-label {
            font-size: 0.9em;
            color: #888;
        }
        
        /* Performance Chart */
        .performance-card {
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 20px;
        }
        
        /* Activity Log */
        .activity-card {
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 15px;
            max-height: 300px;
            overflow-y: auto;
        }
        .activity-item {
            padding: 8px;
            border-bottom: 1px solid #222;
            font-size: 0.9em;
        }
        .activity-time {
            color: #666;
            font-size: 0.8em;
        }
        
        /* Section titles */
        .section-title {
            font-size: 1.2em;
            color: #00ffff;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }
        
        /* Trade execution */
        .trade-button {
            background: #00ff00;
            color: #000;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
        }
        .trade-button:hover {
            background: #00cc00;
            transform: scale(1.05);
        }
        .trade-button:disabled {
            background: #444;
            color: #888;
            cursor: not-allowed;
        }
        
        /* WebSocket status */
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
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
            <span style="margin-right: 20px;">⏰ <span id="current-time"></span></span>
            <span>📊 V1 Mode: API + Blockchain</span>
        </div>
    </div>
    
    <div class="ticker-section">
        <div class="ticker" id="ticker">
            <!-- Populated by JavaScript -->
        </div>
    </div>
    
    <div class="main-container">
        <!-- Markets with Signals -->
        <div class="markets-section">
            <div class="section-title">🏆 Active Markets & Signals</div>
            <div id="markets-container">
                <!-- Populated by JavaScript -->
            </div>
        </div>
        
        <!-- Positions & Trades -->
        <div class="positions-section">
            <div class="section-title">💼 Positions & Recent Trades</div>
            
            <!-- Open Positions -->
            <div style="margin-bottom: 20px;">
                <h3 style="color: #00ff00; margin-bottom: 10px;">Open Positions</h3>
                <div id="positions-container">
                    <!-- Populated by JavaScript -->
                </div>
            </div>
            
            <!-- Recent Trades -->
            <div>
                <h3 style="color: #00ff00; margin-bottom: 10px;">Recent Trades</h3>
                <div id="trades-container">
                    <!-- Populated by JavaScript -->
                </div>
            </div>
        </div>
        
        <!-- Right Panel -->
        <div class="right-panel">
            <!-- Account Stats -->
            <div class="account-card">
                <div class="section-title">📈 Account Overview</div>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value" id="balance">$10,000</div>
                        <div class="stat-label">Balance</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="daily-pnl">+0.00%</div>
                        <div class="stat-label">Daily P&L</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="open-positions">0</div>
                        <div class="stat-label">Positions</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="win-rate">0%</div>
                        <div class="stat-label">Win Rate</div>
                    </div>
                </div>
            </div>
            
            <!-- Performance Chart -->
            <div class="performance-card">
                <div class="section-title">📊 Performance</div>
                <div style="position: relative; height: 300px; width: 100%;">
                    <canvas id="performance-chart" width="400" height="300"></canvas>
                </div>
            </div>
            
            <!-- Activity Log -->
            <div class="activity-card">
                <div class="section-title">📜 Activity Log</div>
                <div id="activity-log">
                    <!-- Populated by JavaScript -->
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Initialize Socket.IO
        const socket = io();
        let performanceChart = null;
        
        // Initialize on load
        document.addEventListener('DOMContentLoaded', function() {
            initializeChart();
            updateTime();
            setInterval(updateTime, 1000);
            
            // Initial data load
            fetch('/api/status')
                .then(response => response.json())
                .then(data => updateDashboard(data));
            
            // Socket.IO event handlers
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
                now.toLocaleTimeString('en-US', { hour12: false });
        }
        
        function initializeChart() {
            const ctx = document.getElementById('performance-chart').getContext('2d');
            performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Equity Curve',
                        data: [],
                        borderColor: '#00ff00',
                        backgroundColor: 'rgba(0, 255, 0, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: false,  // Disable responsive to prevent expansion
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: { 
                            grid: { color: '#333' },
                            ticks: { color: '#888' }
                        },
                        y: { 
                            grid: { color: '#333' },
                            ticks: { 
                                color: '#888',
                                callback: function(value) {
                                    return '$' + value.toLocaleString();
                                }
                            }
                        }
                    }
                }
            });
        }
        
        function updateDashboard(data) {
            // Update account stats
            updateAccountStats(data.account);
            
            // Update markets with signals
            updateMarkets(data.markets);
            
            // Update positions
            updatePositions(data.positions);
            
            // Update trades
            updateTrades(data.trades);
            
            // Update activity log
            updateActivity(data.activity);
            
            // Update ticker
            updateTicker(data.ticker);
            
            // Update performance chart
            if (data.performance) {
                updatePerformanceChart(data.performance);
            }
        }
        
        function updateAccountStats(account) {
            if (!account) return;
            
            document.getElementById('balance').textContent = 
                '$' + (account.balance || 0).toLocaleString();
            
            const pnlElement = document.getElementById('daily-pnl');
            const pnl = account.daily_pnl || 0;
            pnlElement.textContent = (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%';
            pnlElement.className = 'stat-value ' + (pnl >= 0 ? 'profit' : 'loss');
            
            document.getElementById('open-positions').textContent = account.positions || 0;
            document.getElementById('win-rate').textContent = 
                (account.win_rate || 0).toFixed(1) + '%';
        }
        
        function updateMarkets(markets) {
            if (!markets) return;
            
            const container = document.getElementById('markets-container');
            container.innerHTML = markets.map(market => `
                <div class="market-card">
                    <div class="market-header">
                        <div class="market-teams">${market.home_team} vs ${market.away_team}</div>
                        <div class="market-time">${formatTime(market.maturity_date)}</div>
                    </div>
                    
                    <div class="market-odds">
                        <div class="odd-box" onclick="placeBet('${market.id}', 'home', ${market.home_odds})">
                            <div class="odd-label">Home</div>
                            <div class="odd-value">${market.home_odds?.toFixed(2) || '-'}</div>
                        </div>
                        <div class="odd-box" onclick="placeBet('${market.id}', 'draw', ${market.draw_odds})">
                            <div class="odd-label">Draw</div>
                            <div class="odd-value">${market.draw_odds?.toFixed(2) || '-'}</div>
                        </div>
                        <div class="odd-box" onclick="placeBet('${market.id}', 'away', ${market.away_odds})">
                            <div class="odd-label">Away</div>
                            <div class="odd-value">${market.away_odds?.toFixed(2) || '-'}</div>
                        </div>
                    </div>
                    
                    ${market.signal ? `
                        <div class="market-signal">
                            <div class="signal-header">
                                <div>📡 Signal: ${market.signal.provider}</div>
                                <div class="signal-strength">
                                    <span>${(market.signal.confidence * 100).toFixed(0)}%</span>
                                    <div class="signal-bar">
                                        <div class="signal-fill" style="width: ${market.signal.confidence * 100}%"></div>
                                    </div>
                                </div>
                            </div>
                            ${market.signal.recommendation ? `
                                <div class="signal-recommendation">
                                    💡 ${market.signal.recommendation}
                                    ${market.signal.edge > 0 ? 
                                        `<span style="color: #00ff00; margin-left: 10px;">
                                            Edge: +${(market.signal.edge * 100).toFixed(1)}%
                                        </span>` : ''}
                                </div>
                            ` : ''}
                        </div>
                    ` : ''}
                </div>
            `).join('');
        }
        
        function updatePositions(positions) {
            const container = document.getElementById('positions-container');
            if (!positions || positions.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: #666; padding: 20px;">No open positions</div>';
                return;
            }
            
            container.innerHTML = positions.map(pos => `
                <div class="position-card">
                    <div class="position-header">
                        <div>
                            <strong>${pos.market}</strong>
                            <div style="color: #888; font-size: 0.9em;">${pos.selection}</div>
                        </div>
                        <div class="${pos.pnl >= 0 ? 'profit' : 'loss'} position-pnl">
                            ${pos.pnl >= 0 ? '+' : ''}${pos.pnl.toFixed(2)}%
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                        <span>Stake: $${pos.stake}</span>
                        <span>Odds: ${pos.odds}</span>
                        <button class="trade-button" style="padding: 4px 12px; font-size: 0.9em;" 
                                onclick="closePosition('${pos.id}')">
                            Close
                        </button>
                    </div>
                </div>
            `).join('');
        }
        
        function updateTrades(trades) {
            const container = document.getElementById('trades-container');
            if (!trades || trades.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: #666; padding: 20px;">No recent trades</div>';
                return;
            }
            
            container.innerHTML = trades.slice(0, 5).map(trade => `
                <div class="activity-item">
                    <div style="display: flex; justify-content: space-between;">
                        <span>${trade.action} ${trade.market}</span>
                        <span class="${trade.pnl >= 0 ? 'profit' : 'loss'}">
                            ${trade.pnl >= 0 ? '+' : ''}$${Math.abs(trade.pnl).toFixed(2)}
                        </span>
                    </div>
                    <div class="activity-time">${formatTime(trade.timestamp)}</div>
                </div>
            `).join('');
        }
        
        function updateActivity(activity) {
            const container = document.getElementById('activity-log');
            if (!activity || activity.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: #666;">No recent activity</div>';
                return;
            }
            
            container.innerHTML = activity.slice(0, 10).map(item => `
                <div class="activity-item">
                    <div>${item.message}</div>
                    <div class="activity-time">${formatTime(item.timestamp)}</div>
                </div>
            `).join('');
        }
        
        function updateTicker(ticker) {
            if (!ticker) return;
            
            const container = document.getElementById('ticker');
            container.innerHTML = ticker.map(item => `
                <div class="ticker-item">
                    <span style="color: #888;">${item.label}:</span>
                    <span style="color: ${item.color || '#00ff00'}; margin-left: 5px;">
                        ${item.value}
                    </span>
                </div>
            `).join('');
        }
        
        function updatePerformanceChart(data) {
            if (!performanceChart || !data) return;
            
            performanceChart.data.labels = data.dates;
            performanceChart.data.datasets[0].data = data.values;
            performanceChart.update();
        }
        
        function formatTime(timestamp) {
            if (!timestamp) return '-';
            const date = new Date(timestamp);
            const now = new Date();
            const diff = date - now;
            
            if (diff < 0) return 'Started';
            
            const hours = Math.floor(diff / (1000 * 60 * 60));
            const mins = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
            
            if (hours > 24) {
                return date.toLocaleDateString();
            } else if (hours > 0) {
                return `${hours}h ${mins}m`;
            } else {
                return `${mins}m`;
            }
        }
        
        function placeBet(marketId, selection, odds) {
            // Send bet to backend
            socket.emit('place_bet', {
                market_id: marketId,
                selection: selection,
                odds: odds,
                stake: 100  // Default stake
            });
        }
        
        function closePosition(positionId) {
            socket.emit('close_position', { position_id: positionId });
        }
        
        // Auto-refresh every 5 seconds
        setInterval(() => {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => updateDashboard(data));
        }, 5000);
    </script>
</body>
</html>
"""

# Data functions
def get_account_data():
    """Get account overview data."""
    try:
        with db_manager.get_db_session() as db:
            # Get account balance (from paper trading or config)
            balance = 10000.0  # Default paper trading balance
            
            # Count positions (simplified for now)
            positions = 0  # Paper trading positions would go here
            
            # Calculate daily P&L
            today = datetime.now(timezone.utc).date()
            daily_pnl = 0.0
            
            # Calculate win rate (simplified for now)
            win_rate = 0.0  # Would come from paper trading history
            
            return {
                'balance': balance,
                'positions': positions,
                'daily_pnl': daily_pnl,
                'win_rate': win_rate
            }
    except Exception as e:
        logger.error(f"Error getting account data: {e}")
        return {
            'balance': 10000.0,
            'positions': 0,
            'daily_pnl': 0.0,
            'win_rate': 0.0
        }

def get_active_markets_with_signals():
    """Get active markets with signal analysis."""
    markets = []
    
    try:
        with db_manager.get_db_session() as db:
            # Get active soccer markets
            active_markets = db.query(Market).filter(
                Market.sport == 'Soccer',
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).order_by(Market.maturity_date).limit(100).all()  # Show more markets
            
            for market in active_markets:
                # Get latest odds
                home_odd = db.query(Odd).filter(
                    Odd.source_id == market.source_id,
                    Odd.outcome == 'option_1'
                ).order_by(desc(Odd.updated_at)).first()
                
                draw_odd = db.query(Odd).filter(
                    Odd.source_id == market.source_id,
                    Odd.outcome == 'option_3'
                ).order_by(desc(Odd.updated_at)).first()
                
                away_odd = db.query(Odd).filter(
                    Odd.source_id == market.source_id,
                    Odd.outcome == 'option_2'
                ).order_by(desc(Odd.updated_at)).first()
                
                # Calculate simple signal (implied probability analysis)
                signal = None
                if home_odd and draw_odd and away_odd:
                    home_prob = 1 / home_odd.decimal_odds if home_odd.decimal_odds else 0
                    draw_prob = 1 / draw_odd.decimal_odds if draw_odd.decimal_odds else 0
                    away_prob = 1 / away_odd.decimal_odds if away_odd.decimal_odds else 0
                    
                    total_prob = home_prob + draw_prob + away_prob
                    
                    # Find value bets (where implied prob < fair prob)
                    if total_prob > 1.05:  # Bookmaker margin > 5%
                        fair_probs = {
                            'home': home_prob / total_prob,
                            'draw': draw_prob / total_prob,
                            'away': away_prob / total_prob
                        }
                        
                        # Check for value
                        best_edge = 0
                        best_selection = None
                        
                        if home_prob < fair_probs['home'] * 0.95:
                            edge = (fair_probs['home'] - home_prob) / home_prob
                            if edge > best_edge:
                                best_edge = edge
                                best_selection = 'Home'
                        
                        if best_edge > 0.02:  # 2% edge threshold
                            signal = {
                                'provider': 'ValueFinder',
                                'confidence': min(best_edge * 10, 1.0),
                                'recommendation': f'Back {best_selection} - Value bet detected',
                                'edge': best_edge
                            }
                
                markets.append({
                    'id': market.source_id,
                    'home_team': market.home_team,
                    'away_team': market.away_team,
                    'maturity_date': market.maturity_date.isoformat() if market.maturity_date else None,
                    'home_odds': home_odd.decimal_odds if home_odd else None,
                    'draw_odds': draw_odd.decimal_odds if draw_odd else None,
                    'away_odds': away_odd.decimal_odds if away_odd else None,
                    'signal': signal
                })
    
    except Exception as e:
        logger.error(f"Error getting markets: {e}")
    
    return markets

def get_positions():
    """Get open positions."""
    positions = []
    
    try:
        # For now, return mock data - integrate with paper trading
        # In production, this would query the paper_trades database
        positions = [
            # Example position structure
            # {
            #     'id': 'pos_1',
            #     'market': 'Man City vs Liverpool',
            #     'selection': 'Home Win',
            #     'stake': 100,
            #     'odds': 1.85,
            #     'pnl': 5.2
            # }
        ]
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
    
    return positions

def get_recent_trades():
    """Get recent trade history."""
    # Simplified for now - would integrate with paper trading
    return []

def get_activity_log():
    """Get recent system activity."""
    with log_lock:
        return recent_logs[-10:]

def get_ticker_data():
    """Get ticker information."""
    try:
        with db_manager.get_db_session() as db:
            active_markets = db.query(func.count(Market.id)).filter(
                Market.is_finished == False
            ).scalar() or 0
            
            total_volume = 0  # Would calculate from bets
            
            return [
                {'label': 'Active Markets', 'value': str(active_markets), 'color': '#00ff00'},
                {'label': 'Volume Today', 'value': f'${total_volume:,.0f}', 'color': '#00ffff'},
                {'label': 'System Status', 'value': 'Online', 'color': '#00ff00'},
                {'label': 'Data Feed', 'value': 'Live', 'color': '#00ff00'}
            ]
    except:
        return [
            {'label': 'System Status', 'value': 'Error', 'color': '#ff4444'}
        ]

def get_performance_data():
    """Get performance chart data."""
    # Mock data for now - would query paper trading history
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    values = [10000]  # Starting balance
    
    # Simulate some performance
    for i in range(1, 30):
        change = (pd.Series([0]).random.randn() * 0.02 + 0.001)[0]  # Small daily changes
        values.append(values[-1] * (1 + change))
    
    return {
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'values': values
    }

# Routes
@app.route('/')
def index():
    """Main dashboard."""
    return render_template_string(ENHANCED_HTML)

@app.route('/api/status')
def api_status():
    """Get complete dashboard status."""
    return jsonify({
        'account': get_account_data(),
        'markets': get_active_markets_with_signals(),
        'positions': get_positions(),
        'trades': get_recent_trades(),
        'activity': get_activity_log(),
        'ticker': get_ticker_data(),
        'performance': get_performance_data(),
        'timestamp': datetime.now().isoformat()
    })

# SocketIO events
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'data': 'Connected to Ominari Trading System'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('place_bet')
def handle_place_bet(data):
    """Handle bet placement from UI."""
    logger.info(f"Bet request: {data}")
    # TODO: Integrate with paper trading engine
    emit('bet_result', {'status': 'success', 'message': 'Bet placed (paper trading)'})

@socketio.on('close_position')
def handle_close_position(data):
    """Handle position closure."""
    logger.info(f"Close position: {data}")
    # TODO: Integrate with paper trading engine
    emit('position_closed', {'status': 'success', 'position_id': data['position_id']})

# Background update thread
def background_updater():
    """Send updates to connected clients."""
    while True:
        time.sleep(5)  # Update every 5 seconds
        try:
            status = {
                'account': get_account_data(),
                'markets': get_active_markets_with_signals()[:5],  # Top 5 markets
                'positions': get_positions(),
                'ticker': get_ticker_data(),
                'timestamp': datetime.now().isoformat()
            }
            socketio.emit('update', status)
        except Exception as e:
            logger.error(f"Background update error: {e}")

# Start background thread
update_thread = threading.Thread(target=background_updater, daemon=True)
update_thread.start()

if __name__ == '__main__':
    print("🚀 Enhanced Ominari Monitor Starting...")
    print("📊 Dashboard: http://localhost:8888")
    print("🔌 WebSocket enabled for live updates")
    print("📡 Showing markets with signal analysis")
    
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)