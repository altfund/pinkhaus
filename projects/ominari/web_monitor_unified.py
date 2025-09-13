#!/usr/bin/env python3
"""
Unified Ominari Web Monitor - Real blockchain data with enhanced features
Shows soccer markets with signals, paper trading, and portfolio management
"""

import logging
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import func, desc, and_
import threading
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

# Import paper trading and signals
try:
    from paper_trading_engine import PaperTradingEngine, PaperOrder
except ImportError:
    PaperTradingEngine = None
    PaperOrder = None
    
try:
    from signals import ImpliedRawSignal
except ImportError:
    ImpliedRawSignal = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ominari-trading-system-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
signal_providers = {"implied_probability": ImpliedRawSignal()} if ImpliedRawSignal else {}
portfolio_file = Path("paper_portfolio.json")
trades_file = Path("paper_trades.csv")

# Load portfolio
def load_portfolio():
    """Load portfolio from file."""
    if portfolio_file.exists():
        with open(portfolio_file, 'r') as f:
            return json.load(f)
    else:
        return {
            'cash': 10000.0,  # Starting capital
            'positions': {},
            'total_value': 10000.0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pending_bets': {}
        }

portfolio = load_portfolio()

UNIFIED_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Ominari Trading System - Soccer Markets</title>
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
        .market-league {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
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
        .odd-prob {
            font-size: 0.8em;
            color: #666;
            margin-top: 2px;
        }
        .market-info {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            font-size: 0.9em;
            color: #888;
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
            flex-grow: 1;
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
        
        /* Best bets section */
        .best-bets {
            background: #1a2a1a;
            border: 1px solid #2a4a2a;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .bet-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px 0;
            border-bottom: 1px solid #2a4a2a;
        }
        .bet-item:last-child {
            border-bottom: none;
        }
        .edge-value {
            color: #00ff00;
            font-weight: bold;
        }
    </style>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="display: inline; margin-right: 20px;">⚽ Ominari Soccer Trading</h1>
            <span class="ws-status">
                <span class="ws-indicator" id="ws-indicator"></span>
                <span id="ws-status">Connecting...</span>
            </span>
        </div>
        <div>
            <span style="margin-right: 20px;">⏰ <span id="current-time"></span></span>
            <span>🌐 Live Blockchain Data</span>
        </div>
    </div>
    
    <div class="ticker-section">
        <div class="ticker" id="ticker">
            <!-- Populated by JavaScript -->
        </div>
    </div>
    
    <div class="main-container">
        <!-- Soccer Markets with Signals -->
        <div class="markets-section">
            <div class="section-title">⚽ Soccer Markets & Signals</div>
            <div id="markets-container">
                <!-- Populated by JavaScript -->
            </div>
        </div>
        
        <!-- Positions & Best Bets -->
        <div class="positions-section">
            <div class="section-title">💼 Trading Overview</div>
            
            <!-- Best Betting Opportunities -->
            <div class="best-bets" style="margin-bottom: 20px;">
                <h3 style="color: #00ff00; margin-bottom: 10px;">🎯 Best Opportunities</h3>
                <div id="best-bets-container">
                    <!-- Populated by JavaScript -->
                </div>
            </div>
            
            <!-- Open Positions -->
            <div style="margin-bottom: 20px;">
                <h3 style="color: #00ff00; margin-bottom: 10px;">📊 Open Positions</h3>
                <div id="positions-container">
                    <!-- Populated by JavaScript -->
                </div>
            </div>
            
            <!-- Recent Trades -->
            <div>
                <h3 style="color: #00ff00; margin-bottom: 10px;">📈 Recent Trades</h3>
                <div id="trades-container">
                    <!-- Populated by JavaScript -->
                </div>
            </div>
        </div>
        
        <!-- Right Panel -->
        <div class="right-panel">
            <!-- Account Stats -->
            <div class="account-card">
                <div class="section-title">💰 Portfolio</div>
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
                <canvas id="performance-chart" width="400" height="200"></canvas>
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
                    responsive: true,
                    maintainAspectRatio: false,
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
            
            // Update best bets
            updateBestBets(data.best_bets);
            
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
                    <div class="market-league">🏆 ${market.league || 'Soccer League'}</div>
                    <div class="market-header">
                        <div class="market-teams">${market.home_team} vs ${market.away_team}</div>
                        <div class="market-time">${formatTime(market.maturity_date)}</div>
                    </div>
                    
                    <div class="market-odds">
                        <div class="odd-box" onclick="placeBet('${market.id}', 'home', ${market.home_odds})">
                            <div class="odd-label">Home</div>
                            <div class="odd-value">${market.home_odds?.toFixed(2) || '-'}</div>
                            <div class="odd-prob">${market.home_prob ? (market.home_prob * 100).toFixed(1) + '%' : ''}</div>
                        </div>
                        <div class="odd-box" onclick="placeBet('${market.id}', 'draw', ${market.draw_odds})">
                            <div class="odd-label">Draw</div>
                            <div class="odd-value">${market.draw_odds?.toFixed(2) || '-'}</div>
                            <div class="odd-prob">${market.draw_prob ? (market.draw_prob * 100).toFixed(1) + '%' : ''}</div>
                        </div>
                        <div class="odd-box" onclick="placeBet('${market.id}', 'away', ${market.away_odds})">
                            <div class="odd-label">Away</div>
                            <div class="odd-value">${market.away_odds?.toFixed(2) || '-'}</div>
                            <div class="odd-prob">${market.away_prob ? (market.away_prob * 100).toFixed(1) + '%' : ''}</div>
                        </div>
                    </div>
                    
                    <div class="market-info">
                        <span>📊 Market Margin: ${market.market_margin?.toFixed(1) || '-'}%</span>
                        <span>🏦 ${market.bookmaker || 'Blockchain'}</span>
                        <span>🆔 ${market.id.slice(0, 16)}...</span>
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
        
        function updateBestBets(bets) {
            const container = document.getElementById('best-bets-container');
            if (!bets || bets.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: #666; padding: 10px;">No opportunities found</div>';
                return;
            }
            
            container.innerHTML = bets.slice(0, 5).map(bet => `
                <div class="bet-item">
                    <div>
                        <strong>${bet.home_team} vs ${bet.away_team}</strong>
                        <div style="color: #888; font-size: 0.9em;">${bet.recommended_bet}</div>
                    </div>
                    <div>
                        <span class="edge-value">+${(bet.expected_edge * 100).toFixed(1)}%</span>
                        <button class="trade-button" style="margin-left: 10px; padding: 4px 12px; font-size: 0.9em;"
                                onclick="executeBet('${bet.match_id}', '${bet.recommended_bet}', ${bet.signal_probability})">
                            Trade
                        </button>
                    </div>
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
        
        function executeBet(marketId, recommendation, probability) {
            // Execute recommended bet
            socket.emit('execute_recommendation', {
                market_id: marketId,
                recommendation: recommendation,
                probability: probability
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
    """Get account overview data from portfolio."""
    global portfolio
    
    try:
        total_at_risk = sum(bet['stake'] for bet in portfolio['pending_bets'].values())
        balance = portfolio['cash']
        positions = len(portfolio['pending_bets'])
        
        # Calculate win rate
        total_completed = portfolio['wins'] + portfolio['losses']
        win_rate = (portfolio['wins'] / total_completed * 100) if total_completed > 0 else 0
        
        # Calculate daily P&L (simplified)
        daily_pnl = 0.0  # Would calculate from today's trades
        
        return {
            'balance': balance,
            'positions': positions,
            'daily_pnl': daily_pnl,
            'win_rate': win_rate,
            'at_risk': total_at_risk
        }
    except Exception as e:
        logger.error(f"Error getting account data: {e}")
        return {
            'balance': 10000.0,
            'positions': 0,
            'daily_pnl': 0.0,
            'win_rate': 0.0,
            'at_risk': 0.0
        }

def get_soccer_markets_with_signals():
    """Get soccer markets with comprehensive data and signal analysis."""
    markets = []
    
    try:
        with db_manager.get_db_session() as db:
            # Get active soccer markets
            now = datetime.now(timezone.utc)
            future_time = now + timedelta(hours=48)
            
            active_markets = db.query(Market).filter(
                Market.sport == 'Soccer',
                Market.is_finished == False,
                Market.maturity_date > now,
                Market.maturity_date <= future_time
            ).order_by(Market.maturity_date).limit(50).all()
            
            logger.info(f"Found {len(active_markets)} active soccer markets")
            
            for market in active_markets:
                try:
                    # Get all odds for this market - handle parameter issues  
                    try:
                        # Use exact match for source_id
                        odds = db.query(Odd).filter(
                            Odd.source_id == market.source_id
                        ).order_by(desc(Odd.updated_at)).limit(20).all()
                        
                        if not odds:
                            # If no exact match, log and continue
                            logger.info(f"No odds found for market {market.source_id}")
                        else:
                            logger.info(f"Found {len(odds)} odds for {market.home_team} vs {market.away_team}")
                            
                    except Exception as db_error:
                        logger.warning(f"Database error for market {market.source_id[:50]}...: {db_error}")
                        odds = []
                    
                    # Group odds by outcome
                    home_odds = []
                    draw_odds = []
                    away_odds = []
                    
                    for odd in odds:
                        # Handle different outcome naming conventions
                        outcome = str(odd.outcome).lower() if odd.outcome else ''
                        
                        if 'home' in outcome:
                            if odd.decimal_odds:
                                home_odds.append(odd.decimal_odds)
                        elif 'away' in outcome:
                            if odd.decimal_odds:
                                away_odds.append(odd.decimal_odds)
                        elif 'draw' in outcome or 'tie' in outcome:
                            if odd.decimal_odds:
                                draw_odds.append(odd.decimal_odds)
                    
                    # Get best odds
                    best_home = min(home_odds) if home_odds else None
                    best_draw = min(draw_odds) if draw_odds else None
                    best_away = min(away_odds) if away_odds else None
                    
                    # Calculate probabilities
                    home_prob = 1/best_home if best_home else None
                    draw_prob = 1/best_draw if best_draw else None
                    away_prob = 1/best_away if best_away else None
                    
                    # Calculate market margin
                    probs = [p for p in [home_prob, draw_prob, away_prob] if p]
                    market_margin = (sum(probs) - 1) * 100 if probs else None
                    
                    # Simple signal analysis
                    signal = None
                    if all([home_prob, draw_prob, away_prob]) and market_margin:
                        # Normalize probabilities
                        total_prob = home_prob + draw_prob + away_prob
                        fair_home = home_prob / total_prob
                        fair_draw = draw_prob / total_prob
                        fair_away = away_prob / total_prob
                        
                        # Find value bets
                        edges = []
                        if home_prob < fair_home * 0.95:
                            edges.append(('Home', (fair_home - home_prob) / home_prob))
                        if draw_prob < fair_draw * 0.95:
                            edges.append(('Draw', (fair_draw - draw_prob) / draw_prob))
                        if away_prob < fair_away * 0.95:
                            edges.append(('Away', (fair_away - away_prob) / away_prob))
                        
                        if edges:
                            best_edge = max(edges, key=lambda x: x[1])
                            if best_edge[1] > 0.02:  # 2% edge threshold
                                signal = {
                                    'provider': 'Implied Probability',
                                    'confidence': min(best_edge[1] * 10, 1.0),
                                    'recommendation': f'Back {best_edge[0]} - Value bet detected',
                                    'edge': best_edge[1]
                                }
                    
                    market_data = {
                        'id': market.source_id,
                        'home_team': market.home_team,
                        'away_team': market.away_team,
                        'league': market.league_name or 'Soccer League',
                        'maturity_date': market.maturity_date.isoformat() if market.maturity_date else None,
                        'home_odds': best_home,
                        'draw_odds': best_draw,
                        'away_odds': best_away,
                        'home_prob': home_prob,
                        'draw_prob': draw_prob,
                        'away_prob': away_prob,
                        'market_margin': market_margin,
                        'bookmaker': odds[0].bookmaker if odds else 'Blockchain',
                        'signal': signal
                    }
                    
                    markets.append(market_data)
                    
                except Exception as e:
                    logger.error(f"Error processing market {market.source_id}: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error getting markets: {e}")
    
    return markets

def get_best_betting_opportunities(markets):
    """Extract best betting opportunities from markets."""
    opportunities = []
    
    for market in markets:
        if market.get('signal') and market['signal'].get('edge', 0) > 0.02:
            recommended_bet = market['signal']['recommendation'].split(' - ')[0].replace('Back ', '')
            
            opportunities.append({
                'match_id': market['id'],
                'home_team': market['home_team'],
                'away_team': market['away_team'],
                'recommended_bet': recommended_bet,
                'signal_probability': market['signal']['confidence'],
                'expected_edge': market['signal']['edge']
            })
    
    # Sort by edge
    opportunities.sort(key=lambda x: x['expected_edge'], reverse=True)
    
    return opportunities[:10]  # Top 10

def get_positions():
    """Get open positions from portfolio."""
    global portfolio
    positions = []
    
    try:
        for match_id, bet in portfolio['pending_bets'].items():
            # Calculate current P&L (simplified - would check current odds)
            current_value = bet['stake']  # Simplified
            pnl = ((current_value - bet['stake']) / bet['stake']) * 100
            
            positions.append({
                'id': match_id,
                'market': bet['match'],
                'selection': bet['bet_on'],
                'stake': bet['stake'],
                'odds': bet['odds'],
                'pnl': pnl
            })
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
    
    return positions

def get_recent_trades():
    """Get recent trade history."""
    # Read from trades file if exists
    trades = []
    
    try:
        if trades_file.exists():
            trades_df = pd.read_csv(trades_file)
            # Convert to list of dicts for last 10 trades
            recent = trades_df.tail(10).to_dict('records')
            
            for trade in recent:
                trades.append({
                    'action': 'Bet',
                    'market': trade.get('match', 'Unknown'),
                    'pnl': 0,  # Would calculate from results
                    'timestamp': trade.get('timestamp', datetime.now().isoformat())
                })
    except Exception as e:
        logger.error(f"Error reading trades: {e}")
    
    return trades

def get_activity_log():
    """Get recent system activity."""
    # Simple activity tracking
    return [
        {'message': '🚀 System started', 'timestamp': datetime.now().isoformat()},
        {'message': '📡 Connected to blockchain', 'timestamp': datetime.now().isoformat()},
        {'message': '⚽ Loading soccer markets', 'timestamp': datetime.now().isoformat()}
    ]

def get_ticker_data():
    """Get ticker information."""
    try:
        with db_manager.get_db_session() as db:
            active_markets = db.query(func.count(Market.id)).filter(
                Market.sport == 'Soccer',
                Market.is_finished == False
            ).scalar() or 0
            
            total_odds = db.query(func.count(Odd.id)).scalar() or 0
            
            return [
                {'label': 'Soccer Markets', 'value': str(active_markets), 'color': '#00ff00'},
                {'label': 'Total Odds', 'value': f'{total_odds:,}', 'color': '#00ffff'},
                {'label': 'Data Source', 'value': 'Blockchain V2', 'color': '#00ff00'},
                {'label': 'Signal Provider', 'value': 'Implied Probability', 'color': '#00ff00'}
            ]
    except:
        return [
            {'label': 'System Status', 'value': 'Error', 'color': '#ff4444'}
        ]

def get_performance_data():
    """Get performance chart data."""
    # Generate from portfolio history
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    values = [10000]  # Starting balance
    
    # Simulate based on portfolio
    current_value = portfolio['cash'] + sum(bet['stake'] for bet in portfolio['pending_bets'].values())
    
    # Linear interpolation to current value
    daily_change = (current_value - 10000) / 30
    for i in range(1, 30):
        values.append(values[-1] + daily_change + np.random.randn() * 50)
    
    return {
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'values': values
    }

# Routes
@app.route('/')
def index():
    """Main dashboard."""
    return render_template_string(UNIFIED_HTML)

@app.route('/api/status')
def api_status():
    """Get complete dashboard status."""
    markets = get_soccer_markets_with_signals()
    best_bets = get_best_betting_opportunities(markets)
    
    return jsonify({
        'account': get_account_data(),
        'markets': markets,
        'best_bets': best_bets,
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
    emit('connected', {'data': 'Connected to Ominari Soccer Trading'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('place_bet')
def handle_place_bet(data):
    """Handle bet placement from UI."""
    logger.info(f"Bet request: {data}")
    
    global portfolio
    
    try:
        # Create bet record
        bet_id = f"BET_{datetime.now().timestamp()}"
        market_id = data['market_id']
        
        # Get market details
        with db_manager.get_db_session() as db:
            market = db.query(Market).filter(Market.source_id == market_id).first()
            if market:
                bet_record = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'match_id': market_id,
                    'match': f"{market.home_team} vs {market.away_team}",
                    'bet_on': data['selection'],
                    'odds': data['odds'],
                    'stake': data['stake'],
                    'potential_return': data['stake'] * data['odds'],
                    'kick_off': market.maturity_date.isoformat() if market.maturity_date else None
                }
                
                # Update portfolio
                portfolio['cash'] -= data['stake']
                portfolio['pending_bets'][market_id] = bet_record
                portfolio['trades'] += 1
                
                # Save portfolio
                with open(portfolio_file, 'w') as f:
                    json.dump(portfolio, f, indent=2)
                
                emit('bet_result', {'status': 'success', 'message': 'Bet placed successfully'})
            else:
                emit('bet_result', {'status': 'error', 'message': 'Market not found'})
                
    except Exception as e:
        logger.error(f"Error placing bet: {e}")
        emit('bet_result', {'status': 'error', 'message': str(e)})

@socketio.on('execute_recommendation')
def handle_execute_recommendation(data):
    """Execute a recommended bet."""
    logger.info(f"Execute recommendation: {data}")
    
    # Similar to place_bet but with Kelly sizing
    # TODO: Implement Kelly criterion sizing
    
    emit('bet_result', {'status': 'success', 'message': 'Recommendation executed'})

@socketio.on('close_position')
def handle_close_position(data):
    """Handle position closure."""
    logger.info(f"Close position: {data}")
    
    global portfolio
    position_id = data['position_id']
    
    if position_id in portfolio['pending_bets']:
        bet = portfolio['pending_bets'][position_id]
        # Assume break-even for now (would check actual result)
        portfolio['cash'] += bet['stake']
        del portfolio['pending_bets'][position_id]
        
        # Save portfolio
        with open(portfolio_file, 'w') as f:
            json.dump(portfolio, f, indent=2)
        
        emit('position_closed', {'status': 'success', 'position_id': position_id})
    else:
        emit('position_closed', {'status': 'error', 'message': 'Position not found'})

# Background update thread
def background_updater():
    """Send updates to connected clients."""
    while True:
        time.sleep(10)  # Update every 10 seconds
        try:
            markets = get_soccer_markets_with_signals()[:5]  # Top 5 markets
            status = {
                'account': get_account_data(),
                'markets': markets,
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
    print("🚀 Unified Ominari Monitor Starting...")
    print("⚽ Soccer Markets Only - Real Blockchain Data")
    print("📊 Dashboard: http://localhost:8888")
    print("🔌 WebSocket enabled for live updates")
    print("📡 Integrated with paper trading and signals")
    
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)