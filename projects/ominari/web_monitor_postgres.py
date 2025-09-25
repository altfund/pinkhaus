#!/usr/bin/env python3
"""
Ominari Web Monitor - Original UI with Direct PostgreSQL Integration
Single, clean dashboard on port 8888 showing markets, signals, and stats
"""

import os
# Set PostgreSQL environment variables BEFORE imports
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

import json
import logging
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import threading
import time
import numpy as np
import pandas as pd

# Import paper trading
from paper_trading_engine import PaperTradingEngine
from paper_trading_sessions import PaperTradingSessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ominari-trading-system-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize paper trading
session_manager = PaperTradingSessionManager()
paper_engine = PaperTradingEngine()

# Database connection
@contextmanager
def get_db():
    conn = psycopg2.connect(
        host=os.environ['PG_HOST'],
        port=os.environ['PG_PORT'],
        user=os.environ['PG_USER'],
        password=os.environ['PG_PASSWORD'],
        database=os.environ['PG_DB']
    )
    conn.set_session(autocommit=True)
    try:
        yield conn
    finally:
        conn.close()

# Strategy configuration
STRATEGY_CONFIG = {
    'kelly_fraction': 0.25,
    'min_bet': 10,
    'min_bet_pct': 0.001,
    'bankroll': 10000,
    'cap_per_game': 0.25,
    'cap_per_bet': 0.25,
    'cap_per_game_market': 0.10,
    'min_break_minutes': 240,
    'avg_game_duration_minutes': 180,
    'chunk_selection': 'first',
    'chunk_limit_hours': 12,
    'biases': {
        'favorite': -0.01,
        'longshot': 0.01,
        'draw': 0.005
    }
}

def get_market_data():
    """Get market data directly from PostgreSQL"""
    markets = []
    signals = []
    
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get upcoming markets with odds
                query = """
                SELECT 
                    m.source_id as market_id,
                    m.source_id,
                    m.home_team,
                    m.away_team,
                    m.sport,
                    m.league_name,
                    m.maturity_date,
                    m.is_finished,
                    m.source,
                    o1.decimal_odds as home_odds,
                    o2.decimal_odds as draw_odds,
                    o3.decimal_odds as away_odds
                FROM market m
                LEFT JOIN odd o1 ON o1.source_id = m.source_id AND o1.outcome = 'home' AND o1.position = 0
                LEFT JOIN odd o2 ON o2.source_id = m.source_id AND o2.outcome = 'draw' AND o2.position = 1
                LEFT JOIN odd o3 ON o3.source_id = m.source_id AND o3.outcome = 'away' AND o3.position = 2
                WHERE 
                    m.source = 'api_live_real'
                    AND m.is_finished = FALSE
                    AND m.maturity_date > NOW()
                    AND m.sport = 'Soccer'
                    AND (o1.decimal_odds IS NOT NULL OR o2.decimal_odds IS NOT NULL OR o3.decimal_odds IS NOT NULL)
                ORDER BY m.maturity_date ASC
                LIMIT 100
                """
                
                cur.execute(query)
                rows = cur.fetchall()
                
                for row in rows:
                    market = {
                        'id': row['source_id'],
                        'source_id': row['source_id'],
                        'home_team': row['home_team'],
                        'away_team': row['away_team'],
                        'sport': row['sport'],
                        'league_name': row['league_name'],
                        'maturity_date': row['maturity_date'].isoformat() if row['maturity_date'] else None,
                        'is_finished': row['is_finished'],
                        'source': row['source']
                    }
                    markets.append(market)
                    
                    # Calculate simple signals based on odds
                    home_odds = float(row['home_odds']) if row['home_odds'] else 0
                    draw_odds = float(row['draw_odds']) if row['draw_odds'] else 0
                    away_odds = float(row['away_odds']) if row['away_odds'] else 0
                    
                    # Simple edge calculation (placeholder - you'd use real model)
                    # Positive edge for favorites
                    home_edge = (1 / home_odds * 100 - 33.33) if home_odds > 0 else 0
                    draw_edge = (1 / draw_odds * 100 - 33.33) if draw_odds > 0 else 0
                    away_edge = (1 / away_odds * 100 - 33.33) if away_odds > 0 else 0
                    
                    # Simple Kelly stake calculation
                    def calculate_stake(edge, odds):
                        if edge <= 0 or odds <= 1:
                            return 0
                        kelly_stake = (edge / 100) / (odds - 1) * STRATEGY_CONFIG['kelly_fraction'] * STRATEGY_CONFIG['bankroll']
                        if kelly_stake < STRATEGY_CONFIG['min_bet']:
                            return 0
                        return min(kelly_stake, STRATEGY_CONFIG['bankroll'] * STRATEGY_CONFIG['cap_per_bet'])
                    
                    signal = {
                        'home_odds': home_odds,
                        'home_edge': home_edge,
                        'home_stake': calculate_stake(home_edge, home_odds),
                        'draw_odds': draw_odds,
                        'draw_edge': draw_edge,
                        'draw_stake': calculate_stake(draw_edge, draw_odds),
                        'away_odds': away_odds,
                        'away_edge': away_edge,
                        'away_stake': calculate_stake(away_edge, away_odds)
                    }
                    signals.append(signal)
                    
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
    
    return markets, signals, {}, []

SINGLE_PAGE_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Ominari Trading Dashboard</title>
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
        }
        
        .header-title {
            font-size: 1.5em;
            font-weight: bold;
            color: #00ffff;
        }
        
        .header-time {
            color: #888;
            font-size: 0.9em;
        }
        
        /* Main Grid Layout */
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 15px;
            padding: 15px;
            max-width: 1600px;
            margin: 0 auto;
        }
        
        /* Metric Cards */
        .metric-row {
            grid-column: span 12;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .metric-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s;
        }
        
        .metric-card:hover {
            border-color: #00ff00;
            transform: translateY(-2px);
        }
        
        .metric-value {
            font-size: 2.2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .metric-label {
            color: #888;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        
        .metric-sub {
            font-size: 0.8em;
            color: #666;
        }
        
        /* Portfolio Card Special */
        .portfolio-card {
            grid-column: span 3;
        }
        
        .performance-card {
            grid-column: span 3;
        }
        
        .system-card {
            grid-column: span 3;
        }
        
        .exposure-card {
            grid-column: span 3;
        }
        
        /* Markets Section */
        .markets-section {
            grid-column: span 12;
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 15px;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }
        
        .section-title {
            font-size: 1.3em;
            color: #00ffff;
        }
        
        /* Markets Table */
        .markets-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .markets-table th {
            background: #1a1a1a;
            color: #00ffff;
            padding: 10px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        .markets-table td {
            padding: 10px;
            border-bottom: 1px solid #222;
        }
        
        .markets-table tr:hover {
            background: #1a1a1a;
        }
        
        /* Positions Section */
        .positions-section {
            grid-column: span 8;
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 20px;
        }
        
        /* Activity Feed */
        .activity-section {
            grid-column: span 4;
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 20px;
            max-height: 600px;
            overflow-y: auto;
        }
        
        .activity-item {
            padding: 8px;
            margin-bottom: 5px;
            background: #1a1a1a;
            border-left: 3px solid #333;
            border-radius: 4px;
            font-size: 0.85em;
        }
        
        .activity-item.trade {
            border-left-color: #00ff00;
        }
        
        .activity-item.evaluation {
            border-left-color: #00ffff;
        }
        
        .activity-item.error {
            border-left-color: #ff0000;
        }
        
        .activity-time {
            color: #666;
            font-size: 0.8em;
        }
        
        /* Strategy Section */
        .strategy-section {
            grid-column: span 12;
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 20px;
        }
        
        .strategy-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }
        
        /* Tabs for positions */
        .tab-nav {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            border-bottom: 1px solid #333;
        }
        
        .tab-btn {
            background: transparent;
            border: none;
            color: #888;
            padding: 10px 20px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .tab-btn.active {
            color: #00ff00;
            border-bottom: 2px solid #00ff00;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        /* Utility Classes */
        .positive { color: #00ff00; }
        .negative { color: #ff0000; }
        .neutral { color: #ffaa00; }
        
        /* Spreadsheet table styles */
        .spreadsheet-table {
            font-family: 'Courier New', monospace;
        }
        
        .spreadsheet-table thead th {
            background: #1a1a1a;
            font-weight: bold;
            font-size: 0.85em;
            white-space: nowrap;
            border-right: 1px solid #333;
        }
        
        .spreadsheet-table tbody tr {
            transition: background-color 0.1s;
        }
        
        .spreadsheet-table tbody tr:hover {
            background: rgba(255, 255, 255, 0.08);
        }
        
        .spreadsheet-table tbody td {
            padding: 4px 6px;
            border-right: 1px solid #222;
            white-space: nowrap;
        }
        
        .edge-positive { color: #90ff90; }
        .edge-negative { color: #ff8888; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .loading {
            animation: pulse 1.5s infinite;
        }
    </style>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
    <div class="header">
        <div class="header-title">⚽ Ominari Trading System</div>
        <div style="display: flex; gap: 15px; align-items: center;">
            <button onclick="executeTrades()" style="background: #00ff00; color: #000; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; cursor: pointer; font-size: 0.9em;">
                ⚡ Execute Trades
            </button>
            <div class="header-time" id="current-time">--:--:--</div>
        </div>
    </div>
    
    <div class="dashboard-grid">
        <!-- Top Metric Cards -->
        <div class="metric-row">
            <!-- Portfolio Card -->
            <div class="metric-card portfolio-card">
                <div class="metric-label">Portfolio Value</div>
                <div class="metric-value" id="portfolio-value">$0</div>
                <div class="metric-sub">
                    <span id="portfolio-change" class="positive">+$0 (0%)</span>
                </div>
                <div style="margin-top: 10px; font-size: 0.9em;">
                    <div>Cash: <span id="cash-available">$0</span></div>
                    <div>Positions: <span id="positions-value">$0</span></div>
                    <div>Exposure: <span id="exposure-pct" style="color: #ffff00;">0%</span></div>
                </div>
            </div>
            
            <!-- Performance Card -->
            <div class="metric-card performance-card">
                <div class="metric-label">Live Performance</div>
                <div class="metric-value" id="live-win-rate">0%</div>
                <div style="color: #888; margin-bottom: 10px;">Win Rate</div>
                <div style="font-size: 0.9em;">
                    <div>ROI: <span id="roi" class="positive">0%</span></div>
                    <div>Trades: <span id="trades-count">0</span></div>
                    <div>P. Factor: <span id="profit-factor">0.0</span></div>
                </div>
            </div>
            
            <!-- System Card -->
            <div class="metric-card system-card">
                <div class="metric-label">System Status</div>
                <div class="metric-value" id="system-status" style="font-size: 1.5em;">Active</div>
                <div class="metric-sub" id="session-id">Session: --</div>
                <div style="margin-top: 10px; font-size: 0.9em;">
                    <div>Markets: <span id="markets-count">0</span></div>
                    <div>Positions: <span id="positions-count">0</span></div>
                </div>
            </div>
        </div>
        
        <!-- Markets Dashboard -->
        <div class="markets-section">
            <div class="section-header">
                <h3 class="section-title">⚽ Live Markets</h3>
                <div style="display: flex; gap: 15px; align-items: center;">
                    <div style="font-size: 0.9em;">
                        <span style="color: #888;">Total:</span> <span id="total-matches" style="color: #00ff00;">0</span>
                        <span style="color: #888; margin-left: 15px;">Active:</span> <span id="active-positions" style="color: #00ffff;">0</span>
                        <span style="color: #666; margin-left: 20px;">Updated:</span> <span id="markets-update-time" style="color: #888;">--:--:--</span>
                    </div>
                </div>
            </div>
            
            <div style="overflow: auto; max-height: 350px;">
                <table class="spreadsheet-table markets-table" style="width: 100%; min-width: 1200px;">
                    <thead>
                        <tr style="border-bottom: 2px solid #444;">
                            <th style="padding: 6px; text-align: left;">Match</th>
                            <th style="padding: 6px; text-align: center;">Time</th>
                            <th style="padding: 6px; text-align: center;">Status</th>
                            <th style="padding: 6px; text-align: center;">H Odds</th>
                            <th style="padding: 6px; text-align: center;">H Edge</th>
                            <th style="padding: 6px; text-align: center;">H Stake</th>
                            <th style="padding: 6px; text-align: center;">D Odds</th>
                            <th style="padding: 6px; text-align: center;">D Edge</th>
                            <th style="padding: 6px; text-align: center;">D Stake</th>
                            <th style="padding: 6px; text-align: center;">A Odds</th>
                            <th style="padding: 6px; text-align: center;">A Edge</th>
                            <th style="padding: 6px; text-align: center;">A Stake</th>
                            <th style="padding: 6px; text-align: center;">Total</th>
                        </tr>
                    </thead>
                    <tbody id="matches-tbody">
                        <!-- Markets will be populated here -->
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Positions Section -->
        <div class="positions-section">
            <div class="section-header">
                <h3 class="section-title">📊 Positions</h3>
            </div>
            
            <div class="tab-nav">
                <button class="tab-btn active" onclick="showTab('open')">Open</button>
                <button class="tab-btn" onclick="showTab('closed')">Closed</button>
                <button class="tab-btn" onclick="showTab('summary')">Summary</button>
            </div>
            
            <div id="open-positions" class="tab-content active">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                    <thead>
                        <tr style="border-bottom: 1px solid #333;">
                            <th style="padding: 8px; text-align: left;">Market</th>
                            <th style="padding: 8px; text-align: center;">Side</th>
                            <th style="padding: 8px; text-align: right;">Stake</th>
                            <th style="padding: 8px; text-align: center;">Odds</th>
                            <th style="padding: 8px; text-align: right;">Current</th>
                            <th style="padding: 8px; text-align: right;">P&L</th>
                        </tr>
                    </thead>
                    <tbody id="open-positions-tbody">
                        <!-- Open positions will be populated here -->
                    </tbody>
                </table>
            </div>
            
            <div id="closed-positions" class="tab-content">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                    <thead>
                        <tr style="border-bottom: 1px solid #333;">
                            <th style="padding: 8px; text-align: left;">Market</th>
                            <th style="padding: 8px; text-align: center;">Side</th>
                            <th style="padding: 8px; text-align: right;">Stake</th>
                            <th style="padding: 8px; text-align: center;">Odds</th>
                            <th style="padding: 8px; text-align: center;">Result</th>
                            <th style="padding: 8px; text-align: right;">P&L</th>
                        </tr>
                    </thead>
                    <tbody id="closed-positions-tbody">
                        <!-- Closed positions will be populated here -->
                    </tbody>
                </table>
            </div>
            
            <div id="summary-positions" class="tab-content">
                <div style="padding: 20px;">
                    <h4 style="color: #00ff00; margin-bottom: 15px;">Performance Summary</h4>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                        <div>
                            <div style="color: #888;">Total P&L</div>
                            <div style="font-size: 1.5em; color: #00ff00;" id="total-pnl">$0</div>
                        </div>
                        <div>
                            <div style="color: #888;">Win Rate</div>
                            <div style="font-size: 1.5em; color: #00ff00;" id="summary-win-rate">0%</div>
                        </div>
                        <div>
                            <div style="color: #888;">Avg Trade</div>
                            <div style="font-size: 1.5em; color: #00ff00;" id="avg-trade">$0</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Activity Feed -->
        <div class="activity-section">
            <div class="section-header">
                <h3 class="section-title">📋 Activity</h3>
            </div>
            <div id="activity-feed">
                <!-- Activity items will be populated here -->
            </div>
        </div>
        
        <!-- Strategy Parameters -->
        <div class="strategy-section">
            <div class="section-header">
                <h3 class="section-title">⚙️ Strategy Parameters</h3>
            </div>
            <div class="strategy-grid">
                <div>
                    <h4 style="color: #00ff00; margin-bottom: 10px;">Risk Management</h4>
                    <div style="background: #1a1a1a; padding: 15px; border-radius: 5px;">
                        <div style="margin-bottom: 8px;">Kelly Fraction: <span id="kelly-fraction" style="color: #00ff00;">25%</span></div>
                        <div style="margin-bottom: 8px;">Min Bet: <span id="min-bet" style="color: #00ff00;">$10</span></div>
                        <div style="margin-bottom: 8px;">Max per Game: <span id="cap-per-game" style="color: #00ff00;">25%</span></div>
                        <div>Max per Bet: <span id="cap-per-bet" style="color: #00ff00;">25%</span></div>
                    </div>
                </div>
                <div>
                    <h4 style="color: #00ff00; margin-bottom: 10px;">Market Biases</h4>
                    <div style="background: #1a1a1a; padding: 15px; border-radius: 5px;">
                        <div style="margin-bottom: 8px;">Favorites: <span id="bias-favorite" class="negative">-1.0%</span></div>
                        <div style="margin-bottom: 8px;">Draws: <span id="bias-draw" class="positive">+0.5%</span></div>
                        <div>Longshots: <span id="bias-longshot" class="positive">+1.0%</span></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        // Update time
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = now.toLocaleTimeString();
        }
        setInterval(updateTime, 1000);
        updateTime();
        
        // Socket event handlers
        socket.on('connect', () => {
            console.log('Connected to server');
            socket.emit('request_dashboard_data');
        });
        
        socket.on('dashboard_update', (data) => {
            console.log('Received dashboard update:', data);
            updateDashboard(data);
        });
        
        socket.on('activity', (item) => {
            addActivityItem(item);
        });
        
        // Update dashboard
        function updateDashboard(data) {
            // Update metrics
            updateMetrics(data);
            
            // Update markets
            if (data.markets) {
                updateMarkets(data.markets, data.signals);
            }
            
            // Update positions
            if (data.positions) {
                updatePositions(data.positions);
            }
            
            // Update timestamp
            document.getElementById('markets-update-time').textContent = 
                new Date().toLocaleTimeString();
        }
        
        function updateMetrics(data) {
            const portfolio = data.portfolio || {};
            const performance = data.performance || {};
            const session = data.session || {};
            
            // Portfolio
            document.getElementById('portfolio-value').textContent = 
                '$' + (portfolio.portfolio_value || 0).toFixed(2);
            document.getElementById('cash-available').textContent = 
                '$' + (portfolio.current_bankroll || 0).toFixed(2);
            document.getElementById('positions-value').textContent = 
                '$' + (portfolio.positions_value || 0).toFixed(2);
            
            const exposure = portfolio.exposure_pct || 0;
            document.getElementById('exposure-pct').textContent = 
                exposure.toFixed(1) + '%';
            
            // Performance
            document.getElementById('live-win-rate').textContent = 
                (performance.win_rate || 0).toFixed(1) + '%';
            document.getElementById('roi').textContent = 
                (performance.roi || 0).toFixed(1) + '%';
            document.getElementById('roi').className = 
                performance.roi >= 0 ? 'positive' : 'negative';
            document.getElementById('trades-count').textContent = 
                performance.total_trades || 0;
            document.getElementById('profit-factor').textContent = 
                (performance.profit_factor || 0).toFixed(2);
            
            // System
            document.getElementById('system-status').textContent = 
                session.status === 'active' ? 'Active' : 'Inactive';
            document.getElementById('session-id').textContent = 
                'Session: ' + (session.session_id || '--');
            document.getElementById('markets-count').textContent = 
                data.markets?.length || 0;
            document.getElementById('positions-count').textContent = 
                Object.keys(portfolio.positions || {}).length;
            
            // Strategy
            const strategy = data.strategy || {};
            document.getElementById('kelly-fraction').textContent = 
                ((strategy.kelly_fraction || 0.25) * 100) + '%';
            document.getElementById('min-bet').textContent = 
                '$' + (strategy.min_bet || 10);
            document.getElementById('cap-per-game').textContent = 
                ((strategy.cap_per_game || 0.25) * 100) + '%';
            document.getElementById('cap-per-bet').textContent = 
                ((strategy.cap_per_bet || 0.25) * 100) + '%';
            document.getElementById('bias-favorite').textContent = 
                ((strategy.biases?.favorite || -0.01) * 100).toFixed(1) + '%';
            document.getElementById('bias-draw').textContent = 
                '+' + ((strategy.biases?.draw || 0.005) * 100).toFixed(1) + '%';
            document.getElementById('bias-longshot').textContent = 
                '+' + ((strategy.biases?.longshot || 0.01) * 100).toFixed(1) + '%';
        }
        
        function updateMarkets(markets, signals) {
            const tbody = document.getElementById('matches-tbody');
            tbody.innerHTML = '';
            
            let totalExposure = 0;
            let activeCount = 0;
            
            markets.forEach((market, i) => {
                const signal = signals ? signals[i] : {};
                const row = document.createElement('tr');
                
                // Match info
                let matchCell = '<td style="padding: 6px;">' + market.home_team + ' vs ' + market.away_team + '</td>';
                
                // Time
                const kickoff = new Date(market.maturity_date);
                const now = new Date();
                const hoursToKickoff = (kickoff - now) / (1000 * 60 * 60);
                let timeStr = hoursToKickoff > 0 ? 
                    hoursToKickoff.toFixed(1) + 'h' : 
                    'Live';
                let timeCell = '<td style="padding: 6px; text-align: center;">' + timeStr + '</td>';
                
                // Status
                let statusStr = market.is_finished ? 'Finished' : 
                              (hoursToKickoff < 0 ? 'Live' : 'Upcoming');
                let statusCell = '<td style="padding: 6px; text-align: center;">' + statusStr + '</td>';
                
                // Odds and edges
                let cells = '';
                ['home', 'draw', 'away'].forEach(outcome => {
                    const odds = signal[outcome + '_odds'] || 0;
                    const edge = signal[outcome + '_edge'] || 0;
                    const stake = signal[outcome + '_stake'] || 0;
                    
                    const edgeClass = edge > 2 ? 'edge-positive' : 
                                    (edge < -2 ? 'edge-negative' : '');
                    
                    cells += '<td style="padding: 6px; text-align: center;">' + 
                            (odds > 0 ? odds.toFixed(2) : '-') + '</td>';
                    cells += '<td style="padding: 6px; text-align: center;" class="' + edgeClass + '">' + 
                            (edge !== 0 ? edge.toFixed(1) + '%' : '-') + '</td>';
                    cells += '<td style="padding: 6px; text-align: center;">' + 
                            (stake > 0 ? '$' + stake.toFixed(0) : '-') + '</td>';
                    
                    if (stake > 0) {
                        totalExposure += stake;
                        activeCount++;
                    }
                });
                
                // Total
                const totalStake = (signal.home_stake || 0) + (signal.draw_stake || 0) + (signal.away_stake || 0);
                let totalCell = '<td style="padding: 6px; text-align: center; font-weight: bold;">' + 
                               (totalStake > 0 ? '$' + totalStake.toFixed(0) : '-') + '</td>';
                
                row.innerHTML = matchCell + timeCell + statusCell + cells + totalCell;
                tbody.appendChild(row);
            });
            
            // Update header stats
            document.getElementById('total-matches').textContent = markets.length;
            document.getElementById('active-positions').textContent = activeCount;
        }
        
        function updatePositions(positions) {
            const openTbody = document.getElementById('open-positions-tbody');
            const closedTbody = document.getElementById('closed-positions-tbody');
            
            openTbody.innerHTML = '';
            closedTbody.innerHTML = '';
            
            let totalPnl = 0;
            let wins = 0;
            let losses = 0;
            
            // Open positions
            Object.values(positions).forEach(pos => {
                if (pos.status === 'open') {
                    const row = document.createElement('tr');
                    row.innerHTML = \`
                        <td style="padding: 8px;">\${pos.market_name}</td>
                        <td style="padding: 8px; text-align: center;">\${pos.outcome.toUpperCase()}</td>
                        <td style="padding: 8px; text-align: right;">$\${pos.total_stake.toFixed(2)}</td>
                        <td style="padding: 8px; text-align: center;">\${pos.avg_odds.toFixed(2)}</td>
                        <td style="padding: 8px; text-align: right;">$\${pos.current_value.toFixed(2)}</td>
                        <td style="padding: 8px; text-align: right;" class="\${pos.pnl >= 0 ? 'positive' : 'negative'}">
                            \${pos.pnl >= 0 ? '+' : ''}$\${Math.abs(pos.pnl).toFixed(2)}
                        </td>
                    \`;
                    openTbody.appendChild(row);
                }
            });
            
            // Closed positions
            positions.closed?.forEach(pos => {
                const row = document.createElement('tr');
                const resultStr = pos.result === 'won' ? 'Won' : 'Lost';
                const resultClass = pos.result === 'won' ? 'positive' : 'negative';
                
                row.innerHTML = \`
                    <td style="padding: 8px;">\${pos.market_name}</td>
                    <td style="padding: 8px; text-align: center;">\${pos.outcome.toUpperCase()}</td>
                    <td style="padding: 8px; text-align: right;">$\${pos.total_stake.toFixed(2)}</td>
                    <td style="padding: 8px; text-align: center;">\${pos.avg_odds.toFixed(2)}</td>
                    <td style="padding: 8px; text-align: center;" class="\${resultClass}">\${resultStr}</td>
                    <td style="padding: 8px; text-align: right;" class="\${pos.pnl >= 0 ? 'positive' : 'negative'}">
                        \${pos.pnl >= 0 ? '+' : ''}$\${Math.abs(pos.pnl).toFixed(2)}
                    </td>
                \`;
                closedTbody.appendChild(row);
                
                totalPnl += pos.pnl;
                if (pos.result === 'won') wins++;
                else losses++;
            });
            
            // Update summary
            const winRate = (wins + losses) > 0 ? (wins / (wins + losses) * 100) : 0;
            document.getElementById('total-pnl').textContent = 
                (totalPnl >= 0 ? '+' : '') + '$' + Math.abs(totalPnl).toFixed(2);
            document.getElementById('summary-win-rate').textContent = 
                winRate.toFixed(1) + '%';
        }
        
        function addActivityItem(item) {
            const feed = document.getElementById('activity-feed');
            const div = document.createElement('div');
            div.className = 'activity-item ' + item.type;
            
            const time = new Date(item.timestamp || Date.now());
            div.innerHTML = \`
                <div class="activity-time">\${time.toLocaleTimeString()}</div>
                <div>\${item.message}</div>
            \`;
            
            feed.insertBefore(div, feed.firstChild);
            
            // Keep only last 50 items
            while (feed.children.length > 50) {
                feed.removeChild(feed.lastChild);
            }
        }
        
        // Tab switching
        function showTab(tabName) {
            // Update buttons
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            // Update content
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(tabName + '-positions').classList.add('active');
        }
        
        // Execute trades
        function executeTrades() {
            socket.emit('execute_trades');
        }
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            socket.emit('request_dashboard_data');
        }, 30000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(SINGLE_PAGE_DASHBOARD)

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('request_dashboard_data')
def handle_request_dashboard_data():
    """Send dashboard data to client"""
    emit('dashboard_update', get_dashboard_data())

@socketio.on('execute_trades')
def handle_execute_trades():
    """Execute paper trades"""
    result = execute_paper_trades()
    emit('activity', {
        'type': 'trade',
        'message': f"Executed {result.get('trades_made', 0)} trades",
        'timestamp': datetime.now(timezone.utc).isoformat()
    })
    emit('dashboard_update', get_dashboard_data())

def get_dashboard_data():
    """Get all dashboard data"""
    try:
        # Get current session
        session = session_manager.get_current_session()
        if not session:
            session = session_manager.sessions['sessions'][
                session_manager.create_session(initial_bankroll=STRATEGY_CONFIG['bankroll'])
            ]
        
        # Get markets and signals
        markets, signals, stats, chunks = get_market_data()
        
        # Get performance
        performance = session_manager.get_session_performance(session['session_id'])
        
        # Format response
        return {
            'session': {
                'session_id': session['session_id'],
                'status': session['status']
            },
            'portfolio': {
                'portfolio_value': session['portfolio_value'],
                'current_bankroll': session['current_bankroll'],
                'positions': session['positions'],
                'positions_value': sum(p['current_value'] for p in session['positions'].values()),
                'exposure_pct': (sum(p['total_stake'] for p in session['positions'].values()) / STRATEGY_CONFIG['bankroll'] * 100) if session['positions'] else 0
            },
            'performance': performance,
            'markets': markets,
            'signals': signals,
            'positions': {
                **session['positions'],
                'closed': session['closed_positions'][-20:]  # Last 20 closed
            },
            'strategy': STRATEGY_CONFIG
        }
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        return {}

def execute_paper_trades():
    """Execute paper trades using evaluate_open_markets logic"""
    try:
        session = session_manager.get_current_session()
        if not session:
            return {'success': False, 'error': 'No active session'}
        
        # Get market data
        markets, signals, stats, chunks = get_market_data()
        
        # Find trades to execute
        trades_to_execute = []
        for i, market in enumerate(markets):
            signal = signals[i]
            for outcome in ['home', 'draw', 'away']:
                stake = signal.get(f'{outcome}_stake', 0)
                if stake > 0:
                    trades_to_execute.append({
                        'market_id': market['id'],
                        'market_name': f"{market['home_team']} vs {market['away_team']}",
                        'outcome': outcome,
                        'stake': stake,
                        'odds': signal[f'{outcome}_odds'],
                        'edge': signal[f'{outcome}_edge'],
                        'maturity_date': market['maturity_date']
                    })
        
        # Record trades
        if trades_to_execute:
            session_manager.record_trades(session['session_id'], trades_to_execute)
            
            # Log each trade
            for trade in trades_to_execute:
                socketio.emit('activity', {
                    'type': 'trade',
                    'message': f"Placed {trade['outcome']} ${trade['stake']:.2f} on {trade['market_name']} @ {trade['odds']:.2f}",
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
        
        return {
            'success': True,
            'trades_made': len(trades_to_execute),
            'total_stake': sum(t['stake'] for t in trades_to_execute)
        }
        
    except Exception as e:
        logger.error(f"Error executing trades: {e}")
        return {'success': False, 'error': str(e)}

def paper_trading_background_loop():
    """Background loop for automated paper trading"""
    logger.info("🤖 Starting paper trading background loop...")
    
    while True:
        try:
            logger.info("⚡ Paper trading cycle starting...")
            result = execute_paper_trades()
            
            if result.get("success"):
                trades_made = result.get("trades_made", 0)
                if trades_made > 0:
                    logger.info(f"✅ Paper trading cycle complete: {trades_made} trades executed")
                else:
                    logger.info("📊 Paper trading cycle complete: No suitable trades found")
            else:
                logger.warning(f"⚠️ Paper trading cycle failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Paper trading background error: {e}")
        
        # Wait 5 minutes between cycles
        time.sleep(300)

def background_updates():
    """Background thread for periodic dashboard updates"""
    while True:
        try:
            with app.app_context():
                socketio.emit('dashboard_update', get_dashboard_data())
        except Exception as e:
            logger.error(f"Background update error: {e}")
        time.sleep(30)

if __name__ == '__main__':
    # Start background threads
    bg_thread = threading.Thread(target=background_updates, daemon=True)
    bg_thread.start()
    
    trading_thread = threading.Thread(target=paper_trading_background_loop, daemon=True)
    trading_thread.start()
    
    # Start server
    logger.info("Starting Ominari Dashboard on http://localhost:8888")
    logger.info("Using direct PostgreSQL connection - no SQLite dependencies")
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)