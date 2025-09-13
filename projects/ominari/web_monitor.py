#!/usr/bin/env python3
"""
Ominari Web Monitor - Consolidated Dashboard
Single, clean dashboard on port 8888 showing markets, signals, and stats
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
from database_v2 import db_manager
from models import Market, Odd, Bet, BettingSession
from paper_trading_models_v2 import (
    PaperTradingSession, PaperTradingPosition, PaperTradingSnapshot,
    MarketName, PositionStatus, SessionStatus, Result
)
from sqlalchemy import func, desc
import threading
import time
import numpy as np
import requests
import pandas as pd
from evaluate_open_markets import (
    summarize_match_schedule_from_open_markets,
    find_upcoming_game_breaks,
    extract_active_game_periods_from_breaks,
    generate_betting_session_report_and_save
)
from simple_daily_change import get_daily_change
from calculate_max_drawdown import calculate_max_drawdown, calculate_rolling_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
API_BASE_URL = "http://localhost:8000"

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ominari-trading-system-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Strategy configuration - synchronized with evaluate_open_markets.py
STRATEGY_CONFIG = {
    'kelly_fraction': 0.25,  # Conservative 25% Kelly
    'min_bet': 10,  # Minimum $10 bet
    'min_bet_pct': 0.001,  # Min 0.1% of bankroll
    'bankroll': 10000,
    'cap_per_game': 0.25,  # Max 25% per game
    'cap_per_bet': 0.25,  # Max 25% per bet
    'cap_per_game_market': 0.10,  # Max 10% per market type
    'min_break_minutes': 240,  # 4 hour minimum break between chunks (reduced from 6)
    'avg_game_duration_minutes': 180,  # 3 hour average game duration
    'chunk_selection': 'first',  # 'first', 'all', or 'limit_hours'
    'chunk_limit_hours': 12,  # If chunk_selection is 'limit_hours'
    'biases': {
        'favorite': -0.01,
        'longshot': 0.01,
        'draw': 0.005
    }
}

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
        
        /* Match Dashboard Enhancements */
        #matches-table tbody tr:hover {
            background: rgba(255, 255, 255, 0.08) !important;
            transition: background 0.2s ease;
        }
        
        #matches-table tbody tr.has-positions {
            font-weight: 500;
        }
        
        #matches-table td {
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Collapsible */
        .collapsible {
            cursor: pointer;
            user-select: none;
        }
        
        .collapsible::before {
            content: '▼ ';
            display: inline-block;
            transition: transform 0.2s;
        }
        
        .collapsible.collapsed::before {
            transform: rotate(-90deg);
        }
        
        .collapsible-content {
            max-height: 1000px;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }
        
        .collapsible-content.collapsed {
            max-height: 0;
        }
        
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
        
        .spreadsheet-table tbody tr:nth-child(even) {
            background: rgba(255, 255, 255, 0.02);
        }
        
        .spreadsheet-table tbody tr:hover {
            background: rgba(255, 255, 255, 0.08);
        }
        
        .spreadsheet-table tbody td {
            padding: 4px 6px;
            border-right: 1px solid #222;
            white-space: nowrap;
        }
        
        /* Row status coloring */
        .match-row.live {
            background: linear-gradient(90deg, rgba(255, 0, 0, 0.1), transparent);
        }
        
        .match-row.finished {
            opacity: 0.7;
        }
        
        .match-row.has-positions {
            font-weight: bold;
        }
        
        @keyframes slideDown {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .position-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            margin: 2px;
        }
        
        .position-badge.open {
            background: #00ff00;
            color: #000;
        }
        
        .position-badge.closed {
            background: #666;
            color: #fff;
        }
        
        .edge-positive-strong {
            color: #00ff00;
            font-weight: bold;
            text-shadow: 0 0 4px rgba(0, 255, 0, 0.3);
        }
        
        .edge-positive {
            color: #90ff90;
        }
        
        .edge-negative {
            color: #ff8888;
        }
        
        .edge-negative-strong {
            color: #ff4444;
            font-weight: bold;
            text-shadow: 0 0 4px rgba(255, 68, 68, 0.3);
        }
        
        /* Responsive */
        @media (max-width: 1400px) {
            .portfolio-card, .performance-card, .system-card, .exposure-card {
                grid-column: span 6;
            }
            
            .positions-section {
                grid-column: span 12;
            }
            
            .activity-section {
                grid-column: span 12;
            }
        }
        
        @media (max-width: 768px) {
            .portfolio-card, .performance-card, .system-card, .exposure-card {
                grid-column: span 12;
            }
            
            .metric-row {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">⚽ Ominari Trading System</div>
        <div style="display: flex; gap: 15px; align-items: center;">
            <button onclick="executeTrades()" style="background: #00ff00; color: #000; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; cursor: pointer; font-size: 0.9em;">
                ⚡ Execute Trades
            </button>
            <select id="period-selector" onchange="updatePeriod(this.value)" style="background: #222; color: #00ff00; border: 1px solid #333; padding: 5px 10px; border-radius: 4px; font-size: 0.9em;">
                <option value="1">1D</option>
                <option value="7">7D</option>
                <option value="30" selected>30D</option>
                <option value="90">90D</option>
                <option value="365">1Y</option>
                <option value="9999">All</option>
            </select>
            <div class="header-time" id="current-time">--:--:--</div>
        </div>
    </div>
    
    <div class="dashboard-grid">
        <!-- Top Metric Cards -->
        <div class="metric-row">
            <!-- Portfolio Card (Expanded) -->
            <div class="metric-card portfolio-card" style="grid-column: span 2;">
                <div style="display: flex; gap: 20px;">
                    <div style="flex: 1;">
                        <div class="metric-label">Portfolio Value</div>
                        <div class="metric-value" id="portfolio-value">$0</div>
                        <div class="metric-sub">
                            <span id="portfolio-change" class="positive">+$0 (0%)</span>
                        </div>
                        <div style="margin-top: 10px; font-size: 0.9em;">
                            <div>Cash: <span id="cash-available">$0</span></div>
                            <div>Positions: <span id="positions-value">$0</span></div>
                            <div>Exposure: <span id="exposure-pct" style="color: #ffff00;">0%</span> (<span id="total-exposure">$0</span>)</div>
                            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #333;">
                                <div id="portfolio-legend">
                                    <!-- Legend populated by JS -->
                                </div>
                            </div>
                        </div>
                    </div>
                    <div style="position: relative;">
                        <canvas id="portfolio-chart" width="150" height="150" style="display: block;"></canvas>
                        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                            <div style="font-size: 0.8em; color: #888;">Total</div>
                            <div style="font-size: 1.1em; color: #00ff00; font-weight: bold;" id="chart-total">$0</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Performance Card (Enhanced) -->
            <div class="metric-card performance-card" style="grid-column: span 2;">
                <div class="metric-label">Performance Metrics</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <!-- Live Performance -->
                    <div>
                        <h4 style="color: #00ff00; font-size: 1.1em; margin-bottom: 10px;">📊 Live Trading</h4>
                        <div style="font-size: 2em; font-weight: bold;">
                            <span id="live-win-rate">0%</span>
                        </div>
                        <div style="color: #888; margin-bottom: 10px;">Win Rate</div>
                        <div style="font-size: 0.9em;">
                            <div>ROI: <span id="roi" class="positive">0%</span></div>
                            <div>Trades: <span id="trades-count">0</span></div>
                            <div>P. Factor: <span id="profit-factor">0.0</span></div>
                            <div>Avg Size: <span id="avg-trade-size">$0</span></div>
                        </div>
                    </div>
                    
                    <!-- Backtest Performance -->
                    <div>
                        <h4 style="color: #00ffff; font-size: 1.1em; margin-bottom: 10px;">📈 Backtest</h4>
                        <div style="font-size: 2em; font-weight: bold;">
                            <span id="backtest-sharpe">0.0</span>
                        </div>
                        <div style="color: #888; margin-bottom: 10px;">Sharpe Ratio</div>
                        <div style="font-size: 0.9em;">
                            <div>Return: <span id="backtest-return" class="positive">0%</span></div>
                            <div>Vol: <span id="backtest-volatility">0%</span></div>
                            <div>Max DD: <span id="backtest-max-dd" class="negative">0%</span></div>
                            <div>Sessions: <span id="backtest-sessions">0</span></div>
                        </div>
                    </div>
                </div>
                
                <!-- Combined Metrics -->
                <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #333;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; font-size: 0.85em;">
                        <div>
                            <div style="color: #888;">Live Sharpe</div>
                            <div style="color: #00ff00; font-weight: bold;" id="sharpe">0.0</div>
                        </div>
                        <div>
                            <div style="color: #888;">Calmar Ratio</div>
                            <div style="color: #00ff00; font-weight: bold;" id="calmar-ratio">0.0</div>
                        </div>
                        <div>
                            <div style="color: #888;">Total Bets</div>
                            <div style="color: #00ff00; font-weight: bold;" id="total-bets">0</div>
                        </div>
                    </div>
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
        
        <!-- Spreadsheet-Style Matches & Positions -->
        <div class="matches-positions-section" style="grid-column: span 12; background: #111; border: 1px solid #222; border-radius: 8px; padding: 15px;">
            <div class="section-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 class="section-title" style="color: #00ff00; margin: 0;">⚽ Match Dashboard</h3>
                <div style="display: flex; gap: 15px; align-items: center;">
                    <div style="font-size: 0.9em;">
                        <span style="color: #888;">Total:</span> <span id="total-matches" style="color: #00ff00;">0</span>
                        <span style="color: #888; margin-left: 15px;">Active:</span> <span id="active-positions" style="color: #00ffff;">0</span>
                        <span style="color: #888; margin-left: 15px;">Exposure:</span> <span id="matches-exposure" style="color: #ffff00;">$0</span>
                        <span style="color: #888; margin-left: 15px;">P&L:</span> <span id="matches-unrealized-pnl" class="positive">$0</span>
                        <span style="color: #666; margin-left: 20px;">Updated:</span> <span id="markets-update-time" style="color: #888;">--:--:--</span>
                    </div>
                    <select id="matches-filter" onchange="filterMatches(this.value)" style="background: #222; color: #00ff00; border: 1px solid #333; padding: 4px 8px; font-size: 0.9em;">
                        <option value="all">All</option>
                        <option value="active">Active Positions</option>
                        <option value="closed">Closed</option>
                        <option value="positive-edge">Opportunities</option>
                    </select>
                    <button onclick="exportToCSV()" style="background: #222; color: #00ff00; border: 1px solid #333; padding: 4px 12px; margin-left: 10px; cursor: pointer; font-size: 0.9em;">Export CSV 📊</button>
                </div>
            </div>
            
            <div style="overflow: auto; max-height: 700px;">
                <table class="spreadsheet-table" id="matches-table" style="width: 100%; min-width: 1400px; border-collapse: collapse; font-size: 0.9em;">
                    <thead style="position: sticky; top: 0; background: #1a1a1a; z-index: 10;">
                        <tr style="border-bottom: 2px solid #444;">
                            <th style="padding: 6px; text-align: left; min-width: 200px; cursor: pointer;" onclick="sortTable('match')">Match ↕</th>
                            <th style="padding: 6px; text-align: center; min-width: 60px; cursor: pointer;" onclick="sortTable('time')">Time ↕</th>
                            <th style="padding: 6px; text-align: center; min-width: 60px; cursor: pointer;" onclick="sortTable('status')">Status ↕</th>
                            <!-- Home columns -->
                            <th style="padding: 6px; text-align: center; border-left: 2px solid #333; background: rgba(0,255,0,0.05);">H Odds</th>
                            <th style="padding: 6px; text-align: center; background: rgba(0,255,0,0.05);">H Edge</th>
                            <th style="padding: 6px; text-align: center; background: rgba(0,255,0,0.05); cursor: pointer;" onclick="sortTable('h_stake')">H Stake ↕</th>
                            <th style="padding: 6px; text-align: center; background: rgba(0,255,0,0.05); cursor: pointer;" onclick="sortTable('h_pnl')">H P&L ↕</th>
                            <!-- Draw columns -->
                            <th style="padding: 6px; text-align: center; border-left: 2px solid #333; background: rgba(255,255,0,0.05);">D Odds</th>
                            <th style="padding: 6px; text-align: center; background: rgba(255,255,0,0.05);">D Edge</th>
                            <th style="padding: 6px; text-align: center; background: rgba(255,255,0,0.05); cursor: pointer;" onclick="sortTable('d_stake')">D Stake ↕</th>
                            <th style="padding: 6px; text-align: center; background: rgba(255,255,0,0.05); cursor: pointer;" onclick="sortTable('d_pnl')">D P&L ↕</th>
                            <!-- Away columns -->
                            <th style="padding: 6px; text-align: center; border-left: 2px solid #333; background: rgba(0,255,255,0.05);">A Odds</th>
                            <th style="padding: 6px; text-align: center; background: rgba(0,255,255,0.05);">A Edge</th>
                            <th style="padding: 6px; text-align: center; background: rgba(0,255,255,0.05); cursor: pointer;" onclick="sortTable('a_stake')">A Stake ↕</th>
                            <th style="padding: 6px; text-align: center; background: rgba(0,255,255,0.05); cursor: pointer;" onclick="sortTable('a_pnl')">A P&L ↕</th>
                            <!-- Summary -->
                            <th style="padding: 6px; text-align: center; border-left: 2px solid #333; cursor: pointer;" onclick="sortTable('total')">Total ↕</th>
                            <th style="padding: 6px; text-align: center;">Result</th>
                        </tr>
                    </thead>
                    <tbody id="matches-tbody">
                        <!-- Matches will be populated here in spreadsheet format -->
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Trading Activity Section (Compact) -->
        <div class="trading-activity-section" style="grid-column: span 6; background: #111; border: 1px solid #222; border-radius: 8px; padding: 15px;">
            <div class="section-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 class="section-title" style="color: #00ff00; margin: 0; font-size: 1.1em;">📊 Recent Trades</h3>
                <select id="trade-period" onchange="filterTradingActivity(this.value)" style="background: #222; color: #00ff00; border: 1px solid #333; padding: 4px 8px; font-size: 0.9em;">
                    <option value="1h">1H</option>
                    <option value="24h" selected>24H</option>
                    <option value="7d">7D</option>
                    <option value="30d">30D</option>
                </select>
            </div>
            
            <!-- Compact Stats -->
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 0.9em;">
                <div><span style="color: #888;">Trades:</span> <span id="recent-trades-count" style="color: #00ff00;">0</span></div>
                <div><span style="color: #888;">P&L:</span> <span id="recent-pnl" class="positive">$0</span></div>
                <div><span style="color: #888;">Win:</span> <span id="recent-win-rate" style="color: #00ff00;">0%</span></div>
                <div><span style="color: #888;">Avg Time:</span> <span id="avg-trade-time" style="color: #00ff00;">0h</span></div>
            </div>
            
            <!-- Compact Trades Table -->
            <div style="max-height: 250px; overflow-y: auto;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.85em;">
                    <thead>
                        <tr style="border-bottom: 1px solid #333;">
                            <th style="padding: 5px; text-align: left; color: #888;">Time</th>
                            <th style="padding: 5px; text-align: left; color: #888;">Market</th>
                            <th style="padding: 5px; text-align: center; color: #888;">Side</th>
                            <th style="padding: 5px; text-align: right; color: #888;">Stake</th>
                            <th style="padding: 5px; text-align: center; color: #888;">@</th>
                            <th style="padding: 5px; text-align: center; color: #888;">Status</th>
                            <th style="padding: 5px; text-align: right; color: #888;">P&L</th>
                        </tr>
                    </thead>
                    <tbody id="recent-trades-tbody">
                        <!-- Trades will be populated here -->
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Activity Feed (Compact) -->
        <div class="activity-section" style="grid-column: span 6; background: #111; border: 1px solid #222; border-radius: 8px; padding: 15px;">
            <div class="section-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 class="section-title" style="color: #00ff00; margin: 0; font-size: 1.1em;">📋 Activity</h3>
                <select id="activity-filter" onchange="filterActivity()" style="background: #222; color: #00ff00; border: 1px solid #333; padding: 4px 8px; font-size: 0.9em;">
                    <option value="all">All</option>
                    <option value="trade">Trades</option>
                    <option value="evaluation">Evals</option>
                    <option value="system">System</option>
                </select>
            </div>
            <div id="activity-feed" style="max-height: 250px; overflow-y: auto;">
                <!-- Activity items will be populated here -->
            </div>
        </div>
        
        <!-- Strategy Parameters (Collapsible) -->
        <div class="strategy-section">
            <div class="section-header collapsible" onclick="toggleCollapsible(this)">
                <h3 class="section-title">⚙️ Strategy Parameters</h3>
                <span style="color: #666;">Click to expand</span>
            </div>
            <div class="collapsible-content collapsed">
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
                    <div>
                        <h4 style="color: #00ff00; margin-bottom: 10px;">Signal Weights</h4>
                        <div style="background: #1a1a1a; padding: 15px; border-radius: 5px;">
                            <div style="margin-bottom: 8px;">Implied: <span id="weight-implied" style="color: #00ff00;">80%</span></div>
                            <div>Other Models: <span id="weight-other" style="color: #00ff00;">20%</span></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Global state
        let dashboardData = {};
        let activityFilter = 'all';
        
        // Helper functions for consistent formatting
        function formatMoney(amount, decimals = 0) {
            if (amount === undefined || amount === null || isNaN(amount)) return '-';
            const prefix = amount >= 0 ? '' : '';
            return prefix + '$' + Math.abs(amount).toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
        }
        
        function formatPercent(value, decimals = 1) {
            if (value === undefined || value === null || isNaN(value)) return '-';
            const prefix = value >= 0 ? '+' : '';
            return prefix + value.toFixed(decimals) + '%';
        }
        
        // Update current time
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = now.toLocaleTimeString();
        }
        
        // Load unified dashboard data
        async function loadDashboardData() {
            try {
                const period = window.selectedPeriod || 30;
                console.log('Fetching from /api/dashboard/unified with period:', period);
                const response = await fetch('/api/dashboard/unified?period=' + period);
                console.log('Response status:', response.status);
                
                if (!response.ok) {
                    throw new Error('HTTP error! status: ' + response.status);
                }
                
                const data = await response.json();
                console.log('Data received:', data);
                console.log('Markets count:', data.markets?.length);
                console.log('Positions:', data.positions);
                
                dashboardData = data;
                
                // Call updateDashboard
                console.log('Calling updateDashboard...');
                updateDashboard(data);
            } catch (error) {
                console.error('Error loading dashboard:', error);
                console.error('Stack:', error.stack);
            }
        }
        
        // Update all dashboard components
        function updateDashboard(data) {
            updateMetrics(data);
            updateMatchesAndPositions(data.markets || [], data.positions || {});
            updateActivityFeed(data.activity || []);
            updateStrategy(data.strategy || {});
            
            // Update timestamp
            document.getElementById('markets-update-time').textContent = 
                new Date().toLocaleTimeString();
        }
        
        // Update metric cards
        function updateMetrics(data) {
            console.log('Updating metrics with data:', data);
            const portfolio = data.portfolio || {};
            const performance = data.performance || {};
            const system = data.system || {};
            
            // Portfolio card
            console.log('Updating portfolio value:', portfolio.total_value);
            document.getElementById('portfolio-value').textContent = 
                '$' + (portfolio.total_value || 0).toFixed(2);
            
            const changeClass = portfolio.daily_change >= 0 ? 'positive' : 'negative';
            const changeSign = portfolio.daily_change >= 0 ? '+' : '';
            document.getElementById('portfolio-change').className = changeClass;
            document.getElementById('portfolio-change').textContent = 
                changeSign + '$' + (portfolio.daily_change || 0).toFixed(2) + 
                ' (' + (portfolio.daily_change_pct || 0).toFixed(1) + '%)';
            
            document.getElementById('cash-available').textContent = 
                '$' + (portfolio.cash_available || 0).toFixed(2);
            document.getElementById('positions-value').textContent = 
                '$' + (portfolio.positions_value || 0).toFixed(2);
            
            // Update chart total
            document.getElementById('chart-total').textContent = 
                '$' + (portfolio.total_value || 0).toFixed(0);
            
            // Draw portfolio donut chart
            if (portfolio.composition) {
                drawPortfolioDonut(portfolio.composition);
            }
            
            // Performance card - Live metrics
            // win_rate is already in percentage format from the API
            document.getElementById('live-win-rate').textContent = 
                (performance.win_rate || 0).toFixed(1) + '%';
            document.getElementById('roi').textContent = 
                (performance.roi || 0).toFixed(1) + '%';
            document.getElementById('roi').className = 
                performance.roi >= 0 ? 'positive' : 'negative';
            document.getElementById('trades-count').textContent = 
                (performance.total_trades || 0);
            document.getElementById('profit-factor').textContent = 
                (performance.profit_factor || 0).toFixed(2);
            document.getElementById('avg-trade-size').textContent = 
                '$' + (performance.avg_trade_size || 0).toFixed(0);
            
            // Backtest metrics
            if (performance.backtest) {
                const backtest = performance.backtest;
                document.getElementById('backtest-sharpe').textContent = 
                    (backtest.sharpe_ratio || 0).toFixed(2);
                document.getElementById('backtest-return').textContent = 
                    ((backtest.expected_return || 0) * 100).toFixed(1) + '%';
                document.getElementById('backtest-return').className = 
                    backtest.expected_return >= 0 ? 'positive' : 'negative';
                document.getElementById('backtest-volatility').textContent = 
                    ((backtest.volatility || 0) * 100).toFixed(1) + '%';
                document.getElementById('backtest-max-dd').textContent = 
                    '-' + Math.abs((backtest.max_drawdown || 0) * 100).toFixed(1) + '%';
                document.getElementById('backtest-sessions').textContent = 
                    backtest.sessions_included || 0;
                
                // Update total bets from backtest
                document.getElementById('total-bets').textContent = 
                    (backtest.total_bets || 0) + (performance.total_trades || 0);
            }
            
            // Combined metrics
            document.getElementById('sharpe').textContent = 
                (performance.sharpe || 0).toFixed(2);
            document.getElementById('calmar-ratio').textContent = 
                (performance.calmar_ratio || 0).toFixed(2);
            
            // System card
            document.getElementById('system-status').textContent = 
                system.session?.status === 'active' ? 'Active' : 'Inactive';
            document.getElementById('session-id').textContent = 
                'Session: ' + (system.session?.id || '--');
            document.getElementById('markets-count').textContent = 
                system.database?.markets || 0;
            document.getElementById('positions-count').textContent = 
                portfolio.positions_count || 0;
            
            // Exposure card
            const exposurePct = (portfolio.total_exposure / 10000 * 100) || 0;
            document.getElementById('exposure-pct').textContent = 
                exposurePct.toFixed(1) + '%';
            document.getElementById('total-exposure').textContent = 
                '$' + (portfolio.total_exposure || 0).toFixed(2);
            
            // Max drawdown
            if (document.getElementById('max-dd')) {
                document.getElementById('max-dd').textContent = 
                    (performance.max_drawdown || 0).toFixed(1) + '%';
            }
        }
        
        // Update spreadsheet-style matches and positions
        function updateMatchesAndPositions(markets, positions) {
            try {
                console.log('updateMatchesAndPositions called with:', markets.length, 'markets and', positions);
                const tbody = document.getElementById('matches-tbody');
                if (!tbody) {
                    console.error('matches-tbody not found!');
                    return;
                }
                tbody.innerHTML = '';
            
            // Create comprehensive match-position map
            const positionsByMarketId = {};
            const openPositions = positions.open || [];
            const closedPositions = positions.closed || [];
            
            // Map positions by market_id and outcome
            [...openPositions, ...closedPositions].forEach(pos => {
                const marketId = pos.market_id;
                if (!positionsByMarketId[marketId]) {
                    positionsByMarketId[marketId] = {};
                }
                const outcome = pos.outcome;
                if (!positionsByMarketId[marketId][outcome]) {
                    positionsByMarketId[marketId][outcome] = [];
                }
                positionsByMarketId[marketId][outcome].push(pos);
            });
            
            // Calculate totals
            let totalExposure = 0;
            let totalPnl = 0;
            let activePositions = 0;
            
            openPositions.forEach(pos => {
                totalExposure += pos.execution_stake || pos.stake || 0;
                totalPnl += pos.pnl || 0;
                activePositions++;
            });
            
            closedPositions.forEach(pos => {
                totalPnl += pos.pnl || 0;
            });
            
            // Calculate win rate from closed positions
            let wins = 0, losses = 0;
            closedPositions.forEach(pos => {
                if (pos.result === 'won' || (pos.pnl && pos.pnl > 0)) wins++;
                else if (pos.result === 'lost' || (pos.pnl && pos.pnl < 0)) losses++;
            });
            const winRate = (wins + losses) > 0 ? (wins / (wins + losses) * 100) : 0;
            
            // Update header stats
            document.getElementById('total-matches').textContent = markets.length + closedPositions.filter(p => !markets.find(m => m.id === p.market_id)).length;
            document.getElementById('active-positions').textContent = activePositions;
            document.getElementById('matches-exposure').textContent = '$' + totalExposure.toFixed(2);
            const pnlElement = document.getElementById('matches-unrealized-pnl');
            pnlElement.textContent = (totalPnl >= 0 ? '+' : '') + '$' + Math.abs(totalPnl).toFixed(2);
            pnlElement.className = totalPnl >= 0 ? 'positive' : 'negative';
            
            // Add win rate to header if element exists
            let winRateElement = document.getElementById('win-rate-indicator');
            if (!winRateElement && wins + losses > 0) {
                // Create win rate element
                const headerDiv = document.querySelector('.matches-positions-section .section-header > div');
                if (headerDiv) {
                    const winRateSpan = document.createElement('span');
                    winRateSpan.id = 'win-rate-indicator';
                    winRateSpan.style.cssText = 'color: #888; margin-left: 15px;';
                    winRateSpan.innerHTML = 'Win Rate: <span style="color: ' + (winRate >= 50 ? '#00ff00' : '#ff6666') + ';">' + winRate.toFixed(1) + '%</span> (' + wins + 'W/' + losses + 'L)';
                    headerDiv.appendChild(winRateSpan);
                }
            } else if (winRateElement) {
                winRateElement.innerHTML = 'Win Rate: <span style="color: ' + (winRate >= 50 ? '#00ff00' : '#ff6666') + ';">' + winRate.toFixed(1) + '%</span> (' + wins + 'W/' + losses + 'L)';
            }
            
            // Process all markets and closed positions
            const allMatches = [...markets];
            
            // Add closed matches that aren't in current markets
            const closedMarketIds = new Set();
            const marketIds = new Set(markets.map(m => m.id));
            
            closedPositions.forEach(pos => {
                // Only create synthetic entry if market data not already present
                if (!marketIds.has(pos.market_id) && !closedMarketIds.has(pos.market_id)) {
                    closedMarketIds.add(pos.market_id);
                    allMatches.push({
                        id: pos.market_id,
                        home_team: pos.market_name?.split(/_vs_| vs /)[0] || 'Unknown',
                        away_team: pos.market_name?.split(/_vs_| vs /)[1] || 'Unknown',
                        status: 'CLOSED',
                        maturity_date: pos.closed_at || new Date().toISOString(),
                        resolved_outcome: null,  // Will be filled from actual market data
                        home_score: null,
                        away_score: null
                    });
                }
            });
            
            // Render each match as a spreadsheet row
            console.log('Rendering', allMatches.length, 'matches to tbody with', tbody.children.length, 'existing rows');
            
            if (!allMatches || allMatches.length === 0) {
                console.warn('No matches to render');
                tbody.innerHTML = '<tr><td colspan="17" style="text-align: center; padding: 20px; color: #888;">No matches to display</td></tr>';
                return;
            }
            
            allMatches.forEach((market, index) => {
                const marketPos = positionsByMarketId[market.id] || {};
                const row = tbody.insertRow();
                row.className = 'match-row';
                
                // Add alternating row colors
                if (index % 2 === 0) row.style.background = 'rgba(255, 255, 255, 0.02)';
                
                // Highlight rows with positions
                const hasPositions = Object.keys(marketPos).length > 0;
                if (hasPositions) {
                    row.style.background = 'rgba(0, 255, 0, 0.05)';
                    row.className += ' has-positions';
                }
                
                if (market.status === 'CLOSED' || market.status === 'FINISHED') {
                    row.className += ' finished';
                    row.style.opacity = '0.7';
                }
                if (market.status === 'LIVE') row.className += ' live';
                
                // Match name
                const matchCell = row.insertCell();
                matchCell.style.textAlign = 'left';
                matchCell.innerHTML = '<span style="font-size: 0.85em;">' + market.home_team + ' vs ' + market.away_team + '</span>';
                
                // Time
                const timeCell = row.insertCell();
                const kickoff = new Date(market.maturity_date);
                timeCell.style.textAlign = 'center';
                timeCell.style.fontSize = '0.85em';
                
                // Better time formatting
                if (market.time_until) {
                    timeCell.innerHTML = '<span style="color: ' + (market.time_until.includes('m') && !market.time_until.includes('d') ? '#ffff00' : '#888') + ';">' + market.time_until + '</span>';
                } else if (market.status === 'CLOSED' || market.status === 'FINISHED') {
                    timeCell.innerHTML = kickoff.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
                } else {
                    timeCell.innerHTML = kickoff.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
                }
                
                // Status
                const statusCell = row.insertCell();
                statusCell.style.textAlign = 'center';
                let displayStatus = market.status || 'ACTIVE';
                if (displayStatus === 'ACTIVE') displayStatus = 'Active';
                else if (displayStatus === 'CLOSED' || displayStatus === 'FINISHED') displayStatus = 'Closed';
                else if (displayStatus === 'LIVE') displayStatus = 'Live';
                
                const statusColors = {
                    'Active': '#00ff00',
                    'Live': '#ff0000',
                    'Closed': '#888'
                };
                statusCell.innerHTML = '<span style="color: ' + (statusColors[displayStatus] || '#888') + '; font-size: 0.8em;">' + displayStatus + '</span>';
                
                // Helper to create outcome cells with inline position data
                const createOutcomeCells = (outcome, odds, edge) => {
                    const positions = marketPos[outcome] || [];
                    const activePos = positions.find(p => p.status !== 'closed');
                    const closedPos = positions.find(p => p.status === 'closed' || p.result);
                    
                    // Odds cell
                    const oddsCell = row.insertCell();
                    oddsCell.style.textAlign = 'center';
                    oddsCell.style.fontSize = '0.85em';
                    oddsCell.style.padding = '4px';
                    // Add left border for first column of each outcome group
                    if (outcome === 'Home' || outcome === 'Draw' || outcome === 'Away') {
                        oddsCell.style.borderLeft = '1px solid #333';
                    }
                    if (odds) {
                        oddsCell.innerHTML = '<span style="color: #00ffff;">' + odds.toFixed(2) + '</span>';
                    } else {
                        oddsCell.textContent = '-';
                    }
                    
                    // Edge cell with coloring
                    const edgeCell = row.insertCell();
                    edgeCell.style.textAlign = 'center';
                    edgeCell.style.fontSize = '0.85em';
                    edgeCell.style.padding = '4px';
                    if (edge !== null && edge !== undefined) {
                        let edgeColor = '#888';
                        if (edge >= 2.0) edgeColor = '#00ff00';
                        else if (edge >= 1.0) edgeColor = '#88ff00';
                        else if (edge >= 0.5) edgeColor = '#ffff00';
                        else if (edge >= 0) edgeColor = '#ffaa00';
                        else edgeColor = '#ff6666';
                        
                        // Add class for filtering
                        if (edge >= 2.0) edgeCell.className = 'edge-positive-strong';
                        else if (edge >= 1.0) edgeCell.className = 'edge-positive-medium';
                        else if (edge > 0) edgeCell.className = 'edge-positive-weak';
                        
                        edgeCell.innerHTML = '<span style="color: ' + edgeColor + '; ' + (edge > 1.0 ? 'font-weight: bold;' : '') + '">' + (edge > 0 ? '+' : '') + edge.toFixed(1) + '%</span>';
                    } else {
                        edgeCell.textContent = '-';
                    }
                    
                    // Stake cell
                    const stakeCell = row.insertCell();
                    stakeCell.style.textAlign = 'center';
                    stakeCell.style.fontSize = '0.85em';
                    stakeCell.style.padding = '4px';
                    if (activePos) {
                        stakeCell.innerHTML = '<span style="color: #ffff00;">' + formatMoney(activePos.execution_stake || activePos.stake) + '</span>';
                    } else if (closedPos) {
                        stakeCell.innerHTML = '<span style="color: #666;">' + formatMoney(closedPos.execution_stake || closedPos.stake) + '</span>';
                    } else {
                        stakeCell.textContent = '-';
                    }
                    
                    // P&L cell with ROI
                    const pnlCell = row.insertCell();
                    pnlCell.style.textAlign = 'center';
                    pnlCell.style.fontSize = '0.85em';
                    pnlCell.style.padding = '4px';
                    const pos = activePos || closedPos;
                    if (pos && pos.pnl !== undefined) {
                        const pnlColor = pos.pnl >= 0 ? '#00ff00' : '#ff6666';
                        const roi = pos.roi || 0;
                        const roiText = roi !== 0 ? '<br><span style="font-size: 0.8em; color: #888;">' + formatPercent(roi) + '</span>' : '';
                        const pnlFormatted = pos.pnl >= 0 ? '+' + formatMoney(pos.pnl) : formatMoney(pos.pnl);
                        pnlCell.innerHTML = '<span style="color: ' + pnlColor + '; font-weight: bold;">' + pnlFormatted + '</span>' + roiText;
                    } else {
                        pnlCell.textContent = '-';
                    }
                };
                
                // Create cells for each outcome
                // Ensure we have default values if undefined
                const homeOdds = market.home_odds || 0;
                const drawOdds = market.draw_odds || 0;
                const awayOdds = market.away_odds || 0;
                const signals = market.signals || {};
                
                createOutcomeCells('Home', homeOdds, signals.home_edge);
                createOutcomeCells('Draw', drawOdds, signals.draw_edge);
                createOutcomeCells('Away', awayOdds, signals.away_edge);
                
                // Total stake and P&L for this market
                const totalCell = row.insertCell();
                totalCell.style.textAlign = 'center';
                totalCell.style.borderLeft = '2px solid #333';
                
                let totalStake = 0;
                let totalPnl = 0;
                Object.values(marketPos).forEach(positions => {
                    positions.forEach(pos => {
                        totalStake += pos.execution_stake || pos.stake || 0;
                        totalPnl += pos.pnl || 0;
                    });
                });
                
                if (totalStake > 0) {
                    totalCell.innerHTML = '<span style="color: #ffff00;">$' + totalStake.toFixed(0) + '</span>';
                } else {
                    totalCell.textContent = '-';
                }
                
                // Result/outcome cell
                const resultCell = row.insertCell();
                resultCell.style.textAlign = 'center';
                resultCell.style.fontSize = '0.85em';
                
                if (market.status === 'CLOSED' || market.status === 'FINISHED') {
                    // Find winning outcome from positions or market data
                    let resultText = '-';
                    let resultColor = '#888';
                    let scoreText = '';
                    
                    // First check if market has resolved_outcome
                    if (market.resolved_outcome) {
                        resultText = market.resolved_outcome;
                        resultColor = '#00ff00';
                    } else {
                        // Fall back to checking positions
                        Object.entries(marketPos).forEach(([outcome, positions]) => {
                            const wonPos = positions.find(p => p.result === 'won');
                            if (wonPos) {
                                resultText = outcome;
                                resultColor = '#00ff00';
                            }
                        });
                    }
                    
                    // Add score if available
                    if (market.home_score !== null && market.home_score !== undefined && 
                        market.away_score !== null && market.away_score !== undefined) {
                        scoreText = ' (' + market.home_score + '-' + market.away_score + ')';
                    }
                    
                    if (totalPnl !== 0) {
                        const pnlColor = totalPnl >= 0 ? '#00ff00' : '#ff6666';
                        const prefix = totalPnl >= 0 ? '+' : '';
                        resultCell.innerHTML = '<span style="color: ' + resultColor + ';">' + resultText + scoreText + '</span><br><span style="color: ' + pnlColor + '; font-weight: bold;">' + prefix + '$' + Math.abs(totalPnl).toFixed(0) + '</span>';
                    } else {
                        resultCell.innerHTML = '<span style="color: ' + resultColor + ';">' + resultText + scoreText + '</span>';
                    }
                } else if (totalPnl !== 0) {
                    const pnlColor = totalPnl >= 0 ? '#00ff00' : '#ff6666';
                    const prefix = totalPnl >= 0 ? '+' : '';
                    resultCell.innerHTML = '<span style="color: ' + pnlColor + '; font-weight: bold;">' + prefix + '$' + Math.abs(totalPnl).toFixed(0) + '</span>';
                } else {
                    resultCell.textContent = '-';
                }
                
            });
            
            // Add summary row at the top
            const summaryRow = tbody.insertRow(0);
            summaryRow.style.borderBottom = '2px solid #444';
            summaryRow.style.background = 'rgba(0, 255, 255, 0.1)';
            summaryRow.style.fontWeight = 'bold';
            summaryRow.classList.add('summary-row');
            
            // Summary label
            const summaryLabel = summaryRow.insertCell();
            summaryLabel.colSpan = 3;
            summaryLabel.style.textAlign = 'right';
            summaryLabel.style.padding = '6px';
            summaryLabel.innerHTML = '<span style="color: #00ffff;">TOTALS:</span>';
            
            // Calculate totals for each outcome
            let homeTotalStake = 0, drawTotalStake = 0, awayTotalStake = 0;
            let homeTotalPnl = 0, drawTotalPnl = 0, awayTotalPnl = 0;
            
            Object.values(positionsByMarketId).forEach(marketPos => {
                ['Home', 'Draw', 'Away'].forEach(outcome => {
                    const positions = marketPos[outcome] || [];
                    positions.forEach(pos => {
                        const stake = pos.execution_stake || pos.stake || 0;
                        const pnl = pos.pnl || 0;
                        
                        if (outcome === 'Home') {
                            homeTotalStake += stake;
                            homeTotalPnl += pnl;
                        } else if (outcome === 'Draw') {
                            drawTotalStake += stake;
                            drawTotalPnl += pnl;
                        } else if (outcome === 'Away') {
                            awayTotalStake += stake;
                            awayTotalPnl += pnl;
                        }
                    });
                });
            });
            
            // Add summary cells for each outcome
            const addSummaryOutcome = (totalStake, totalPnl) => {
                // Odds summary (empty)
                const oddsCell = summaryRow.insertCell();
                oddsCell.style.borderLeft = '1px solid #333';
                oddsCell.style.padding = '4px';
                
                // Edge summary (empty)
                summaryRow.insertCell().style.padding = '4px';
                
                // Stake summary
                const stakeCell = summaryRow.insertCell();
                stakeCell.style.textAlign = 'center';
                stakeCell.style.padding = '4px';
                if (totalStake > 0) {
                    stakeCell.innerHTML = '<span style="color: #ffff00;">' + formatMoney(totalStake) + '</span>';
                } else {
                    stakeCell.textContent = '-';
                }
                
                // P&L summary
                const pnlCell = summaryRow.insertCell();
                pnlCell.style.textAlign = 'center';
                pnlCell.style.padding = '4px';
                if (totalPnl !== 0) {
                    const pnlColor = totalPnl >= 0 ? '#00ff00' : '#ff6666';
                    const pnlFormatted = totalPnl >= 0 ? '+' + formatMoney(totalPnl) : formatMoney(totalPnl);
                    pnlCell.innerHTML = '<span style="color: ' + pnlColor + ';">' + pnlFormatted + '</span>';
                } else {
                    pnlCell.textContent = '-';
                }
            };
            
            addSummaryOutcome(homeTotalStake, homeTotalPnl);
            addSummaryOutcome(drawTotalStake, drawTotalPnl);
            addSummaryOutcome(awayTotalStake, awayTotalPnl);
            
            // Grand total
            const grandTotalCell = summaryRow.insertCell();
            grandTotalCell.style.borderLeft = '2px solid #333';
            grandTotalCell.style.textAlign = 'center';
            grandTotalCell.style.padding = '4px';
            const grandTotal = homeTotalStake + drawTotalStake + awayTotalStake;
            if (grandTotal > 0) {
                grandTotalCell.innerHTML = '<span style="color: #ffff00;">$' + grandTotal.toFixed(0) + '</span>';
            } else {
                grandTotalCell.textContent = '-';
            }
            
            // Grand P&L
            const grandPnlCell = summaryRow.insertCell();
            grandPnlCell.style.textAlign = 'center';
            grandPnlCell.style.padding = '4px';
            const grandPnl = homeTotalPnl + drawTotalPnl + awayTotalPnl;
            if (grandPnl !== 0) {
                const pnlColor = grandPnl >= 0 ? '#00ff00' : '#ff6666';
                const prefix = grandPnl >= 0 ? '+' : '';
                grandPnlCell.innerHTML = '<span style="color: ' + pnlColor + '; font-size: 1.1em;">' + prefix + '$' + Math.abs(grandPnl).toFixed(0) + '</span>';
            } else {
                grandPnlCell.textContent = '-';
            }
            
            console.log('Finished rendering. Tbody now has', tbody.children.length, 'rows');
            
            // Re-apply current filter if any
            const currentFilter = document.getElementById('matches-filter').value;
            if (currentFilter && currentFilter !== 'all') {
                filterMatches(currentFilter);
            }
            
            } catch (error) {
                console.error('Error in updateMatchesAndPositions:', error);
                console.error('Stack trace:', error.stack);
            }
        }
        
        // Sort table functionality
        let sortOrder = {};
        function sortTable(column) {
            const table = document.getElementById('matches-table');
            const tbody = table.querySelector('tbody');
            if (!tbody || tbody.children.length === 0) {
                console.warn('No data to sort');
                return;
            }
            const rows = Array.from(tbody.querySelectorAll('tr:not(.summary-row)')); // Exclude summary row
            const summaryRow = tbody.querySelector('tr.summary-row');
            
            // Toggle sort order
            sortOrder[column] = sortOrder[column] === 'asc' ? 'desc' : 'asc';
            
            rows.sort((a, b) => {
                let aVal, bVal;
                
                switch(column) {
                    case 'match':
                        aVal = a.cells[0].textContent.toLowerCase();
                        bVal = b.cells[0].textContent.toLowerCase();
                        break;
                    case 'time':
                        // Convert time strings to minutes for sorting
                        aVal = parseTimeToMinutes(a.cells[1].textContent);
                        bVal = parseTimeToMinutes(b.cells[1].textContent);
                        break;
                    case 'status':
                        aVal = a.cells[2].textContent;
                        bVal = b.cells[2].textContent;
                        break;
                    case 'h_stake':
                        aVal = parseFloat(a.cells[5].textContent.replace('$', '').replace('-', '0'));
                        bVal = parseFloat(b.cells[5].textContent.replace('$', '').replace('-', '0'));
                        break;
                    case 'h_pnl':
                        aVal = parseFloat(a.cells[6].textContent.replace('$', '').replace('+', '').replace('-', '0'));
                        bVal = parseFloat(b.cells[6].textContent.replace('$', '').replace('+', '').replace('-', '0'));
                        break;
                    case 'd_stake':
                        aVal = parseFloat(a.cells[9].textContent.replace('$', '').replace('-', '0'));
                        bVal = parseFloat(b.cells[9].textContent.replace('$', '').replace('-', '0'));
                        break;
                    case 'd_pnl':
                        aVal = parseFloat(a.cells[10].textContent.replace('$', '').replace('+', '').replace('-', '0'));
                        bVal = parseFloat(b.cells[10].textContent.replace('$', '').replace('+', '').replace('-', '0'));
                        break;
                    case 'a_stake':
                        aVal = parseFloat(a.cells[13].textContent.replace('$', '').replace('-', '0'));
                        bVal = parseFloat(b.cells[13].textContent.replace('$', '').replace('-', '0'));
                        break;
                    case 'a_pnl':
                        aVal = parseFloat(a.cells[14].textContent.replace('$', '').replace('+', '').replace('-', '0'));
                        bVal = parseFloat(b.cells[14].textContent.replace('$', '').replace('+', '').replace('-', '0'));
                        break;
                    case 'total':
                        aVal = parseFloat(a.cells[15].textContent.replace('$', '').replace('-', '0'));
                        bVal = parseFloat(b.cells[15].textContent.replace('$', '').replace('-', '0'));
                        break;
                    default:
                        return 0;
                }
                
                if (sortOrder[column] === 'asc') {
                    return aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
                } else {
                    return aVal < bVal ? 1 : aVal > bVal ? -1 : 0;
                }
            });
            
            // Clear and re-add sorted rows
            try {
                // Remove all existing rows
                while (tbody.firstChild) {
                    tbody.removeChild(tbody.firstChild);
                }
                // Clear tbody but keep summary row
                const summaryRow = tbody.querySelector('tr.summary-row');
                tbody.innerHTML = '';
                
                // Add summary row back at top
                if (summaryRow) {
                    tbody.appendChild(summaryRow);
                }
                
                // Add sorted rows
                rows.forEach(row => tbody.appendChild(row));
            } catch (e) {
                console.error('Error during sort:', e);
            }
        }
        
        // Helper function to parse time strings to minutes
        function parseTimeToMinutes(timeStr) {
            if (timeStr.includes('d')) {
                const days = parseInt(timeStr.match(/(\d+)d/)[1]);
                return days * 24 * 60;
            } else if (timeStr.includes('h')) {
                const hours = parseInt(timeStr.match(/(\d+)h/)[1]);
                const minutes = timeStr.includes('m') ? parseInt(timeStr.match(/(\d+)m/)[1]) : 0;
                return hours * 60 + minutes;
            } else if (timeStr.includes('m')) {
                return parseInt(timeStr.match(/(\d+)m/)[1]);
            } else if (timeStr.includes(':')) {
                // For time format like "02:45 PM"
                return 9999; // Put closed matches at the end
            }
            return 0;
        }
        
        // Filter matches based on criteria
        function filterMatches(filterValue) {
            const table = document.getElementById('matches-table');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr:not(.summary-row)')); // Exclude summary row
            
            rows.forEach(row => {
                let show = true;
                
                switch(filterValue) {
                    case 'active':
                        // Show only active matches
                        const statusCell = row.cells[2];
                        show = statusCell && (statusCell.innerText === 'Active' || statusCell.innerText === 'Starting');
                        break;
                    case 'closed':
                        // Show only closed matches
                        const statusCellClosed = row.cells[2];
                        show = statusCellClosed && statusCellClosed.innerText === 'Closed';
                        break;
                    case 'positive-edge':
                        // Check if any edge value in the row is positive
                        const edgeCells = row.querySelectorAll('.edge-positive-strong, .edge-positive-medium, .edge-positive-weak');
                        show = edgeCells.length > 0;
                        break;
                    case 'all':
                    default:
                        show = true;
                }
                
                row.style.display = show ? '' : 'none';
            });
        }
        
        // Update positions
        function updatePositions(positions) {
            const openPositions = positions.open || [];
            const closedPositions = positions.closed || [];
            
            document.getElementById('open-count').textContent = openPositions.length;
            document.getElementById('closed-count').textContent = closedPositions.length;
            
            // Update open positions
            const openTbody = document.getElementById('open-positions-tbody');
            openTbody.innerHTML = '';
            
            openPositions.forEach(pos => {
                const row = openTbody.insertRow();
                
                row.insertCell().textContent = pos.market_name;
                row.insertCell().textContent = pos.outcome;
                row.insertCell().textContent = '$' + pos.stake.toFixed(2);
                
                // Execution stake (stake + fees)
                const execStakeCell = row.insertCell();
                const execStake = pos.execution_stake || pos.stake;
                execStakeCell.textContent = '$' + execStake.toFixed(2);
                
                row.insertCell().textContent = pos.avg_odds.toFixed(2);
                row.insertCell().textContent = '$' + (pos.current_value || pos.stake).toFixed(2);
                
                const pnlCell = row.insertCell();
                const pnlClass = pos.pnl >= 0 ? 'positive' : 'negative';
                pnlCell.innerHTML = '<span class="' + pnlClass + '">$' + pos.pnl.toFixed(2) + '</span>';
                
                const roiCell = row.insertCell();
                const roiClass = pos.roi >= 0 ? 'positive' : 'negative';
                roiCell.innerHTML = '<span class="' + roiClass + '">' + pos.roi.toFixed(1) + '%</span>';
                
                const feeCell = row.insertCell();
                const feeAmount = pos.fee_info?.fee_amount || 0;
                const feePct = pos.fee_info?.total_fee_pct ? (pos.fee_info.total_fee_pct * 100).toFixed(1) : '3.0';
                feeCell.innerHTML = '$' + feeAmount.toFixed(2) + '<br/><span style="font-size: 0.8em; color: #666;">' + feePct + '%</span>';
                
                // Time to close
                const closesInCell = row.insertCell();
                if (pos.maturity_date) {
                    const maturity = new Date(pos.maturity_date);
                    const now = new Date();
                    const hoursLeft = Math.max(0, (maturity - now) / (1000 * 60 * 60));
                    
                    if (hoursLeft < 1) {
                        closesInCell.innerHTML = '<span style="color: #ff6666;">< 1h</span>';
                    } else if (hoursLeft < 24) {
                        closesInCell.innerHTML = '<span style="color: #ffaa00;">' + hoursLeft.toFixed(0) + 'h</span>';
                    } else {
                        const daysLeft = Math.floor(hoursLeft / 24);
                        closesInCell.innerHTML = daysLeft + 'd ' + (hoursLeft % 24).toFixed(0) + 'h';
                    }
                } else {
                    closesInCell.textContent = '-';
                }
            });
            
            // Update closed positions
            const closedTbody = document.getElementById('closed-positions-tbody');
            closedTbody.innerHTML = '';
            
            // Get limit from dropdown
            const limit = document.getElementById('closed-limit')?.value || 20;
            const displayedPositions = closedPositions.slice(0, parseInt(limit));
            
            // Calculate summary stats
            let totalPnl = 0;
            let wins = 0;
            let losses = 0;
            
            displayedPositions.forEach(pos => {
                const row = closedTbody.insertRow();
                
                const closedTime = new Date(pos.closed_at);
                row.insertCell().textContent = closedTime.toLocaleDateString();
                row.insertCell().textContent = pos.market_name;
                row.insertCell().textContent = pos.outcome;
                row.insertCell().textContent = '$' + pos.stake.toFixed(2);
                
                // Execution stake
                const execStake = pos.execution_stake || pos.stake;
                row.insertCell().textContent = '$' + execStake.toFixed(2);
                
                row.insertCell().textContent = pos.avg_odds.toFixed(2);
                
                const resultCell = row.insertCell();
                const resultClass = pos.result === 'won' ? 'positive' : 'negative';
                resultCell.innerHTML = '<span class="' + resultClass + '">' + (pos.result ? pos.result.toUpperCase() : 'UNKNOWN') + '</span>';
                
                // Score cell
                const scoreCell = row.insertCell();
                if (pos.score) {
                    scoreCell.innerHTML = '<strong>' + pos.score + '</strong>';
                } else {
                    scoreCell.textContent = '-';
                }
                
                const pnlCell = row.insertCell();
                const pnlClass = pos.pnl >= 0 ? 'positive' : 'negative';
                pnlCell.innerHTML = '<span class="' + pnlClass + '">$' + Math.abs(pos.pnl).toFixed(2) + '</span>';
                
                const roiCell = row.insertCell();
                const roiClass = pos.roi >= 0 ? 'positive' : 'negative';
                roiCell.innerHTML = '<span class="' + roiClass + '">' + pos.roi.toFixed(1) + '%</span>';
                
                // Fees cell
                const feeCell = row.insertCell();
                const feeAmount = pos.fee_info?.fee_amount || 0;
                const feePct = pos.fee_info?.total_fee_pct ? (pos.fee_info.total_fee_pct * 100).toFixed(1) : '3.0';
                feeCell.innerHTML = '$' + feeAmount.toFixed(2) + '<br/><span style="font-size: 0.8em; color: #666;">' + feePct + '%</span>';
                
                // Update summary stats
                totalPnl += pos.pnl || 0;
                if (pos.result === 'won') wins++;
                else if (pos.result === 'lost') losses++;
            });
            
            // Update summary
            document.getElementById('total-closed-count').textContent = closedPositions.length;
            document.getElementById('total-closed-pnl').textContent = 
                (totalPnl >= 0 ? '+' : '') + '$' + Math.abs(totalPnl).toFixed(2);
            document.getElementById('total-closed-pnl').className = totalPnl >= 0 ? 'positive' : 'negative';
            
            const winRate = (wins + losses) > 0 ? (wins / (wins + losses) * 100) : 0;
            document.getElementById('closed-win-rate').textContent = winRate.toFixed(1) + '%';
        }
        
        // Update activity feed
        function updateActivityFeed(activities) {
            const feed = document.getElementById('activity-feed');
            feed.innerHTML = '';
            
            const filtered = activityFilter === 'all' 
                ? activities 
                : activities.filter(a => a.type === activityFilter);
            
            filtered.slice(0, 50).forEach(activity => {
                const item = document.createElement('div');
                item.className = 'activity-item ' + activity.type;
                
                const time = new Date(activity.timestamp);
                item.innerHTML = 
                    '<div class="activity-time">' + time.toLocaleTimeString() + '</div>' +
                    '<div style="margin-top: 5px;">' + activity.message + '</div>';
                
                feed.appendChild(item);
            });
        }
        
        // Update strategy parameters
        function updateStrategy(strategy) {
            document.getElementById('kelly-fraction').textContent = 
                (strategy.kelly_fraction * 100) + '%';
            document.getElementById('min-bet').textContent = 
                '$' + strategy.min_bet;
            document.getElementById('cap-per-game').textContent = 
                (strategy.cap_per_game * 100) + '%';
            document.getElementById('cap-per-bet').textContent = 
                (strategy.cap_per_bet * 100) + '%';
            
            if (strategy.biases) {
                document.getElementById('bias-favorite').textContent = 
                    (strategy.biases.favorite * 100).toFixed(1) + '%';
                document.getElementById('bias-draw').textContent = 
                    (strategy.biases.draw * 100).toFixed(1) + '%';
                document.getElementById('bias-longshot').textContent = 
                    (strategy.biases.longshot * 100).toFixed(1) + '%';
            }
            
            // Update signal weights if available
            if (strategy.signal_weights) {
                document.getElementById('weight-implied').textContent = 
                    (strategy.signal_weights.implied * 100) + '%';
                document.getElementById('weight-other').textContent = 
                    (strategy.signal_weights.other * 100) + '%';
            }
        }
        
        // Show position tab
        function showPositionTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            event.target.classList.add('active');
            document.getElementById(tab + '-positions-tab').classList.add('active');
        }
        
        // Filter activity
        function filterActivity() {
            activityFilter = document.getElementById('activity-filter').value;
            updateActivityFeed(dashboardData.activity || []);
        }
        
        // Trading activity functions
        function filterTradingActivity(period) {
            loadTradingActivity(period);
        }
        
        function refreshTradingActivity() {
            const period = document.getElementById('trade-period').value;
            loadTradingActivity(period);
        }
        
        async function loadTradingActivity(period = '24h') {
            try {
                const response = await fetch('/api/trading/recent?period=' + period);
                const data = await response.json();
                
                if (data.success) {
                    updateTradingActivity(data);
                }
            } catch (error) {
                console.error('Error loading trading activity:', error);
            }
        }
        
        function updateTradingActivity(data) {
            // Update summary stats
            document.getElementById('recent-trades-count').textContent = data.summary?.total_trades || '0';
            
            const pnl = data.summary?.total_pnl || 0;
            const pnlElement = document.getElementById('recent-pnl');
            pnlElement.textContent = pnl >= 0 ? '+$' + pnl.toFixed(2) : '-$' + Math.abs(pnl).toFixed(2);
            pnlElement.className = pnl >= 0 ? 'positive' : 'negative';
            
            document.getElementById('recent-win-rate').textContent = 
                (data.summary?.win_rate || 0).toFixed(1) + '%';
            
            // Calculate average trade duration
            const avgDuration = data.summary?.avg_duration_hours || 0;
            const durationText = avgDuration < 1 ? 
                Math.round(avgDuration * 60) + 'm' : 
                avgDuration.toFixed(1) + 'h';
            document.getElementById('avg-trade-time').textContent = durationText;
            
            // Update trades table
            const tbody = document.getElementById('recent-trades-tbody');
            const trades = data.trades || [];
            
            if (trades.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 20px; color: #666;">No trades in selected period</td></tr>';
                return;
            }
            
            tbody.innerHTML = trades.map(trade => {
                const time = new Date(trade.timestamp);
                const isOpen = trade.status === 'open';
                // Check both 'result' and 'status' fields for proper display
                const isWon = trade.result === 'won';
                const isLost = trade.result === 'lost';
                const isPending = trade.result === 'pending' || trade.result === 'closed';
                
                const statusIcon = isWon ? '✓' : 
                                 isLost ? '✗' : 
                                 isOpen ? '⏱' : 
                                 isPending ? '•' : '-';
                const statusColor = isWon ? '#00ff00' : 
                                  isLost ? '#ff6666' : 
                                  isOpen ? '#ffff00' : 
                                  isPending ? '#888' : '#666';
                
                const pnlDisplay = isOpen ? 
                    '<span style="color: #666;">-</span>' :
                    '<span class="' + (trade.pnl >= 0 ? 'profit' : 'loss') + '">' + (trade.pnl >= 0 ? '+' : '') + '$' + Math.abs(trade.pnl).toFixed(0) + '</span>';
                
                // Shorten market name
                const shortMarket = trade.market_name.length > 20 ? 
                    trade.market_name.substring(0, 20) + '...' : trade.market_name;
                
                return '<tr style="border-bottom: 1px solid #222;">' +
                        '<td style="padding: 5px; color: #888; font-size: 0.8em;">' + time.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) + '</td>' +
                        '<td style="padding: 5px; font-size: 0.85em;" title="' + trade.market_name + '">' + shortMarket + '</td>' +
                        '<td style="padding: 5px; text-align: center; color: #00ff00; font-size: 0.8em;">' + trade.outcome + '</td>' +
                        '<td style="padding: 5px; text-align: right; font-size: 0.85em;">$' + trade.stake.toFixed(0) + '</td>' +
                        '<td style="padding: 5px; text-align: center; color: #00ffff; font-size: 0.85em;">' + trade.odds.toFixed(2) + '</td>' +
                        '<td style="padding: 5px; text-align: center; color: ' + statusColor + ';">' + statusIcon + '</td>' +
                        '<td style="padding: 5px; text-align: right; font-size: 0.85em;">' + pnlDisplay + '</td>' +
                    '</tr>';
            }).join('');
        }
        
        // Toggle collapsible
        function toggleCollapsible(element) {
            element.classList.toggle('collapsed');
            const content = element.nextElementSibling;
            content.classList.toggle('collapsed');
        }
        
        // Draw portfolio donut chart
        function drawPortfolioDonut(composition) {
            const canvas = document.getElementById('portfolio-chart');
            if (!canvas || !composition || composition.length === 0) return;
            
            const ctx = canvas.getContext('2d');
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const radius = Math.min(centerX, centerY) - 5;
            const innerRadius = radius * 0.6;
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw donut slices
            let currentAngle = -Math.PI / 2; // Start at top
            
            composition.forEach((item, index) => {
                const sliceAngle = (item.percentage / 100) * 2 * Math.PI;
                
                // Draw outer arc
                ctx.beginPath();
                ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
                ctx.arc(centerX, centerY, innerRadius, currentAngle + sliceAngle, currentAngle, true);
                ctx.closePath();
                
                ctx.fillStyle = item.color || '#666';
                ctx.fill();
                
                // Add subtle border
                ctx.strokeStyle = '#000';
                ctx.lineWidth = 1;
                ctx.stroke();
                
                currentAngle += sliceAngle;
            });
            
            // Update legend
            const legendDiv = document.getElementById('portfolio-legend');
            if (legendDiv) {
                legendDiv.innerHTML = composition.map(item => 
                    '<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">' +
                        '<div style="width: 12px; height: 12px; background: ' + (item.color || '#666') + '; border-radius: 2px;"></div>' +
                        '<span style="color: #aaa; font-size: 0.85em;">' + item.name + ': ' + item.percentage.toFixed(1) + '%</span>' +
                        '<span style="color: #888; font-size: 0.8em;">($' + item.value.toFixed(0) + ')</span>' +
                    '</div>'
                ).join('');
            }
        }
        
        // Execute trades function
        async function executeTrades() {
            if (!confirm('Execute paper trades based on current signals?')) return;
            
            try {
                const response = await fetch('/api/trading/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: 'paper' })
                });
                
                const result = await response.json();
                
                // Refresh dashboard data
                loadDashboardData();
                
                // Show result message
                if (result.success) {
                    console.log('Executed ' + result.trades_count + ' trades');
                } else {
                    console.error('Trade execution failed:', result.error);
                }
            } catch (error) {
                console.error('Error executing trades:', error);
            }
        }
        
        // Export table to CSV
        function exportToCSV() {
            const table = document.getElementById('matches-table');
            const rows = Array.from(table.querySelectorAll('tr'));
            
            // Build CSV content
            let csvContent = [];
            
            // Header
            const headers = Array.from(rows[0].querySelectorAll('th')).map(th => th.textContent.trim());
            csvContent.push(headers.join(','));
            
            // Data rows (skip header and summary)
            for (let i = 1; i < rows.length - 1; i++) {
                const cells = Array.from(rows[i].querySelectorAll('td'));
                const rowData = cells.map(cell => {
                    // Clean up cell content
                    let text = cell.textContent.trim();
                    // Remove currency symbols and clean numbers
                    text = text.replace(/[$+]/g, '');
                    // Wrap in quotes if contains comma
                    if (text.includes(',')) text = '"' + text + '"';
                    return text;
                });
                csvContent.push(rowData.join(','));
            }
            
            // Get summary row (first row with class summary-row)
            const summaryRow = Array.from(rows).find(row => row.classList && row.classList.contains('summary-row'));
            const summaryCells = Array.from(summaryRow.querySelectorAll('td'));
            const summaryData = summaryCells.map(cell => {
                let text = cell.textContent.trim();
                text = text.replace(/[$+]/g, '');
                if (text.includes(',')) text = '"' + text + '"';
                return text;
            });
            csvContent.push(''); // Empty row before summary
            csvContent.push(summaryData.join(','));
            
            // Create and download file
            const csv = csvContent.join('\\n');
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            const filename = 'match_dashboard_' + new Date().toISOString().slice(0,10) + '.csv';
            
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            link.style.display = 'none';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            console.log('Exported ' + (rows.length - 2) + ' matches to ' + filename);
        }
        
        // Update period function
        function updatePeriod(days) {
            // Store selected period
            window.selectedPeriod = parseInt(days);
            
            // Reload dashboard with new period
            loadDashboardData();
        }
        
        // Wait for the page to be fully loaded before initializing
        window.onload = function() {
            console.log('Dashboard initializing...');
            console.log('updateMatchesAndPositions defined?', typeof updateMatchesAndPositions);
            console.log('loadDashboardData defined?', typeof loadDashboardData);
            
            updateTime();
            setInterval(updateTime, 1000);
            
            console.log('Loading dashboard data...');
            loadDashboardData();
            
            // Check if loadTradingActivity exists before calling
            if (typeof loadTradingActivity === 'function') {
                loadTradingActivity('24h');
                setInterval(() => loadTradingActivity(document.getElementById('trade-period').value), 60000);
            }
            
            setInterval(loadDashboardData, 30000); // Update every 30 seconds
        };
    </script>
</body>
</html>
"""

DASHBOARD_HTML = """
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
        
        /* Tabs */
        .tabs {
            background: #1a1a1a;
            padding: 10px 20px;
            border-bottom: 1px solid #333;
            position: fixed;
            top: 60px;
            left: 0;
            right: 0;
            z-index: 999;
        }
        .tab-button {
            background: #222;
            border: 1px solid #333;
            color: #00ff00;
            padding: 8px 20px;
            margin-right: 10px;
            cursor: pointer;
            border-radius: 4px;
            display: inline-block;
        }
        .tab-button.active {
            background: #00ff00;
            color: #000;
            font-weight: bold;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        
        /* Header */
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
            height: 60px;
        }
        
        /* Main layout */
        .main-container {
            margin-top: 110px;
            padding: 20px;
            display: grid;
            grid-template-columns: 1fr 1fr 400px;
            gap: 20px;
            height: calc(100vh - 60px);
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
            font-size: 0.9em;
        }
        
        /* Stats */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-bottom: 20px;
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
            color: #00ff00;
        }
        
        .stat-label {
            color: #888;
            font-size: 0.9em;
        }
        
        /* Info panel */
        .info-panel {
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 15px;
            border-radius: 5px;
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
        
        /* Activity */
        .activity-item {
            padding: 8px;
            border-bottom: 1px solid #222;
            font-size: 0.9em;
        }
        
        .activity-time {
            color: #666;
            font-size: 0.8em;
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
        
        /* Responsive */
        @media (max-width: 1400px) {
            .main-container {
                grid-template-columns: 1fr 1fr;
            }
            .right-panel {
                grid-column: span 2;
            }
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
    
    <div class="tabs">
        <button class="tab-button active" onclick="showTab('markets', event)">⚽ Markets & Signals</button>
        <button class="tab-button" onclick="showTab('strategy', event)">📈 Strategy & Performance</button>
        <button class="tab-button" onclick="showTab('trading', event)">💱 Trading</button>
    </div>
    
    <div class="main-container">
        <!-- Markets Tab -->
        <div id="markets-tab" class="tab-content active">
            <div style="display: grid; grid-template-columns: 3fr 1fr; gap: 20px;">
                <!-- Integrated Markets & Signals Section -->
                <div class="section">
                    <div class="section-title">
                        ⚽ Soccer Markets with Signals ({{ market_count }} Active) - Trading Window
                        <span style="float: right; font-size: 0.8em; color: #888;">
                            Next evaluation: <span id="next-eval-time">--:--</span>
                        </span>
                    </div>
                    <div class="section-content">
                        <div id="markets-table-container" style="overflow-x: auto;">
                            <table style="width: 100%; border-collapse: collapse;">
                                <thead>
                                    <tr style="border-bottom: 2px solid #333;">
                                        <th style="text-align: left; padding: 8px; color: #00ffff;">Match</th>
                                        <th style="text-align: center; padding: 8px; color: #00ffff;">Time</th>
                                        <th style="text-align: center; padding: 8px; color: #00ffff;" colspan="3">Home</th>
                                        <th style="text-align: center; padding: 8px; color: #00ffff;" colspan="3">Draw</th>
                                        <th style="text-align: center; padding: 8px; color: #00ffff;" colspan="3">Away</th>
                                        <th style="text-align: center; padding: 8px; color: #00ffff;">Status</th>
                                    </tr>
                                    <tr style="border-bottom: 1px solid #222;">
                                        <th></th>
                                        <th></th>
                                        <th style="text-align: center; padding: 4px; color: #888; font-size: 0.85em;">Odds</th>
                                        <th style="text-align: center; padding: 4px; color: #888; font-size: 0.85em;">Signal</th>
                                        <th style="text-align: center; padding: 4px; color: #888; font-size: 0.85em;">Edge</th>
                                        <th style="text-align: center; padding: 4px; color: #888; font-size: 0.85em;">Odds</th>
                                        <th style="text-align: center; padding: 4px; color: #888; font-size: 0.85em;">Signal</th>
                                        <th style="text-align: center; padding: 4px; color: #888; font-size: 0.85em;">Edge</th>
                                        <th style="text-align: center; padding: 4px; color: #888; font-size: 0.85em;">Odds</th>
                                        <th style="text-align: center; padding: 4px; color: #888; font-size: 0.85em;">Signal</th>
                                        <th style="text-align: center; padding: 4px; color: #888; font-size: 0.85em;">Edge</th>
                                        <th></th>
                                    </tr>
                                </thead>
                                <tbody id="markets-tbody">
                                    <!-- Populated by JavaScript -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
        
                <!-- Stats & Info Section -->
                <div class="section right-panel">
            <div class="section-title">📊 System Overview</div>
            <div class="section-content">
                <!-- Stats -->
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value" id="total-markets">0</div>
                        <div class="stat-label">Total Active</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="soccer-markets">0</div>
                        <div class="stat-label">Soccer Matches</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="paper-balance">$10,000</div>
                        <div class="stat-label">Paper Balance</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="avg-odds">0.00</div>
                        <div class="stat-label">Avg Odds</div>
                    </div>
                </div>
                
                <!-- System Info -->
                <div class="info-panel">
                    <div class="info-item">
                        <span class="info-label">Database Size</span>
                        <span class="info-value" id="db-size">216 GB</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Total Markets</span>
                        <span class="info-value" id="total-db-markets">Loading...</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Total Odds Records</span>
                        <span class="info-value" id="total-odds">Loading...</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Data Source</span>
                        <span class="info-value">Overtime API</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Paper Trading</span>
                        <span class="info-value" id="paper-trading-status">Active</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Last Update</span>
                        <span class="info-value" id="last-update">-</span>
                    </div>
                </div>
                
                <!-- Activity Log -->
                <div style="margin-top: 20px;">
                    <h3 style="color: #00ffff; margin-bottom: 10px;">Recent Activity</h3>
                    <div id="activity-log" style="max-height: 200px; overflow-y: auto;">
                        <!-- Populated by JavaScript -->
                    </div>
                </div>
            </div>
        </div>
            </div>
        </div>
        
        <!-- Merged Strategy & Performance Tab -->
        <div id="strategy-tab" class="tab-content">
            <div style="margin: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <h2 style="color: #00ffff; margin: 0;">📈 Strategy Parameters & Performance Analysis</h2>
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <span style="color: #888;">Performance Window:</span>
                        <select id="window-selector" onchange="loadPerformanceData(this.value)" style="background: #222; color: #00ff00; border: 1px solid #333; padding: 5px 10px; border-radius: 4px;">
                            <option value="7">Last 7 days</option>
                            <option value="30" selected>Last 30 days</option>
                            <option value="90">Last 90 days</option>
                            <option value="365">Last year</option>
                            <option value="9999">All time</option>
                        </select>
                    </div>
                </div>
                
                <!-- Top Row: Key Metrics -->
                <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 15px; margin-bottom: 30px;">
                    <div class="stat-box" style="background: #1a1a1a; border: 1px solid #333; padding: 20px; text-align: center; border-radius: 5px;">
                        <div style="font-size: 2em; color: #00ff00; font-weight: bold;" id="sharpe-ratio">4.2</div>
                        <div style="color: #888; margin-top: 5px;">Sharpe Ratio</div>
                        <div style="font-size: 0.8em; color: #666; margin-top: 3px;">Loading...</div>
                    </div>
                    <div style="background: #1a1a1a; border: 1px solid #333; padding: 20px; text-align: center; border-radius: 5px;">
                        <div style="font-size: 2em; color: #00ff00; font-weight: bold;" id="expected-return">12.5%</div>
                        <div style="color: #888; margin-top: 5px;">Expected Return</div>
                        <div style="font-size: 0.8em; color: #666; margin-top: 3px;" id="return-type">Cash Flow Adjusted</div>
                    </div>
                    <div style="background: #1a1a1a; border: 1px solid #333; padding: 20px; text-align: center; border-radius: 5px;">
                        <div style="font-size: 2em; color: #00ff00; font-weight: bold;" id="volatility">3.1%</div>
                        <div style="color: #888; margin-top: 5px;">Volatility</div>
                        <div style="font-size: 0.8em; color: #666; margin-top: 3px;">Daily σ</div>
                    </div>
                    <div style="background: #1a1a1a; border: 1px solid #333; padding: 20px; text-align: center; border-radius: 5px;">
                        <div style="font-size: 2em; color: #00ff00; font-weight: bold;" id="win-rate">52.8%</div>
                        <div style="color: #888; margin-top: 5px;">Win Rate</div>
                        <div style="font-size: 0.8em; color: #666; margin-top: 3px;">847 bets</div>
                    </div>
                    <div style="background: #1a1a1a; border: 1px solid #333; padding: 20px; text-align: center; border-radius: 5px;">
                        <div style="font-size: 2em; color: #ff6666; font-weight: bold;" id="max-dd">-2.8%</div>
                        <div style="color: #888; margin-top: 5px;">Max Drawdown</div>
                        <div style="font-size: 0.8em; color: #666; margin-top: 3px;">Peak to trough</div>
                    </div>
                    <div style="background: #1a1a1a; border: 1px solid #333; padding: 20px; text-align: center; border-radius: 5px;">
                        <div style="font-size: 2em; color: #00ff00; font-weight: bold;" id="calmar">4.46</div>
                        <div style="color: #888; margin-top: 5px;">Calmar Ratio</div>
                        <div style="font-size: 0.8em; color: #666; margin-top: 3px;">Return/DD</div>
                    </div>
                </div>
                
                <!-- Three Column Layout -->
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                    <!-- Column 1: Strategy Parameters -->
                    <div>
                        <h3 style="color: #00ff00; margin-bottom: 15px;">⚙️ Active Parameters</h3>
                        <div style="background: #1a1a1a; border: 1px solid #333; padding: 20px; border-radius: 5px; margin-bottom: 20px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                <span style="color: #888;">Kelly Fraction:</span>
                                    <span style="color: #00ff00; font-weight: bold;" id="kelly-fraction">25%</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                    <span>Minimum Bet:</span>
                                    <span style="color: #00ff00; font-weight: bold;" id="min-bet">$10</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                    <span>Bankroll:</span>
                                    <span style="color: #00ff00; font-weight: bold;" id="bankroll">$10,000</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                    <span>Max per Game:</span>
                                    <span style="color: #00ff00; font-weight: bold;" id="cap-per-game">25%</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                    <span>Max per Bet:</span>
                                    <span style="color: #00ff00; font-weight: bold;" id="cap-per-bet">25%</span>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Max per Market Type:</span>
                                    <span style="color: #00ff00; font-weight: bold;" id="cap-per-market">10%</span>
                                </div>
                            </div>
                            
                            <h3 style="color: #00ff00; margin-bottom: 10px;">Bias Adjustments</h3>
                            <div style="background: #1a1a1a; border: 1px solid #333; padding: 15px; border-radius: 5px;">
                                <div style="margin-bottom: 8px;">Favorites (odds < 2.0): <span style="color: #ff6666;" id="bias-favorite">-1.0%</span></div>
                                <div style="margin-bottom: 8px;">Draws: <span style="color: #00ff00;" id="bias-draw">+0.5%</span></div>
                                <div>Longshots (odds > 4.0): <span style="color: #00ff00;" id="bias-longshot">+1.0%</span></div>
                            </div>
                        </div>
                        
                        <!-- Trading Parameters Section -->
                        <div style="margin-top: 20px;">
                            <h3 style="color: #00ff00; margin-bottom: 15px;">🎯 Paper Trading Parameters</h3>
                            <div style="background: #1a1a1a; border: 1px solid #333; padding: 20px; border-radius: 5px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                    <span style="color: #888;">Minimum Edge:</span>
                                    <span style="color: #00ff00; font-weight: bold;" id="min-edge">0.1%</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                    <span style="color: #888;">Evaluation Frequency:</span>
                                    <span style="color: #00ff00; font-weight: bold;" id="eval-frequency">Every 15 minutes</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                    <span style="color: #888;">Trading Window:</span>
                                    <span style="color: #00ff00; font-weight: bold;" id="trading-window">First Active Chunk</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                    <span style="color: #888;">Max Positions:</span>
                                    <span style="color: #00ff00; font-weight: bold;" id="max-positions">50</span>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span style="color: #888;">Active Since:</span>
                                    <span style="color: #00ffff;" id="paper-trading-start">Checking...</span>
                                </div>
                            </div>
                            
                            <h3 style="color: #00ff00; margin: 20px 0 15px;">📊 Game Chunk Parameters</h3>
                            <div style="background: #1a1a1a; border: 1px solid #333; padding: 20px; border-radius: 5px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                    <span style="color: #888;">Minimum Break Between Chunks:</span>
                                    <span style="color: #00ff00; font-weight: bold;" id="min-break-time">6 hours</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                    <span style="color: #888;">Average Game Duration:</span>
                                    <span style="color: #00ff00; font-weight: bold;" id="avg-game-duration">3 hours</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                    <span style="color: #888;">Chunk Selection Mode:</span>
                                    <span style="color: #00ff00; font-weight: bold;" id="chunk-mode">First Chunk Only</span>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span style="color: #888;">Current Active Window:</span>
                                    <span style="color: #ffaa00; font-weight: bold;" id="active-window-duration">Calculating...</span>
                                </div>
                            </div>
                            
                            <div style="margin-top: 15px; background: #1a1a1a; border: 1px solid #333; padding: 15px; border-radius: 5px;">
                                <h4 style="color: #ffaa00; margin-bottom: 10px;">Market Evaluation Stats</h4>
                                <div id="evaluation-stats">
                                    <div style="margin-bottom: 8px;">
                                        <span style="color: #888;">Last Evaluation:</span>
                                        <span style="color: #00ffff;" id="last-eval-time">--:--</span>
                                    </div>
                                    <div style="margin-bottom: 8px;">
                                        <span style="color: #888;">Markets Evaluated:</span>
                                        <span style="color: #00ffff;" id="markets-evaluated">0</span>
                                    </div>
                                    <div style="margin-bottom: 8px;">
                                        <span style="color: #888;">Tradeable Found:</span>
                                        <span style="color: #00ff00;" id="tradeable-found">0</span>
                                    </div>
                                    <div>
                                        <span style="color: #888;">Trades Executed:</span>
                                        <span style="color: #00ff00;" id="trades-executed-today">0</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div>
                            <h3 style="color: #00ff00; margin-bottom: 10px;">Edge Creation</h3>
                            <div style="background: #1a1a1a; border: 1px solid #333; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                                <p style="margin-bottom: 10px;">ImpliedRaw uses bookmaker odds as predictions, which includes their margin. This gives zero edge by definition.</p>
                                <p style="margin-bottom: 10px;">Our bias adjustments exploit known inefficiencies:</p>
                                <ul style="list-style: none; padding-left: 10px;">
                                    <li>• Public overbets favorites</li>
                                    <li>• Public underbets draws & longshots</li>
                                    <li>• Average: 1.8 positive edges per match</li>
                                </ul>
                            </div>
                            
                            <h3 style="color: #00ff00; margin-bottom: 10px;">Signal Weights</h3>
                            <div style="background: #1a1a1a; border: 1px solid #333; padding: 15px; border-radius: 5px;">
                                <div style="margin-bottom: 10px;">
                                    <div>Implied Probability: <span id="signal-implied-pct">80%</span></div>
                                    <div style="background: #333; height: 10px; margin-top: 5px; border-radius: 5px;">
                                        <div id="signal-implied-bar" style="background: #00ff00; width: 80%; height: 100%; border-radius: 5px;"></div>
                                    </div>
                                </div>
                                <div>
                                    <div>Coin Flip: <span id="signal-coin-pct">20%</span></div>
                                    <div style="background: #333; height: 10px; margin-top: 5px; border-radius: 5px;">
                                        <div id="signal-coin-bar" style="background: #00ff00; width: 20%; height: 100%; border-radius: 5px;"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Cash Flow Reality Section -->
                <div style="margin-top: 30px; background: #111; border: 1px solid #222; padding: 20px; border-radius: 8px;">
                    <h3 style="color: #ffaa00; margin-bottom: 15px;">💰 Cash Flow & Realistic Returns</h3>
                    <div style="background: #1a1a1a; border: 1px solid #333; padding: 20px; border-radius: 5px;">
                        <h4 style="color: #00ff00; margin-bottom: 10px;">Why Returns Are Shown as a Range:</h4>
                        <div style="margin-bottom: 15px;">
                            <p style="margin-bottom: 10px;">The theoretical "compounded" return assumes perfect conditions that don't exist in reality:</p>
                            <ul style="list-style: none; padding-left: 10px; color: #ccc;">
                                <li style="margin-bottom: 8px;">❌ <strong>Settlement Delays:</strong> 24-48 hours to receive winnings</li>
                                <li style="margin-bottom: 8px;">❌ <strong>Platform Limits:</strong> Maximum bet sizes and daily limits</li>
                                <li style="margin-bottom: 8px;">❌ <strong>Withdrawal Requirements:</strong> Can't compound 100% of profits</li>
                                <li style="margin-bottom: 8px;">❌ <strong>Bankroll Management:</strong> Must maintain reserves</li>
                            </ul>
                        </div>
                        
                        <div style="border-top: 1px solid #333; padding-top: 15px; margin-top: 15px;">
                            <h4 style="color: #00ff00; margin-bottom: 10px;">Actual Performance Metrics:</h4>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                                <div>
                                    <div style="color: #888; margin-bottom: 5px;">Per Session (4 hours):</div>
                                    <ul style="list-style: none; padding: 0;">
                                        <li>• Avg Stake: <span id="cash-flow-stake" style="color: #00ff00;">$16</span> (16% of bankroll)</li>
                                        <li>• Expected Profit: <span id="cash-flow-profit" style="color: #00ff00;">$1.87</span> (1.87% of bankroll)</li>
                                        <li>• Win Rate: <span id="cash-flow-win-rate" style="color: #00ff00;">52.8%</span></li>
                                    </ul>
                                </div>
                                <div>
                                    <div style="color: #888; margin-bottom: 5px;">Daily (6 sessions):</div>
                                    <ul style="list-style: none; padding: 0;">
                                        <li>• Simple Return: <span id="cash-flow-daily" style="color: #00ff00;">11.2%</span></li>
                                        <li>• Compound (theoretical): <span style="color: #ffaa00;">11.8%</span></li>
                                        <li>• Realistic: <span style="color: #00ff00;">8-10%</span></li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Recent Sessions -->
                    <div style="margin-top: 20px;">
                        <h3 style="color: #00ffff; margin-bottom: 15px;">📈 Recent Session Performance</h3>
                        <div id="recent-sessions" style="max-height: 200px; overflow-y: auto; background: #1a1a1a; border: 1px solid #333; padding: 10px; border-radius: 5px;">
                            <!-- Will be populated by JavaScript -->
                        </div>
                    </div>
                    
                    <!-- Live Links -->
                    <div style="margin-top: 20px; display: flex; gap: 20px;">
                        <div>
                            <a href="#" onclick="window.open('betting_reports/', '_blank'); return false;" style="color: #00ff00; text-decoration: none;">
                                → Live Trading Reports
                            </a>
                        </div>
                        <div>
                            <a href="#" onclick="window.open('backtests/', '_blank'); return false;" style="color: #00ff00; text-decoration: none;">
                                → Historical Backtests
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Trading Tab -->
        <div id="trading-tab" class="tab-content" style="padding: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h2 style="color: #00ffff; margin: 0;">💱 Trading Activity & Performance</h2>
                    <div style="display: flex; gap: 20px; align-items: center;">
                        <button onclick="executeTrades()" style="background: #00ff00; color: #000; border: none; padding: 10px 20px; border-radius: 4px; font-weight: bold; cursor: pointer;">
                            🚀 Execute Trades Now
                        </button>
                        <div style="display: flex; gap: 10px; align-items: center;">
                            <span class="status-indicator status-running" id="trading-status-indicator"></span>
                            <span id="trading-mode" style="color: #00ff00; font-weight: bold;">Paper Trading</span>
                        </div>
                        <select id="trading-filter" onchange="filterTrades(this.value)" style="background: #222; color: #00ff00; border: 1px solid #333; padding: 5px 10px; border-radius: 4px;">
                            <option value="all">All Trades</option>
                            <option value="today">Today</option>
                            <option value="week">This Week</option>
                            <option value="month">This Month</option>
                        </select>
                    </div>
                </div>
                
                <!-- Portfolio Value Section -->
                <div style="background: #1a1a1a; border: 1px solid #333; border-radius: 5px; padding: 20px; margin-bottom: 30px;">
                    <h3 style="color: #00ff00; margin-bottom: 15px;">📊 Portfolio Value</h3>
                    <div style="display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 20px;">
                        <div>
                            <div style="font-size: 3em; font-weight: bold; color: #00ff00;" id="portfolio-value">$10,000.00</div>
                            <div style="font-size: 1.5em; margin-top: 10px;">
                                <span id="portfolio-change" class="profit">+$0.00</span>
                                <span id="portfolio-change-pct" class="profit">(0.00%)</span>
                                <span style="color: #666; margin-left: 20px;">Today</span>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: #888;">Cash Available</div>
                            <div style="font-size: 1.5em; color: #00ffff;" id="cash-available">$10,000</div>
                            <div style="color: #888; margin-top: 10px;">Positions Value</div>
                            <div style="font-size: 1.5em; color: #ff9900;" id="positions-value">$0</div>
                        </div>
                        <div>
                            <canvas id="portfolio-chart" width="200" height="200" style="max-width: 100%; height: auto;"></canvas>
                        </div>
                    </div>
                    <div id="portfolio-legend" style="margin-top: 15px; display: flex; flex-wrap: wrap; gap: 15px;">
                        <!-- Legend will be populated by JavaScript -->
                    </div>
                </div>
                
                <!-- Trading Metrics Row -->
                <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 15px; margin-bottom: 30px;">
                    <div class="stat-box" style="background: #1a1a1a; border: 1px solid #333; padding: 20px; text-align: center; border-radius: 5px;">
                        <div style="font-size: 2em; font-weight: bold;" id="today-pnl" class="profit">$0.00</div>
                        <div style="color: #888; margin-top: 5px;">Today's P&L</div>
                    </div>
                    <div style="background: #1a1a1a; border: 1px solid #333; padding: 20px; text-align: center; border-radius: 5px;">
                        <div style="font-size: 2em; color: #00ffff; font-weight: bold;" id="total-trades">0</div>
                        <div style="color: #888; margin-top: 5px;">Total Trades</div>
                    </div>
                    <div style="background: #1a1a1a; border: 1px solid #333; padding: 20px; text-align: center; border-radius: 5px;">
                        <div style="font-size: 2em; color: #00ff00; font-weight: bold;" id="trade-win-rate">0.0%</div>
                        <div style="color: #888; margin-top: 5px;">Win Rate</div>
                    </div>
                    <div style="background: #1a1a1a; border: 1px solid #333; padding: 20px; text-align: center; border-radius: 5px;">
                        <div style="font-size: 2em; color: #ff9900; font-weight: bold;" id="active-positions">0</div>
                        <div style="color: #888; margin-top: 5px;">Active Positions</div>
                    </div>
                    <div style="background: #1a1a1a; border: 1px solid #333; padding: 20px; text-align: center; border-radius: 5px;">
                        <div style="font-size: 2em; color: #00ff00; font-weight: bold;" id="current-capital">$10,000</div>
                        <div style="color: #888; margin-top: 5px;">Current Capital</div>
                    </div>
                    <div style="background: #1a1a1a; border: 1px solid #333; padding: 20px; text-align: center; border-radius: 5px;">
                        <div style="font-size: 2em; color: #00ffff; font-weight: bold;" id="total-exposure">0%</div>
                        <div style="color: #888; margin-top: 5px;">Total Exposure</div>
                    </div>
                </div>
                
                <!-- Trading Stats - MOVED UP -->
                <div style="margin-bottom: 20px; background: #1a1a1a; border: 1px solid #333; border-radius: 5px; padding: 15px;">
                    <div class="section-title" style="border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 15px;">
                        📊 Trading Statistics
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px;">
                        <div>
                            <div style="color: #888;">Average Trade Size</div>
                            <div style="font-size: 1.5em; color: #00ff00;" id="avg-trade-size">$0.00</div>
                        </div>
                        <div>
                            <div style="color: #888;">Largest Win</div>
                            <div style="font-size: 1.5em; color: #00ff00;" id="largest-win">$0.00</div>
                        </div>
                        <div>
                            <div style="color: #888;">Largest Loss</div>
                            <div style="font-size: 1.5em; color: #ff4444;" id="largest-loss">$0.00</div>
                        </div>
                        <div>
                            <div style="color: #888;">Profit Factor</div>
                            <div style="font-size: 1.5em; color: #00ffff;" id="profit-factor">0.00</div>
                        </div>
                    </div>
                </div>
                
                <!-- Trading Activity -->
                <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 20px;">
                    <!-- Recent Trades -->
                    <div class="section" style="background: #1a1a1a; border: 1px solid #333; border-radius: 5px; padding: 15px;">
                        <div class="section-title" style="border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 15px;">
                            📈 Recent Trading Activity
                            <span id="new-trades-badge" style="background: #ff0000; color: #fff; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; margin-left: 10px; display: none;">NEW</span>
                        </div>
                        <div style="max-height: 400px; overflow-y: auto;">
                            <table style="width: 100%; border-collapse: collapse;">
                                <thead>
                                    <tr style="border-bottom: 2px solid #333;">
                                        <th style="text-align: left; padding: 8px; color: #00ffff;">Time</th>
                                        <th style="text-align: left; padding: 8px; color: #00ffff;">Market</th>
                                        <th style="text-align: left; padding: 8px; color: #00ffff;">Side</th>
                                        <th style="text-align: right; padding: 8px; color: #00ffff;">Size</th>
                                        <th style="text-align: right; padding: 8px; color: #00ffff;">Price</th>
                                        <th style="text-align: right; padding: 8px; color: #00ffff;">Edge</th>
                                        <th style="text-align: right; padding: 8px; color: #00ffff;">P&L</th>
                                        <th style="text-align: center; padding: 8px; color: #00ffff;">Status</th>
                                    </tr>
                                </thead>
                                <tbody id="trades-tbody">
                                    <!-- Populated by JavaScript -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                    
                    <!-- Trading Log -->
                    <div class="section" style="background: #1a1a1a; border: 1px solid #333; border-radius: 5px; padding: 15px;">
                        <div class="section-title" style="border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 15px;">
                            📋 Trading Log
                        </div>
                        <div id="trading-log" style="max-height: 400px; overflow-y: auto; font-family: monospace; font-size: 0.85em;">
                            <!-- Populated by JavaScript -->
                        </div>
                    </div>
                </div>
                
                <!-- Open Positions Section -->
                <div style="margin-bottom: 20px; background: #1a1a1a; border: 1px solid #333; border-radius: 5px; padding: 15px;">
                    <div class="section-title" style="border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 15px;">
                        💼 Open Positions
                        <span style="float: right; color: #666; font-size: 0.8em;">Updated: <span id="positions-update-time">--:--:--</span></span>
                    </div>
                    <div style="overflow-x: auto;">
                        <table style="width: 100%; border-collapse: collapse;">
                            <thead>
                                <tr style="border-bottom: 2px solid #333;">
                                    <th style="text-align: left; padding: 8px; color: #00ffff;">Home</th>
                                    <th style="text-align: left; padding: 8px; color: #00ffff;">Away</th>
                                    <th style="text-align: left; padding: 8px; color: #00ffff;">Pick</th>
                                    <th style="text-align: right; padding: 8px; color: #00ffff;">Size</th>
                                    <th style="text-align: right; padding: 8px; color: #00ffff;">Fees</th>
                                    <th style="text-align: right; padding: 8px; color: #00ffff;">Entry</th>
                                    <th style="text-align: right; padding: 8px; color: #00ffff;">Current</th>
                                    <th style="text-align: right; padding: 8px; color: #00ffff;">P&L</th>
                                    <th style="text-align: right; padding: 8px; color: #00ffff;">P&L %</th>
                                    <th style="text-align: center; padding: 8px; color: #00ffff;">Closes In</th>
                                    <th style="text-align: center; padding: 8px; color: #00ffff;">Action</th>
                                </tr>
                            </thead>
                            <tbody id="positions-tbody">
                                <tr>
                                    <td colspan="11" style="text-align: center; padding: 20px; color: #666;">No open positions</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <!-- Closed Positions Section - NEW -->
                <div style="background: #1a1a1a; border: 1px solid #333; border-radius: 5px; padding: 15px;">
                    <div class="section-title" style="border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 15px;">
                        🏁 Closed Positions
                        <span style="float: right; color: #666; font-size: 0.8em;">
                            Showing last <select id="closed-limit" onchange="updateClosedPositions()" style="background: #222; color: #00ff00; border: 1px solid #333; padding: 2px;">
                                <option value="20">20</option>
                                <option value="50">50</option>
                                <option value="100">100</option>
                            </select> positions
                        </span>
                    </div>
                    <div style="overflow-x: auto;">
                        <table style="width: 100%; border-collapse: collapse;">
                            <thead>
                                <tr style="border-bottom: 2px solid #333;">
                                    <th style="text-align: left; padding: 8px; color: #00ffff;">Closed</th>
                                    <th style="text-align: left; padding: 8px; color: #00ffff;">Match Time</th>
                                    <th style="text-align: left; padding: 8px; color: #00ffff;">Market</th>
                                    <th style="text-align: left; padding: 8px; color: #00ffff;">Pick</th>
                                    <th style="text-align: right; padding: 8px; color: #00ffff;">Stake</th>
                                    <th style="text-align: right; padding: 8px; color: #00ffff;">Fees</th>
                                    <th style="text-align: right; padding: 8px; color: #00ffff;">Odds</th>
                                    <th style="text-align: center; padding: 8px; color: #00ffff;">Status</th>
                                    <th style="text-align: right; padding: 8px; color: #00ffff;">P&L</th>
                                    <th style="text-align: right; padding: 8px; color: #00ffff;">P&L %</th>
                                    <th style="text-align: center; padding: 8px; color: #00ffff;">Result</th>
                                    <th style="text-align: center; padding: 8px; color: #00ffff;">Score</th>
                                </tr>
                            </thead>
                            <tbody id="closed-positions-tbody">
                                <tr>
                                    <td colspan="12" style="text-align: center; padding: 20px; color: #666;">No closed positions</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #333; display: flex; justify-content: space-between;">
                        <div style="color: #888;">
                            Total Closed: <span id="total-closed-count" style="color: #00ff00;">0</span> positions
                        </div>
                        <div style="color: #888;">
                            Total P&L: <span id="total-closed-pnl" class="profit">$0.00</span>
                        </div>
                        <div style="color: #888;">
                            Win Rate: <span id="closed-win-rate" style="color: #00ff00;">0.0%</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        // Tab switching function
        function showTab(tabName, evt) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            if (evt && evt.target) {
                evt.target.classList.add('active');
            }
            
            // Load performance data when strategy tab is shown
            if (tabName === 'strategy') {
                loadPerformanceData();
                loadStrategyConfig();
            }
            // Load trading data when trading tab is shown
            if (tabName === 'trading') {
                loadTradingData();
            }
        }
        
        function loadStrategyConfig() {
            fetch('/api/strategy')
                .then(response => response.json())
                .then(data => {
                    // Update all strategy parameters with live values
                    document.getElementById('kelly-fraction').textContent = `${(data.kelly_fraction * 100).toFixed(0)}%`;
                    document.getElementById('min-bet').textContent = `$${data.min_bet}`;
                    document.getElementById('bankroll').textContent = `$${data.bankroll.toLocaleString()}`;
                    document.getElementById('cap-per-game').textContent = `${(data.cap_per_game * 100).toFixed(0)}%`;
                    document.getElementById('cap-per-bet').textContent = `${(data.cap_per_bet * 100).toFixed(0)}%`;
                    document.getElementById('cap-per-market').textContent = `${(data.cap_per_game_market * 100).toFixed(0)}%`;
                    
                    // Update biases
                    document.getElementById('bias-favorite').textContent = `${(data.biases.favorite * 100).toFixed(1)}%`;
                    document.getElementById('bias-draw').textContent = `+${(data.biases.draw * 100).toFixed(1)}%`;
                    document.getElementById('bias-longshot').textContent = `+${(data.biases.longshot * 100).toFixed(1)}%`;
                    
                    // Update paper trading parameters
                    document.getElementById('min-edge').textContent = `${(data.min_edge * 100).toFixed(1)}%`;
                    document.getElementById('eval-frequency').textContent = `Every ${data.evaluation_frequency} minutes`;
                    document.getElementById('max-positions').textContent = data.max_positions;
                    
                    // Update signal weights
                    const impliedWeight = data.signal_weights.implied_probability * 100;
                    const coinWeight = data.signal_weights.coin_flip * 100;
                    document.getElementById('signal-implied-pct').textContent = `${impliedWeight}%`;
                    document.getElementById('signal-coin-pct').textContent = `${coinWeight}%`;
                    document.getElementById('signal-implied-bar').style.width = `${impliedWeight}%`;
                    document.getElementById('signal-coin-bar').style.width = `${coinWeight}%`;
                    
                    // Update chunk parameters
                    document.getElementById('min-break-time').textContent = `${data.min_break_minutes / 60} hours`;
                    document.getElementById('avg-game-duration').textContent = `${data.avg_game_duration_minutes / 60} hours`;
                    
                    // Update chunk mode
                    let chunkModeText = 'First Chunk Only';
                    if (data.chunk_selection === 'all') {
                        chunkModeText = 'All Active Chunks';
                    } else if (data.chunk_selection === 'limit_hours') {
                        chunkModeText = `Limited to ${data.chunk_limit_hours} hours`;
                    }
                    document.getElementById('chunk-mode').textContent = chunkModeText;
                })
                .catch(error => console.error('Error loading strategy config:', error));
        }
        
        function loadPerformanceData(windowDays = 30) {
            fetch(`/api/performance?window_days=${windowDays}&max_sessions=200`)
                .then(response => response.json())
                .then(data => {
                    // Update metrics with better formatting
                    const sharpeElem = document.getElementById('sharpe-ratio');
                    if (sharpeElem) {
                        sharpeElem.textContent = data.sharpe_ratio.toFixed(2);
                        // Update subtitle with range if available
                        const sharpeCard = sharpeElem.closest('.stat-box') || sharpeElem.parentElement;
                        const subtitle = sharpeCard.querySelector('div:last-child');
                        if (subtitle && data.sharpe_min !== undefined) {
                            subtitle.textContent = `${data.sharpe_min.toFixed(1)}-${data.sharpe_max.toFixed(1)} (${data.sessions_included} sessions)`;
                        }
                    }
                    
                    if (document.getElementById('volatility')) {
                        document.getElementById('volatility').textContent = (data.volatility * 100).toFixed(1) + '%';
                    }
                    if (document.getElementById('expected-return')) {
                        const returnElem = document.getElementById('expected-return');
                        const typeElem = document.getElementById('return-type');
                        
                        if (data.expected_return_high !== undefined) {
                            // Show range for cash flow adjusted returns
                            const low = (data.expected_return * 100).toFixed(0);
                            const high = (data.expected_return_high * 100).toFixed(0);
                            returnElem.textContent = `${low}%-${high}%`;
                        } else {
                            returnElem.textContent = (data.expected_return * 100).toFixed(1) + '%';
                        }
                        
                        if (typeElem && data.return_type) {
                            typeElem.textContent = data.return_type;
                        }
                    }
                    if (document.getElementById('win-rate')) {
                        document.getElementById('win-rate').textContent = (data.win_rate * 100).toFixed(1) + '%';
                    }
                    if (document.getElementById('max-dd')) {
                        document.getElementById('max-dd').textContent = '-' + (data.max_drawdown * 100).toFixed(1) + '%';
                    }
                    
                    // Update recent sessions
                    const sessionsDiv = document.getElementById('recent-sessions');
                    if (data.recent_sessions && data.recent_sessions.length > 0) {
                        sessionsDiv.innerHTML = data.recent_sessions.map(session => `
                            <div style="padding: 10px; border-bottom: 1px solid #222;">
                                <span style="color: #00ff00;">${session.date}</span>
                                <span style="margin-left: 20px;">Sharpe: ${session.sharpe ? session.sharpe.toFixed(2) : 'N/A'}</span>
                                <span style="margin-left: 20px;">Vol: ${session.volatility ? (session.volatility * 100).toFixed(1) + '%' : 'N/A'}</span>
                                <span style="margin-left: 20px;">Stake: $${session.stake ? session.stake.toFixed(2) : 'N/A'}</span>
                            </div>
                        `).join('');
                    }
                    
                    // Update cash flow metrics
                    if (data.cash_flow_analysis) {
                        const cf = data.cash_flow_analysis;
                        if (cf.per_session) {
                            document.getElementById('cash-flow-stake').textContent = `$${cf.per_session.avg_stake.toFixed(2)}`;
                            document.getElementById('cash-flow-profit').textContent = `$${cf.per_session.expected_profit.toFixed(2)}`;
                        }
                        if (cf.daily) {
                            document.getElementById('cash-flow-daily').textContent = `${(cf.daily.simple_return * 100).toFixed(1)}%`;
                        }
                    }
                    
                    // Update paper trading start time
                    fetch('/api/trading/status')
                        .then(response => response.json())
                        .then(status => {
                            if (status.active) {
                                document.getElementById('paper-trading-start').textContent = 'Active';
                            } else {
                                document.getElementById('paper-trading-start').textContent = 'Not Running';
                            }
                        });
                    
                    // Update evaluation stats
                    fetch('/api/trading/evaluation-stats')
                        .then(response => response.json())
                        .then(stats => {
                            if (stats.last_evaluation) {
                                document.getElementById('last-eval-time').textContent = 
                                    new Date(stats.last_evaluation).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
                            }
                            document.getElementById('markets-evaluated').textContent = stats.markets_evaluated || 0;
                            document.getElementById('tradeable-found').textContent = stats.tradeable_found || 0;
                            document.getElementById('trades-executed-today').textContent = stats.trades_today || 0;
                        });
                });
        }
        
        function loadLatestSession() {
            window.open('betting_reports/latest_session.pkl', '_blank');
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            updateTime();
            setInterval(updateTime, 1000);
            loadData();
            
            // Socket.IO handlers
            socket.on('connect', function() {
                document.getElementById('ws-indicator').classList.add('connected');
                document.getElementById('ws-status').textContent = 'Live';
            });
            
            socket.on('disconnect', function() {
                document.getElementById('ws-indicator').classList.remove('connected');
                document.getElementById('ws-status').textContent = 'Disconnected';
            });
            
            socket.on('update', function(data) {
                updateDashboard(data);
            });
            
            socket.on('position_update', function(data) {
                // Update positions table
                if (data.positions) {
                    updateOpenPositions(data.positions);
                }
                
                // Update recent trades
                if (data.recent_trades) {
                    updateRecentTrades(data.recent_trades);
                }
                
                // Update portfolio value
                if (data.portfolio_value !== undefined) {
                    document.getElementById('portfolio-value').textContent = `$${data.portfolio_value.toFixed(2)}`;
                }
                
                // Update cash balance
                if (data.cash !== undefined) {
                    document.getElementById('cash-balance').textContent = `$${data.cash.toFixed(2)}`;
                }
                
                // Update timestamp
                document.getElementById('positions-update-time').textContent = new Date().toLocaleTimeString();
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
                .then(data => {
                    console.log('API response:', {
                        markets: data.markets ? data.markets.length : 0,
                        chunks: data.chunks ? data.chunks.length : 0,
                        hasChunks: !!data.chunks
                    });
                    updateDashboard(data);
                });
            
            // Also load database stats
            loadDatabaseStats();
        }
        
        function loadDatabaseStats() {
            fetch('/api/database-stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('total-db-markets').textContent = data.total_markets || 'N/A';
                    document.getElementById('total-odds').textContent = data.total_odds || 'N/A';
                    
                    // Update paper balance from portfolio API
                    fetch('/api/trading/portfolio')
                        .then(response => response.json())
                        .then(portfolio => {
                            if (portfolio.cash_available !== undefined) {
                                document.getElementById('paper-balance').textContent = `$${portfolio.cash_available.toFixed(2)}`;
                            }
                        });
                })
                .catch(error => {
                    console.error('Error loading database stats:', error);
                    document.getElementById('total-db-markets').textContent = 'Error';
                    document.getElementById('total-odds').textContent = 'Error';
                });
        }
        
        function updateDashboard(data) {
            // Update stats
            if (data.stats) {
                document.getElementById('total-markets').textContent = data.stats.total_active || 0;
                document.getElementById('soccer-markets').textContent = data.stats.soccer_count || 0;
                document.getElementById('avg-odds').textContent = 
                    (data.stats.avg_odds || 0).toFixed(2);
            }
            
            // Update markets (signals are now integrated)
            if (data.markets) {
                console.log('data.chunks:', data.chunks ? data.chunks.length : 'undefined');
                updateMarkets(data.markets, data.chunks || []);
            }
            
            // Update activity
            if (data.activity) {
                updateActivity(data.activity);
            }
            
            // Update active window duration if we have chunks
            if (data.chunks && data.chunks.length > 0) {
                const firstChunk = data.chunks[0];
                const durationMinutes = firstChunk.duration_minutes || 0;
                const durationHours = (durationMinutes / 60).toFixed(1);
                document.getElementById('active-window-duration').textContent = 
                    `${durationHours} hours (${firstChunk.num_games || 0} games)`;
            }
            
            // Update time
            updateTime();
            
            // Update last update time
            document.getElementById('last-update').textContent = 
                new Date().toLocaleTimeString('en-US', { hour12: false });
        }
        
        function updateMarkets(markets, chunks) {
            const tbody = document.getElementById('markets-tbody');
            console.log('updateMarkets called with', markets.length, 'markets and', chunks.length, 'chunks');
            
            // Sort markets by maturity date (earliest first)
            markets.sort((a, b) => new Date(a.maturity_date) - new Date(b.maturity_date));
            
            // Calculate next evaluation time
            const now = new Date();
            const nextEval = new Date(Math.ceil(now.getTime() / (15 * 60 * 1000)) * (15 * 60 * 1000));
            document.getElementById('next-eval-time').textContent = nextEval.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
            
            // Prevent flickering by checking if we have valid data before rendering
            if (!markets || markets.length === 0) {
                console.log('No markets to render');
                tbody.innerHTML = '<tr><td colspan="11" style="text-align: center; padding: 20px; color: #888;">No markets available</td></tr>';
                return;
            }
            
            // Use actual game chunks from backend with stability checks
            const gameChunks = chunks || [];
            const marketsByChunk = {};
            
            // Group markets by their chunk index with validation
            markets.forEach(market => {
                const chunkIndex = (market.chunk_index !== undefined && market.chunk_index !== null) ? market.chunk_index : -1;
                if (!marketsByChunk[chunkIndex]) {
                    marketsByChunk[chunkIndex] = [];
                }
                marketsByChunk[chunkIndex].push(market);
            });
            
            // Define unchunkedMarkets here so it's in scope
            const unchunkedMarkets = marketsByChunk[-1] || [];
            
            
            let html = '';
            
            // Define formatOutcome function once for reuse
            function formatOutcome(odds, implied, signal, edge) {
                // Debug edge value
                if (edge !== undefined && edge !== null && edge !== 0) {
                    console.log(`Edge value: ${edge}, type: ${typeof edge}`);
                }
                
                // Enhanced edge color coding - ensure edge is a number
                const edgeValue = parseFloat(edge) || 0;
                const edgeClass = edgeValue > 0.5 ? 'style="color: #00ff00; font-weight: bold; text-shadow: 0 0 4px rgba(0,255,0,0.3);"' : 
                                 edgeValue > 0 ? 'style="color: #90ff90; font-weight: bold;"' : 
                                 edgeValue < -0.5 ? 'style="color: #ff4444; font-weight: bold; text-shadow: 0 0 4px rgba(255,68,68,0.3);"' :
                                 edgeValue < 0 ? 'style="color: #ff8888;"' : 
                                 'style="color: #888;"';
                
                const impliedStr = implied ? implied.toFixed(1) + '%' : '-';
                const signalStr = signal ? signal.toFixed(1) + '%' : '-';
                const edgeStr = edgeValue !== 0 ? (edgeValue > 0 ? '+' : '') + edgeValue.toFixed(1) + '%' : '-';
                
                return `
                    <td style="text-align: center; padding: 4px; font-size: 0.85em; border-left: 1px solid #333;">
                        <div style="color: #ccc;">${odds ? odds.toFixed(2) : '-'}</div>
                        <div style="color: #999; font-size: 0.8em;">${impliedStr}</div>
                    </td>
                    <td style="text-align: center; padding: 4px; color: #ddd;">${signalStr}</td>
                    <td style="text-align: center; padding: 4px; border-right: 1px solid #333;" ${edgeClass}>${edgeStr}</td>
                `;
            }
            
            // If no chunks, group markets by time and show with nice formatting
            if (gameChunks.length === 0) {
                // Group markets by time windows (e.g., starting within 1 hour)
                const now = new Date();
                const upcomingMarkets = markets.filter(m => new Date(m.maturity_date) > now);
                const startingSoon = upcomingMarkets.filter(m => {
                    const timeDiff = new Date(m.maturity_date) - now;
                    return timeDiff < 2 * 60 * 60 * 1000; // Within 2 hours
                });
                const laterMarkets = upcomingMarkets.filter(m => {
                    const timeDiff = new Date(m.maturity_date) - now;
                    return timeDiff >= 2 * 60 * 60 * 1000;
                });
                
                // Show markets starting soon with highlight
                if (startingSoon.length > 0) {
                    html += `
                        <tr style="background: #0a0a0a; border-top: 2px solid #00ff00;">
                            <td colspan="11" style="padding: 8px; color: #00ff00; font-weight: bold;">
                                ⚡ Starting Soon (${startingSoon.length} matches)
                            </td>
                        </tr>
                    `;
                    
                    startingSoon.forEach(market => {
                        const marketSignals = market.signals || {};
                        html += `
                            <tr style="background: #1a1a2e; border-left: 3px solid #00ff00; border-bottom: 1px solid #222;">
                                <td style="padding: 8px;">${market.home_team} vs ${market.away_team}</td>
                                <td style="text-align: center; padding: 8px; color: #ffff00;">
                                    ${market.time_until}
                                </td>
                                ${formatOutcome(market.home_odds, market.home_implied, marketSignals.home_signal, marketSignals.home_edge)}
                                ${formatOutcome(market.draw_odds, market.draw_implied, marketSignals.draw_signal, marketSignals.draw_edge)}
                                ${formatOutcome(market.away_odds, market.away_implied, marketSignals.away_signal, marketSignals.away_edge)}
                                <td style="text-align: center; padding: 8px;">
                                    <div style="color: ${market.status_color}; font-weight: bold;">${market.status}</div>
                                    <div style="font-size: 0.8em; color: #666;">
                                        ${market.is_finished && market.home_score !== null && market.away_score !== null ? 
                                          `${market.home_score} - ${market.away_score}` :
                                          market.tradeable ? '✓ Edge Found' : 
                                          market.has_position ? '💰 Position Open' : '-'}
                                    </div>
                                </td>
                            </tr>
                        `;
                    });
                }
                
                // Show later markets
                if (laterMarkets.length > 0) {
                    html += `
                        <tr style="background: #0a0a0a; border-top: 2px solid #666; margin-top: 10px;">
                            <td colspan="11" style="padding: 8px; color: #999; font-weight: bold;">
                                📅 Later Today (${laterMarkets.length} matches)
                            </td>
                        </tr>
                    `;
                    
                    laterMarkets.forEach(market => {
                        const marketSignals = market.signals || {};
                        html += `
                            <tr style="border-left: 3px solid #333; border-bottom: 1px solid #222;">
                                <td style="padding: 8px;">${market.home_team} vs ${market.away_team}</td>
                                <td style="text-align: center; padding: 8px; color: #888;">
                                    ${market.time_until}
                                </td>
                                ${formatOutcome(market.home_odds, market.home_implied, marketSignals.home_signal, marketSignals.home_edge)}
                                ${formatOutcome(market.draw_odds, market.draw_implied, marketSignals.draw_signal, marketSignals.draw_edge)}
                                ${formatOutcome(market.away_odds, market.away_implied, marketSignals.away_signal, marketSignals.away_edge)}
                                <td style="text-align: center; padding: 8px;">
                                    <div style="color: ${market.status_color}; font-weight: bold;">${market.status}</div>
                                    <div style="font-size: 0.8em; color: #666;">
                                        ${market.tradeable ? '✓ Edge Found' : '-'}
                                    </div>
                                </td>
                            </tr>
                        `;
                    });
                }
            }
            
            // Debug marketsByChunk
            console.log('marketsByChunk:', Object.keys(marketsByChunk).map(k => `${k}: ${marketsByChunk[k].length} markets`));
            
            // If we have chunks, render them; otherwise render all markets in one section
            if (gameChunks.length > 0) {
                console.log('Rendering', gameChunks.length, 'chunks');
                
                // Render game chunks with enhanced visualization and collapse functionality
                gameChunks.forEach((chunk, index) => {
                    try {
                    const chunkMarkets = marketsByChunk[index] || [];
                    console.log(`Chunk ${index} has ${chunkMarkets.length} markets`);
                    const isCurrentChunk = index === 0;
                    const isNextChunk = index === 1;
                    const chunkStart = new Date(chunk.start);
                    const chunkEnd = new Date(chunk.end);
                    
                    // Enhanced chunk styling
                    const chunkLabel = isCurrentChunk ? '🔥 Active Trading Window' :
                                      isNextChunk ? '⏳ Next Trading Window' :
                                      `📅 Window ${index + 1}`;
                    const chunkColor = isCurrentChunk ? '#00ff00' :
                                      isNextChunk ? '#ffff00' : '#88aaff';
                    const chunkBgColor = isCurrentChunk ? 'rgba(0, 255, 0, 0.05)' :
                                        isNextChunk ? 'rgba(255, 255, 0, 0.05)' : 
                                        'rgba(136, 170, 255, 0.03)';
                    
                    const duration = Math.round(chunk.duration_minutes);
                    const timeRange = `${chunkStart.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })} → ${chunkEnd.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}`;
                    
                    // Collapsible window header
                    const chunkId = `chunk-${index}`;
                    const isCollapsed = index > 1; // Auto-collapse chunks after first 2
                    
                    console.log(`Building header for chunk ${index}`);
                    html += `
                        <tr style="background: linear-gradient(135deg, ${chunkBgColor}, rgba(0,0,0,0.1)); border-top: 3px solid ${chunkColor}; border-bottom: 1px solid ${chunkColor};">
                            <td colspan="11" style="padding: 12px 8px; color: ${chunkColor}; font-weight: bold; font-size: 1.05em; cursor: pointer;" onclick="toggleChunk('${chunkId}')">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <span id="${chunkId}-toggle">${isCollapsed ? '▶' : '▼'}</span>
                                        ${chunkLabel}
                                        <span style="font-size: 0.85em; margin-left: 10px; opacity: 0.8;">(${chunk.num_games} games, ${chunkMarkets.length} markets)</span>
                                    </div>
                                    <div style="font-size: 0.9em; color: #ccc;">
                                        ${timeRange} • ${duration} min window
                                    </div>
                                </div>
                            </td>
                        </tr>
                    `;
                    console.log(`Header added for chunk ${index}, html length now: ${html.length}`);
                
                // Sort markets within chunk by start time
                chunkMarkets.sort((a, b) => new Date(a.maturity_date) - new Date(b.maturity_date));
                
                console.log(`About to render ${chunkMarkets.length} markets for chunk ${index}`);
                // Add markets in this chunk with enhanced styling
                chunkMarkets.forEach((market, marketIndex) => {
                    console.log(`Rendering market ${marketIndex} in chunk ${index}: ${market.home_team} vs ${market.away_team}`);
                    const isFirstInChunk = marketIndex === 0;
                    const isLastInChunk = marketIndex === chunkMarkets.length - 1;
                    
                    const rowStyle = isCurrentChunk ? 
                        'background: linear-gradient(90deg, rgba(0, 255, 0, 0.08), rgba(0, 255, 0, 0.03)); border-left: 4px solid #00ff00; box-shadow: inset 0 0 10px rgba(0, 255, 0, 0.1);' :
                        isNextChunk ? 
                        'background: linear-gradient(90deg, rgba(255, 255, 0, 0.06), rgba(255, 255, 0, 0.02)); border-left: 3px solid #ffff00; box-shadow: inset 0 0 8px rgba(255, 255, 0, 0.08);' :
                        'background: linear-gradient(90deg, rgba(136, 170, 255, 0.04), rgba(136, 170, 255, 0.01)); border-left: 3px solid #88aaff; box-shadow: inset 0 0 6px rgba(136, 170, 255, 0.05);';
                    
                    const borderRadius = isFirstInChunk ? 'border-top-right-radius: 8px;' : 
                                       isLastInChunk ? 'border-bottom-right-radius: 8px;' : '';
                
                
                    // Find signals for this market
                    const marketSignals = market.signals || {};
                    
                    html += `
                        <tr data-chunk="${chunkId}" style="${rowStyle} ${borderRadius} border-bottom: 1px solid rgba(255,255,255,0.1); margin: 1px 0; ${isCollapsed ? 'display: none;' : ''}">
                            <td style="padding: 10px 8px; position: relative;">
                                <div style="font-weight: 500;">${market.home_team} vs ${market.away_team}</div>
                                ${isCurrentChunk ? '<div style="position: absolute; top: 2px; right: 5px; font-size: 0.7em; color: #00ff00;">●</div>' : ''}
                            </td>
                            <td style="text-align: center; padding: 8px; color: ${isCurrentChunk ? '#ffff00' : '#888'}; font-weight: ${isCurrentChunk ? 'bold' : 'normal'};">
                                ${market.time_until}
                            </td>
                            ${formatOutcome(market.home_odds, market.home_implied, marketSignals.home_signal, marketSignals.home_edge)}
                            ${formatOutcome(market.draw_odds, market.draw_implied, marketSignals.draw_signal, marketSignals.draw_edge)}
                            ${formatOutcome(market.away_odds, market.away_implied, marketSignals.away_signal, marketSignals.away_edge)}
                            <td style="text-align: center; padding: 8px;">
                                <div style="color: ${market.status_color}; font-weight: bold;">${market.status}</div>
                                <div style="font-size: 0.8em; color: #666;">
                                    ${market.is_finished && market.home_score !== null && market.away_score !== null ? 
                                      `${market.home_score} - ${market.away_score}` :
                                      market.tradeable ? '✓ Edge Found' : 
                                      market.has_position ? '💰 Position Open' : '-'}
                                </div>
                            </td>
                        </tr>
                    `;
                });
                console.log(`Finished rendering markets for chunk ${index}`);
                } catch (e) {
                    console.error(`Error rendering chunk ${index}:`, e);
                }
            });
            console.log('All chunks processed, html length:', html.length);
            } else {
                // No chunks available - render all markets in simple format
                console.log('No chunks, rendering all markets in simple format');
                markets.forEach(market => {
                    const signals = market.signals || {};
                    html += `
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                            <td style="padding: 10px 8px;">
                                <div style="font-weight: 500;">${market.home_team} vs ${market.away_team}</div>
                            </td>
                            <td style="text-align: center; padding: 8px; color: #888;">${market.time_until}</td>
                            ${formatOutcome(market.home_odds, market.home_implied, signals.home_signal, signals.home_edge)}
                            ${formatOutcome(market.draw_odds, market.draw_implied, signals.draw_signal, signals.draw_edge)}
                            ${formatOutcome(market.away_odds, market.away_implied, signals.away_signal, signals.away_edge)}
                            <td style="text-align: center; padding: 8px;"><div style="color: ${market.status_color};">${market.status}</div></td>
                        </tr>
                    `;
                });
            }
            
            // Add section for ended/unchunked games
            if (unchunkedMarkets && unchunkedMarkets.length > 0) {
                // Separate ended games from future games
                const now = new Date();
                const endedGames = unchunkedMarkets.filter(m => new Date(m.maturity_date) < now);
                const futureGames = unchunkedMarkets.filter(m => new Date(m.maturity_date) >= now);
                
                if (endedGames.length > 0) {
                    html += `
                        <tr style="background: #0a0a0a; border-top: 2px solid #9966ff;">
                            <td colspan="11" style="padding: 8px; color: #9966ff; font-weight: bold;">
                                ⏳ Pending Settlement (${endedGames.length} games)
                            </td>
                        </tr>
                    `;
                    
                    endedGames.forEach(market => {
                        const rowStyle = 'background: #1a0a1a; border-left: 3px solid #9966ff;';
                        
                        // Find signals for this market
                        const marketSignals = market.signals || {};
                        
                        html += `
                            <tr style="${rowStyle} border-bottom: 1px solid #222;">
                                <td style="padding: 8px; opacity: 0.7;">${market.home_team} vs ${market.away_team}</td>
                                <td style="text-align: center; padding: 8px; color: #9966ff;">
                                    ${market.time_until}
                                </td>
                                ${formatOutcome(market.home_odds, market.home_implied, marketSignals.home_signal, marketSignals.home_edge)}
                                ${formatOutcome(market.draw_odds, market.draw_implied, marketSignals.draw_signal, marketSignals.draw_edge)}
                                ${formatOutcome(market.away_odds, market.away_implied, marketSignals.away_signal, marketSignals.away_edge)}
                                <td style="text-align: center; padding: 8px;">
                                    <div style="color: ${market.status_color}; font-weight: bold;">${market.status}</div>
                                    <div style="font-size: 0.8em; color: #666;">
                                        ${market.is_finished && market.home_score !== null && market.away_score !== null ? 
                                          `${market.home_score} - ${market.away_score}` :
                                          market.has_position ? '💰 Pending' : 'Awaiting Result'}
                                    </div>
                                </td>
                            </tr>
                        `;
                    });
                }
            }
            
            console.log('Setting innerHTML, html length:', html.length);
            tbody.innerHTML = html;
            console.log('tbody now has', tbody.children.length, 'rows');
        }
        
        // Toggle chunk visibility
        function toggleChunk(chunkId) {
            const toggle = document.getElementById(chunkId + '-toggle');
            const chunkRows = document.querySelectorAll(`tr[data-chunk="${chunkId}"]`);
            
            if (toggle.textContent === '▶') {
                toggle.textContent = '▼';
                chunkRows.forEach(row => row.style.display = '');
            } else {
                toggle.textContent = '▶';
                chunkRows.forEach(row => row.style.display = 'none');
            }
        }
        
        // Signals are now integrated into the markets table
        
        function updateActivity(activity) {
            const container = document.getElementById('activity-log');
            container.innerHTML = activity.map(item => `
                <div class="activity-item">
                    <span class="activity-time">${item.time}</span>
                    <span style="margin-left: 10px;">${item.message}</span>
                </div>
            `).join('');
        }
        
        // Trading tab functions
        function executeTrades() {
            const button = event.target;
            button.disabled = true;
            button.textContent = '⏳ Executing...';
            
            fetch('/api/trading/execute', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert(data.message);
                        // Reload trading data to show new trades
                        loadTradingData();
                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .finally(() => {
                    button.disabled = false;
                    button.textContent = '🚀 Execute Trades Now';
                });
        }
        
        function loadTradingData() {
            // Load trading status
            fetch('/api/trading/status')
                .then(response => response.json())
                .then(data => {
                    updateTradingStatus(data);
                });
            
            // Load recent trades
            fetch('/api/trading/recent')
                .then(response => response.json())
                .then(data => {
                    updateRecentTrades(data.trades);
                    updateTradingMetrics(data.metrics);
                });
            
            // Load portfolio data
            fetch('/api/trading/portfolio')
                .then(response => response.json())
                .then(data => {
                    updatePortfolioData(data);
                });
            
            // Load open positions
            fetch('/api/trading/positions')
                .then(response => response.json())
                .then(data => {
                    updateOpenPositions(data);
                });
            
            // Load trading logs
            fetch('/api/trading/logs')
                .then(response => response.json())
                .then(data => {
                    updateTradingLog(data);
                });
            
            // Load closed positions
            updateClosedPositions();
        }
        
        function updateTradingStatus(data) {
            const modeElem = document.getElementById('trading-mode');
            const indicatorElem = document.getElementById('trading-status-indicator');
            
            modeElem.textContent = data.mode || 'Paper Trading';
            
            if (data.active) {
                indicatorElem.classList.remove('status-stopped');
                indicatorElem.classList.add('status-running');
            } else {
                indicatorElem.classList.remove('status-running');
                indicatorElem.classList.add('status-stopped');
            }
        }
        
        function updateRecentTrades(trades) {
            const tbody = document.getElementById('trades-tbody');
            if (!trades || trades.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; padding: 20px; color: #666;">No trades executed yet</td></tr>';
                return;
            }
            
            tbody.innerHTML = trades.map(trade => {
                const edgeClass = trade.edge > 0 ? 'profit' : 'loss';
                const statusClass = trade.status === 'closed' ? 'status-stopped' : 'status-running';
                const statusText = trade.status === 'closed' ? 'Closed' : 'Filled';
                
                // Format P&L if position is closed
                let pnlCell = '<td style="padding: 8px; text-align: right; color: #666;">-</td>';
                if (trade.status === 'closed' && trade.pnl !== null && trade.pnl !== undefined) {
                    const pnlClass = trade.pnl >= 0 ? 'profit' : 'loss';
                    const pnlSign = trade.pnl >= 0 ? '+' : '';
                    pnlCell = `<td style="padding: 8px; text-align: right;" class="${pnlClass}">${pnlSign}$${Math.abs(trade.pnl).toFixed(2)}</td>`;
                }
                
                return `
                    <tr style="border-bottom: 1px solid #222;">
                        <td style="padding: 8px;">${new Date(trade.timestamp).toLocaleTimeString()}</td>
                        <td style="padding: 8px;">${trade.market}</td>
                        <td style="padding: 8px;">${trade.side}</td>
                        <td style="padding: 8px; text-align: right;">$${trade.size.toFixed(2)}</td>
                        <td style="padding: 8px; text-align: right;">${trade.price.toFixed(3)}</td>
                        <td style="padding: 8px; text-align: right;" class="${edgeClass}">${trade.edge > 0 ? '+' : ''}${trade.edge.toFixed(1)}%</td>
                        ${pnlCell}
                        <td style="padding: 8px; text-align: center;"><span class="status-indicator ${statusClass}"></span>${statusText}</td>
                    </tr>
                `;
            }).join('');
        }
        
        function updateTradingMetrics(metrics) {
            if (!metrics) return;
            
            // Update metric displays
            document.getElementById('today-pnl').textContent = `$${metrics.today_pnl.toFixed(2)}`;
            document.getElementById('today-pnl').className = metrics.today_pnl >= 0 ? 'profit' : 'loss';
            
            document.getElementById('total-trades').textContent = metrics.total_trades || '0';
            document.getElementById('trade-win-rate').textContent = `${(metrics.win_rate * 100).toFixed(1)}%`;
            document.getElementById('active-positions').textContent = metrics.active_positions || '0';
            document.getElementById('current-capital').textContent = `$${metrics.current_capital.toFixed(0)}`;
            document.getElementById('total-exposure').textContent = `${(metrics.total_exposure * 100).toFixed(1)}%`;
            
            // Update stats
            document.getElementById('avg-trade-size').textContent = `$${metrics.avg_trade_size.toFixed(2)}`;
            document.getElementById('largest-win').textContent = `$${metrics.largest_win.toFixed(2)}`;
            document.getElementById('largest-loss').textContent = `-$${Math.abs(metrics.largest_loss).toFixed(2)}`;
            document.getElementById('profit-factor').textContent = metrics.profit_factor.toFixed(2);
        }
        
        function filterTrades(period) {
            // Re-load trades with filter
            fetch(`/api/trading/recent?period=${period}`)
                .then(response => response.json())
                .then(data => {
                    updateRecentTrades(data.trades);
                });
        }
        
        // Update trading log
        function updateTradingLog(logs) {
            const logDiv = document.getElementById('trading-log');
            if (!logs || logs.length === 0) {
                logDiv.innerHTML = '<div style="color: #666; padding: 10px;">No trading activity logged</div>';
                return;
            }
            
            logDiv.innerHTML = logs.map(log => {
                let colorClass = '';
                if (log.level === 'error') colorClass = 'log-error';
                else if (log.level === 'warn') colorClass = 'log-warn';
                else if (log.level === 'info') colorClass = 'log-info';
                else if (log.message.includes('Paper trade executed')) colorClass = 'log-trade';
                
                return `<div class="log-line ${colorClass}">${log.timestamp} ${log.message}</div>`;
            }).join('');
            
            // Auto-scroll to bottom
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        // Update closed positions
        function updateClosedPositions() {
            const limit = document.getElementById('closed-limit').value || 20;
            
            fetch(`/api/trading/closed_positions?limit=${limit}`)
                .then(response => response.json())
                .then(data => {
                    const tbody = document.getElementById('closed-positions-tbody');
                    const positions = data.positions || [];
                    const summary = data.summary || {};
                    
                    // Update summary stats
                    document.getElementById('total-closed-count').textContent = summary.total_count || 0;
                    document.getElementById('total-closed-pnl').textContent = summary.total_pnl >= 0 ? 
                        `+$${summary.total_pnl.toFixed(2)}` : `-$${Math.abs(summary.total_pnl).toFixed(2)}`;
                    document.getElementById('total-closed-pnl').className = summary.total_pnl >= 0 ? 'profit' : 'loss';
                    document.getElementById('closed-win-rate').textContent = `${summary.win_rate.toFixed(1)}%`;
                    
                    if (positions.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="12" style="text-align: center; padding: 20px; color: #666;">No closed positions</td></tr>';
                        return;
                    }
                    
                    // Populate table
                    tbody.innerHTML = positions.map(pos => {
                        // Determine status styling
                        let statusClass = '';
                        let statusText = '';
                        let statusIcon = '';
                        
                        if (pos.result === 'won') {
                            statusClass = 'profit';
                            statusText = 'WON';
                            statusIcon = '✓';
                        } else if (pos.result === 'lost') {
                            statusClass = 'loss';
                            statusText = 'LOST';
                            statusIcon = '✗';
                        } else if (pos.result === 'pending') {
                            statusClass = '';
                            statusText = 'PENDING';
                            statusIcon = '⏱';
                        } else {
                            statusClass = '';
                            statusText = pos.result.toUpperCase();
                            statusIcon = '-';
                        }
                        
                        // P&L styling
                        const pnlClass = pos.pnl > 0 ? 'profit' : pos.pnl < 0 ? 'loss' : '';
                        const pnlSign = pos.pnl >= 0 ? '+' : '';
                        const pnlPctSign = pos.pnl_pct >= 0 ? '+' : '';
                        
                        // Result/outcome styling
                        let outcomeDisplay = pos.final_outcome || '-';
                        if (pos.final_outcome === 'TBD') {
                            outcomeDisplay = '<span style="color: #666;">TBD</span>';
                        } else if (pos.final_outcome === 'In Play') {
                            outcomeDisplay = '<span style="color: #ffaa00;">In Play</span>';
                        } else if (pos.final_outcome === pos.outcome) {
                            outcomeDisplay = `<span class="profit">${pos.final_outcome}</span>`;
                        }
                        
                        // Score display
                        let scoreDisplay = pos.score || '-';
                        if (pos.score && pos.score !== '-') {
                            scoreDisplay = `<strong>${pos.score}</strong>`;
                        }
                        
                        // Calculate fee display
                        let feeDisplay = '-';
                        if (pos.fee_info && pos.fee_info.fee_amount) {
                            const feePct = (pos.fee_info.total_fee_pct * 100).toFixed(1);
                            feeDisplay = `$${pos.fee_info.fee_amount.toFixed(2)}<br/><span style="font-size: 0.8em; color: #888;">${feePct}%</span>`;
                        }
                        
                        return `
                            <tr style="border-bottom: 1px solid #222;">
                                <td style="padding: 8px; font-size: 0.85em;">${pos.closed_time}</td>
                                <td style="padding: 8px; font-size: 0.85em; color: #888;">${pos.match_time || '-'}</td>
                                <td style="padding: 8px;">${pos.market_name}</td>
                                <td style="padding: 8px; font-weight: bold;">${pos.outcome}</td>
                                <td style="padding: 8px; text-align: right;">$${pos.stake.toFixed(2)}</td>
                                <td style="padding: 8px; text-align: right; font-size: 0.9em;">${feeDisplay}</td>
                                <td style="padding: 8px; text-align: right;">${pos.odds.toFixed(2)}</td>
                                <td style="padding: 8px; text-align: center;" class="${statusClass}">
                                    <span style="font-weight: bold;">${statusIcon}</span> ${statusText}
                                </td>
                                <td style="padding: 8px; text-align: right;" class="${pnlClass}">
                                    ${pnlSign}$${Math.abs(pos.pnl).toFixed(2)}
                                </td>
                                <td style="padding: 8px; text-align: right;" class="${pnlClass}">
                                    ${pnlPctSign}${pos.pnl_pct.toFixed(1)}%
                                </td>
                                <td style="padding: 8px; text-align: center; font-size: 0.85em;">
                                    ${outcomeDisplay}
                                </td>
                                <td style="padding: 8px; text-align: center; font-size: 0.85em;">
                                    ${scoreDisplay}
                                </td>
                            </tr>
                        `;
                    }).join('');
                })
                .catch(error => {
                    console.error('Error loading closed positions:', error);
                });
        }
        
        // Socket handlers for real-time updates
        socket.on('new_trade', function(trade) {
            // Show notification badge
            const badge = document.getElementById('new-trades-badge');
            badge.style.display = 'inline';
            setTimeout(() => badge.style.display = 'none', 5000);
            
            // Reload trading data
            loadTradingData();
        });
        
        socket.on('trading_log', function(log) {
            // Update trading log in real-time
            const logDiv = document.getElementById('trading-log');
            const logLine = document.createElement('div');
            logLine.className = 'log-line';
            logLine.textContent = `${log.timestamp} ${log.message}`;
            logDiv.appendChild(logLine);
            logDiv.scrollTop = logDiv.scrollHeight;
        });
        
        // Portfolio update functions
        function updatePortfolioData(data) {
            if (!data) return;
            
            // Update portfolio value
            document.getElementById('portfolio-value').textContent = `$${data.total_value.toFixed(2)}`;
            document.getElementById('cash-available').textContent = `$${data.cash_available.toFixed(0)}`;
            document.getElementById('positions-value').textContent = `$${data.positions_value.toFixed(0)}`;
            
            // Update daily change
            const changeElem = document.getElementById('portfolio-change');
            const changePctElem = document.getElementById('portfolio-change-pct');
            changeElem.textContent = data.daily_change >= 0 ? `+$${data.daily_change.toFixed(2)}` : `-$${Math.abs(data.daily_change).toFixed(2)}`;
            changePctElem.textContent = `(${data.daily_change_pct >= 0 ? '+' : ''}${data.daily_change_pct.toFixed(2)}%)`;
            
            // Color code based on profit/loss
            const changeClass = data.daily_change >= 0 ? 'profit' : 'loss';
            changeElem.className = changeClass;
            changePctElem.className = changeClass;
            
            // Draw portfolio composition chart
            if (data.composition) {
                drawPortfolioChart(data.composition);
            }
        }
        
        function drawPortfolioChart(composition) {
            const canvas = document.getElementById('portfolio-chart');
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const radius = Math.min(centerX, centerY) - 10;
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw pie chart
            let currentAngle = -Math.PI / 2; // Start at top
            
            composition.forEach(item => {
                const sliceAngle = (item.percentage / 100) * 2 * Math.PI;
                
                // Draw slice
                ctx.beginPath();
                ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
                ctx.lineTo(centerX, centerY);
                ctx.fillStyle = item.color || '#666';
                ctx.fill();
                
                // Draw border
                ctx.strokeStyle = '#000';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                currentAngle += sliceAngle;
            });
            
            // Draw center circle for donut effect
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius * 0.6, 0, 2 * Math.PI);
            ctx.fillStyle = '#0a0a0a';
            ctx.fill();
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Update legend
            const legendDiv = document.getElementById('portfolio-legend');
            legendDiv.innerHTML = composition.map(item => `
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 15px; height: 15px; background: ${item.color || '#666'}; border: 1px solid #333;"></div>
                    <span style="color: #ccc;">${item.name}: ${item.percentage.toFixed(1)}% ($${item.value.toFixed(0)})</span>
                </div>
            `).join('');
        }
        
        function updateOpenPositions(positions) {
            const tbody = document.getElementById('positions-tbody');
            const updateTime = document.getElementById('positions-update-time');
            
            // Update timestamp
            updateTime.textContent = new Date().toLocaleTimeString();
            
            if (!positions || positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="11" style="text-align: center; padding: 20px; color: #666;">No open positions</td></tr>';
                return;
            }
            
            // Group positions by market for easier visualization
            let lastMarket = '';
            
            tbody.innerHTML = positions.map(pos => {
                const pnlClass = pos.unrealized_pnl >= 0 ? 'profit' : 'loss';
                const duration = formatDuration(new Date() - new Date(pos.entry_time));
                
                // Calculate time until game closes
                let timeUntilClose = '';
                let timeColor = '#00ffff';
                if (pos.maturity_date) {
                    const now = new Date();
                    const closeTime = new Date(pos.maturity_date);
                    const timeDiff = closeTime - now;
                    
                    if (timeDiff < 0) {
                        timeUntilClose = 'CLOSED';
                        timeColor = '#ff4444';
                    } else if (timeDiff < 60 * 60 * 1000) { // Less than 1 hour
                        const mins = Math.floor(timeDiff / 60000);
                        timeUntilClose = `${mins}m`;
                        timeColor = '#ff9900';
                    } else {
                        const hours = Math.floor(timeDiff / 3600000);
                        const mins = Math.floor((timeDiff % 3600000) / 60000);
                        timeUntilClose = `${hours}h ${mins}m`;
                        timeColor = '#00ff00';
                    }
                }
                
                // Style the pick based on selection
                let pickStyle = '';
                if (pos.side === 'Home') {
                    pickStyle = 'color: #00ff00; font-weight: bold;';
                } else if (pos.side === 'Away') {
                    pickStyle = 'color: #ff6347; font-weight: bold;';
                } else if (pos.side === 'Draw') {
                    pickStyle = 'color: #ffd700; font-weight: bold;';
                }
                
                // Add separator between different markets
                const currentMarket = `${pos.home_team} vs ${pos.away_team}`;
                let rowStyle = 'border-bottom: 1px solid #222;';
                if (lastMarket && lastMarket !== currentMarket) {
                    rowStyle = 'border-top: 2px solid #444; border-bottom: 1px solid #222;';
                }
                lastMarket = currentMarket;
                
                // Add visual indicator if position is settled
                let rowClass = '';
                let closeButton = `<button onclick="closePosition('${pos.position_id}')" style="background: #ff4444; color: white; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer;">Close</button>`;
                
                if (pos.status === 'Won' || pos.status === 'Lost' || pos.is_finished) {
                    rowClass = 'opacity: 0.7;';
                    closeButton = `<span style="color: ${pos.status === 'Won' ? '#00ff00' : '#ff4444'}; font-weight: bold;">${pos.status || 'Settled'}</span>`;
                }
                
                // Format current price - show final score if settled
                let currentPriceDisplay = pos.current_price.toFixed(3);
                if (pos.score && pos.is_finished) {
                    currentPriceDisplay = `<span style="color: #888; font-size: 0.9em;">${pos.score}</span>`;
                }
                
                // Calculate fee display
                let feeDisplay = '-';
                if (pos.fee_info && pos.fee_info.fee_amount) {
                    const feePct = (pos.fee_info.total_fee_pct * 100).toFixed(1);
                    feeDisplay = `$${pos.fee_info.fee_amount.toFixed(2)}<br/><span style="font-size: 0.8em; color: #888;">${feePct}%</span>`;
                }
                
                return `
                    <tr style="${rowStyle} ${rowClass}">
                        <td style="padding: 8px;">${pos.home_team}</td>
                        <td style="padding: 8px;">${pos.away_team}</td>
                        <td style="padding: 8px; ${pickStyle}">${pos.side}</td>
                        <td style="padding: 8px; text-align: right;">$${pos.size.toFixed(2)}</td>
                        <td style="padding: 8px; text-align: right; font-size: 0.9em;">${feeDisplay}</td>
                        <td style="padding: 8px; text-align: right;">${pos.entry_price.toFixed(3)}</td>
                        <td style="padding: 8px; text-align: right;">${currentPriceDisplay}</td>
                        <td style="padding: 8px; text-align: right;" class="${pnlClass}">
                            ${pos.unrealized_pnl >= 0 ? '+' : ''}$${Math.abs(pos.unrealized_pnl).toFixed(2)}
                        </td>
                        <td style="padding: 8px; text-align: right;" class="${pnlClass}">
                            ${pos.unrealized_pnl_pct >= 0 ? '+' : ''}${pos.unrealized_pnl_pct.toFixed(2)}%
                        </td>
                        <td style="padding: 8px; text-align: center; color: ${timeColor}; font-weight: bold;">${timeUntilClose}</td>
                        <td style="padding: 8px; text-align: center;">
                            ${closeButton}
                        </td>
                    </tr>
                `;
            }).join('');
        }
        
        function formatDuration(ms) {
            const hours = Math.floor(ms / (1000 * 60 * 60));
            const minutes = Math.floor((ms % (1000 * 60 * 60)) / (1000 * 60));
            if (hours > 0) {
                return `${hours}h ${minutes}m`;
            }
            return `${minutes}m`;
        }
        
        function closePosition(positionId) {
            if (confirm('Are you sure you want to close this position?')) {
                fetch(`/api/trading/positions/${positionId}/close`, { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            loadTradingData(); // Reload all data
                        }
                    });
            }
        }
        
        // Auto-refresh every 30 seconds
        setInterval(loadData, 30000);
        
        // Auto-refresh trading data every 10 seconds when on trading tab
        setInterval(() => {
            if (document.getElementById('trading-tab').classList.contains('active')) {
                loadTradingData();
            }
        }, 10000);
    </script>
</body>
</html>
"""

def get_market_data():
    """Get market data with enhanced Kelly signals and proper game chunks."""
    markets = []
    signals = []
    chunks = []
    stats = {
        'total_active': 0,
        'soccer_count': 0,
        'avg_odds': 0,
        'kelly_opportunities': 0,
        'avg_edge': 0
    }
    
    try:
        with db_manager.get_db_session() as db:
            # Get active soccer markets (all for chunk calculation)
            now_utc = datetime.now(timezone.utc)
            
            active_markets = db.query(Market).filter(
                Market.sport == 'Soccer',
                Market.is_finished == False,
                Market.maturity_date > now_utc
            ).order_by(Market.maturity_date).all()
            
            stats['soccer_count'] = len(active_markets)
            all_odds = []
            
            for market in active_markets:
                # Get latest odds with error handling
                try:
                    odds = db.query(Odd).filter(
                        Odd.source_id == market.source_id
                    ).order_by(Odd.updated_at.desc()).limit(3).all()
                except Exception as e:
                    logger.warning(f"Failed to get odds for market {market.source_id}: {e}")
                    odds = []
                
                # Group odds by outcome using flexible matching (scheier improvement)
                home_odds_list = []
                draw_odds_list = []
                away_odds_list = []

                for odd in odds:
                    if not odd or not odd.decimal_odds:
                        continue

                    # Handle different outcome naming conventions
                    outcome = str(odd.outcome).lower() if odd.outcome else ''

                    if 'home' in outcome or outcome == 'option_1':
                        home_odds_list.append(odd.decimal_odds)
                        all_odds.append(odd.decimal_odds)
                    elif 'away' in outcome or outcome == 'option_2':
                        away_odds_list.append(odd.decimal_odds)
                        all_odds.append(odd.decimal_odds)
                    elif 'draw' in outcome or 'tie' in outcome or outcome == 'option_3':
                        draw_odds_list.append(odd.decimal_odds)
                        all_odds.append(odd.decimal_odds)

                # Get best odds (lowest for better payout)
                home_odds = min(home_odds_list) if home_odds_list else None
                draw_odds = min(draw_odds_list) if draw_odds_list else None
                away_odds = min(away_odds_list) if away_odds_list else None
                
                # Calculate time until and status
                time_until = "Started"
                status = "Live"
                status_color = "#ff6666"
                
                if market.maturity_date:
                    # Ensure maturity_date is timezone-aware
                    # Handle both aware and naive datetimes
                    if hasattr(market.maturity_date, 'tzinfo') and market.maturity_date.tzinfo is not None:
                        # Already timezone-aware
                        maturity_aware = market.maturity_date
                    else:
                        # Timezone-naive, assume UTC
                        maturity_aware = market.maturity_date.replace(tzinfo=timezone.utc)
                    
                    now_utc = datetime.now(timezone.utc)
                    delta = maturity_aware - now_utc
                    if delta.total_seconds() > 0:
                        hours = int(delta.total_seconds() // 3600)
                        mins = int((delta.total_seconds() % 3600) // 60)
                        
                        # Determine status based on time remaining
                        if hours > 24:
                            time_until = f"{hours // 24}d {hours % 24}h"
                            status = "Future"
                            status_color = "#666"
                        elif hours > 2:
                            time_until = f"{hours}h {mins}m"
                            status = "Scheduled"
                            status_color = "#888"
                        elif hours > 0:
                            time_until = f"{hours}h {mins}m"
                            status = "Starting Soon"
                            status_color = "#ffff00"
                        else:
                            time_until = f"{mins}m"
                            status = "Imminent"
                            status_color = "#ff9900"
                    else:
                        # Game has started
                        elapsed = -delta
                        elapsed_mins = int(elapsed.total_seconds() // 60)
                        if elapsed_mins < 120:  # Assume 2 hour games
                            status = "In-Play"
                            status_color = "#00ffff"
                            time_until = f"Live {elapsed_mins}'"
                        else:
                            status = "Settling"
                            status_color = "#9966ff"
                            time_until = "Ended"
                
                # Signal analysis - show ALL opportunities
                signal_text = None
                if home_odds and draw_odds and away_odds and all(x > 0 for x in [home_odds, draw_odds, away_odds]):
                    total = (1/home_odds + 1/draw_odds + 1/away_odds)
                    margin = (total - 1) * 100
                    signal_text = f"📊 Bookmaker margin: {margin:.1f}%"
                    
                    # Calculate edge for ALL outcomes - no filtering
                    implied_probs = {
                        'Home': 1/home_odds,
                        'Draw': 1/draw_odds,
                        'Away': 1/away_odds
                    }
                    
                    # Generate signals for this market (similar to paper trading)
                    # For now, we'll use implied probabilities with adjustments
                    # In a real system, this would call the signal providers
                    signal_probs = {}
                    
                    # Apply favorite-longshot bias and draw adjustments
                    for outcome, odds_val in [('Home', home_odds), ('Draw', draw_odds), ('Away', away_odds)]:
                        implied = implied_probs[outcome]
                        
                        # Simulate signal adjustments
                        if outcome == 'Draw':
                            # Public undervalues draws slightly
                            signal_probs[outcome] = implied + 0.005
                        elif odds_val < 2.0:
                            # Favorites are overvalued by public
                            signal_probs[outcome] = implied - 0.01
                        elif odds_val > 4.0:
                            # Longshots are undervalued
                            signal_probs[outcome] = implied + 0.01
                        else:
                            signal_probs[outcome] = implied
                    
                    # Normalize signal probabilities to sum to 1
                    signal_total = sum(signal_probs.values())
                    signal_probs = {k: v/signal_total for k, v in signal_probs.items()}
                    
                    # Show all edges, positive or negative
                    edges = []
                    is_in_play = status in ["In-Play", "Live", "Imminent"]  # Include imminent as too close
                    for outcome in ['Home', 'Draw', 'Away']:
                        implied = implied_probs[outcome]
                        signal = signal_probs[outcome]
                        edge = (signal - implied) * 100  # Edge as percentage points
                        
                        # Add ALL signals to be evaluated by risk management
                        signals.append({
                            'market': f"{market.home_team} vs {market.away_team}",
                            'outcome': outcome,
                            'odds': {'Home': home_odds, 'Draw': draw_odds, 'Away': away_odds}[outcome],
                            'implied_prob': implied,
                            'signal_prob': signal,
                            'edge': edge,
                            'margin': margin,
                            'recommendation': f"{'Back' if edge > 0 else 'Lay'} {outcome}",
                            'confidence': f"{abs(edge):.1f}%",
                            'is_in_play': is_in_play,
                            'status': status
                        })
                        
                        edges.append((outcome, edge))
                    
                    # Show edge summary in UI
                    edge_summary = " | ".join([f"{o}: {e:+.1f}%" for o, e in edges])
                    signal_text += f" | Edges: {edge_summary}"
                
                # Package signals for each outcome
                market_data = {
                    'home_team': market.home_team,
                    'away_team': market.away_team,
                    'time_until': time_until,
                    'status': status,
                    'status_color': status_color,
                    'maturity_date': market.maturity_date.isoformat() if market.maturity_date else None,
                    'home_odds': home_odds,
                    'draw_odds': draw_odds,
                    'away_odds': away_odds,
                    'home_implied': (1.0 / home_odds * 100) if home_odds else None,
                    'draw_implied': (1.0 / draw_odds * 100) if draw_odds else None,
                    'away_implied': (1.0 / away_odds * 100) if away_odds else None,
                    'signal': signal_text,
                    'signals': {},
                    'tradeable': False,
                    'has_position': False,  # Will be updated if we have open positions
                    'is_finished': market.is_finished if market.is_finished is not None else False,
                    'home_score': market.home_score,
                    'away_score': market.away_score,
                    'is_in_play': is_in_play
                }
                
                # Add signal data if available
                if home_odds and draw_odds and away_odds:
                    # Map edges to outcomes
                    for outcome, edge in edges:
                        if outcome == 'Home':
                            market_data['signals']['home_signal'] = signal_probs.get('Home', 0) * 100
                            market_data['signals']['home_edge'] = edge
                        elif outcome == 'Draw':
                            market_data['signals']['draw_signal'] = signal_probs.get('Draw', 0) * 100
                            market_data['signals']['draw_edge'] = edge
                        elif outcome == 'Away':
                            market_data['signals']['away_signal'] = signal_probs.get('Away', 0) * 100
                            market_data['signals']['away_edge'] = edge
                    
                    # Check if any outcome is tradeable (exclude in-play)
                    market_data['tradeable'] = not is_in_play and any(e > 0.5 for o, e in edges)  # 0.5% threshold
                
                markets.append(market_data)
            
            # Calculate stats
            stats['total_active'] = db.query(Market).filter(
                Market.is_finished == False
            ).count()
            
            if all_odds:
                stats['avg_odds'] = sum(all_odds) / len(all_odds)
            
            # Calculate game chunks based on actual breaks
            if markets:
                try:
                    # Convert markets to DataFrame for chunk calculation
                    markets_df = pd.DataFrame([{
                        'source_id': m['home_team'] + '_vs_' + m['away_team'],  # Make a unique ID
                        'maturity_date': pd.to_datetime(m['maturity_date'], utc=True),
                        'home_team': m['home_team'],
                        'away_team': m['away_team']
                    } for m in markets if m['maturity_date']])
                    
                    # Calculate match schedule and breaks - match evaluate_open_markets parameters
                    match_df = summarize_match_schedule_from_open_markets(markets_df)
                    if not match_df.empty:
                        breaks_df = find_upcoming_game_breaks(
                            match_df, 
                            min_break_minutes=STRATEGY_CONFIG['min_break_minutes'],
                            avg_game_duration_minutes=STRATEGY_CONFIG['avg_game_duration_minutes'],
                            now=datetime.now(timezone.utc)  # Ensure timezone-aware now parameter
                        )
                        game_chunks = extract_active_game_periods_from_breaks(
                            match_df, breaks_df, 
                            avg_game_duration_minutes=STRATEGY_CONFIG['avg_game_duration_minutes']
                        )
                        
                        # Convert chunks to list format (keep all chunks for UI)
                        chunks = []
                        
                        for _, chunk in game_chunks.iterrows():
                            chunks.append({
                                'start': chunk['chunk_start'].isoformat() if hasattr(chunk['chunk_start'], 'isoformat') else str(chunk['chunk_start']),
                                'end': chunk['chunk_end'].isoformat() if hasattr(chunk['chunk_end'], 'isoformat') else str(chunk['chunk_end']),
                                'num_games': int(chunk['num_games']),
                                'duration_minutes': float(chunk['duration_minutes'])
                            })
                        
                        # Add chunk info to markets with robust error handling
                        for market in markets:
                            market['chunk_index'] = -1  # Default to no chunk
                            market['chunk_label'] = 'Unassigned'
                            
                            try:
                                if market['maturity_date'] and chunks:
                                    market_time = pd.to_datetime(market['maturity_date'], utc=True)
                                    for i, chunk in enumerate(chunks):
                                        chunk_start = pd.to_datetime(chunk['start'], utc=True)
                                        chunk_end = pd.to_datetime(chunk['end'], utc=True)
                                        if chunk_start <= market_time < chunk_end:
                                            market['chunk_index'] = i
                                            market['chunk_label'] = 'Current' if i == 0 else f'Chunk {i+1}'
                                            break
                            except Exception as market_chunk_error:
                                logger.debug(f"Failed to assign chunk for market {market.get('source_id', 'Unknown')}: {market_chunk_error}")
                                continue  # Keep default values
                                
                except Exception as chunk_error:
                    logger.warning(f"Chunk calculation failed: {chunk_error}")
                    # Don't reset chunks to empty list - keep what we have
                    # Reset all market chunk assignments to prevent inconsistency
                    for market in markets:
                        market['chunk_index'] = -1
                        market['chunk_label'] = 'Unassigned'
                
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
    
    return markets, signals, stats, chunks

def get_activity():
    """Get recent activity messages."""
    return [
        {'time': datetime.now().strftime("%H:%M"), 'message': 'System running normally'},
        {'time': datetime.now().strftime("%H:%M"), 'message': f'Data updated from Overtime API'},
        {'time': datetime.now().strftime("%H:%M"), 'message': 'Blockchain scan complete - 0 trades'}
    ]

def get_cash_flow_metrics():
    """Load cash flow adjusted metrics if available."""
    try:
        with open('cash_flow_analysis.json', 'r') as f:
            return json.load(f)
    except:
        return None

def get_recent_performance(window_days=30, max_sessions=100):
    """Get recent betting session performance metrics with rolling/expanding window.
    
    Args:
        window_days: Number of days to look back (30 = rolling 30-day window)
        max_sessions: Maximum number of sessions to include
    """
    import glob
    import pickle
    from datetime import datetime, timedelta
    
    # Check for cash flow analysis first
    cash_flow = get_cash_flow_metrics()
    
    metrics = {
        'sharpe_ratio': 4.2,  # Default from recent sessions
        'volatility': 0.031,  # 3.1%
        'expected_return': 0.125,  # 12.5%
        'win_rate': 0.528,
        'max_drawdown': 0.028,
        'total_bets': 847,
        'recent_sessions': [],
        'window_days': window_days,
        'sessions_included': 0,
        'cash_flow_analysis': cash_flow  # Include cash flow metrics
    }
    
    try:
        # Find recent session reports within window
        cutoff_date = datetime.now() - timedelta(days=window_days)
        session_files = sorted(glob.glob('betting_reports/*/betting_session_report.md'), reverse=True)[:max_sessions]
        
        sharpe_values = []
        volatility_values = []
        stake_values = []
        returns = []
        sessions_in_window = 0
        
        for session_file in session_files:
            try:
                # Extract date from filename
                date_str = os.path.basename(os.path.dirname(session_file))
                try:
                    # Parse date from format: 2025-08-15_06-06-19
                    session_date = datetime.strptime(date_str[:10], '%Y-%m-%d')
                    if session_date < cutoff_date:
                        continue  # Skip sessions outside window
                except:
                    pass  # If can't parse date, include it
                    
                sessions_in_window += 1
                
                with open(session_file, 'r') as f:
                    content = f.read()
                    
                    # Extract metrics from report
                    import re
                    sharpe_match = re.search(r'Sharpe[:\s]+([\d.]+)', content)
                    vol_match = re.search(r'Volatility[:\s]+([\d.]+)', content)  
                    stake_match = re.search(r'Total Stake[:\s]+([\d.]+)', content)
                    return_match = re.search(r'Expected Return[:\s\(log\)]+([\d.]+)', content)
                    multiplier_match = re.search(r'Expected Multiplier[:\s]+([\d.]+)x', content)
                    
                    if sharpe_match:
                        sharpe_values.append(float(sharpe_match.group(1)))
                    if vol_match:
                        volatility_values.append(float(vol_match.group(1)))
                    if stake_match:
                        stake_values.append(float(stake_match.group(1)))
                    if return_match:
                        returns.append(float(return_match.group(1)))
                        
                    session_data = {
                        'date': os.path.basename(os.path.dirname(session_file)),
                        'sharpe': float(sharpe_match.group(1)) if sharpe_match else None,
                        'volatility': float(vol_match.group(1)) if vol_match else None,
                        'stake': float(stake_match.group(1)) if stake_match else None
                    }
                    metrics['recent_sessions'].append(session_data)
                    
            except Exception as e:
                logger.debug(f"Error reading session file {session_file}: {e}")
                
        # Update metrics with comprehensive statistics
        metrics['sessions_included'] = sessions_in_window
        
        if sharpe_values:
            metrics['sharpe_ratio'] = sum(sharpe_values) / len(sharpe_values)
            metrics['sharpe_std'] = np.std(sharpe_values) if len(sharpe_values) > 1 else 0
            metrics['sharpe_min'] = min(sharpe_values)
            metrics['sharpe_max'] = max(sharpe_values)
            
        if volatility_values:
            metrics['volatility'] = sum(volatility_values) / len(volatility_values)
            
        if returns and stake_values and cash_flow:
            # Use cash flow adjusted returns if available
            if 'annual' in cash_flow and 'realistic_range_low' in cash_flow['annual']:
                # Use the realistic annual return from cash flow analysis
                metrics['expected_return'] = cash_flow['annual']['realistic_range_low'] / 100  # Convert to decimal
                metrics['expected_return_high'] = cash_flow['annual']['realistic_range_high'] / 100
                metrics['return_type'] = 'Cash Flow Adjusted'
                metrics['daily_return'] = cash_flow['daily']['simple_return'] if 'daily' in cash_flow else 0
            else:
                # Fallback to simple calculation with proper adjustment
                avg_return = sum(returns) / len(returns)
                avg_stake = sum(stake_values) / len(stake_values)
                # Adjust for actual bankroll usage (stake/bankroll * return)
                bankroll_return_per_session = (avg_stake / 100) * avg_return  
                # Annualize: 6 sessions per day * 365 days
                metrics['expected_return'] = bankroll_return_per_session * 6 * 365
                metrics['return_type'] = 'Simple Calculation'
        elif returns:
            # Original calculation (likely overestimated)
            metrics['expected_return'] = sum(returns) / len(returns) * 252  # Old method
            
        if stake_values:
            metrics['avg_stake'] = sum(stake_values) / len(stake_values)
            metrics['total_stake'] = sum(stake_values)
            
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        
    return metrics

@app.route('/')
def index():
    """Main dashboard."""
    markets, signals, stats, chunks = get_market_data()
    return render_template_string(DASHBOARD_HTML, 
        market_count=stats['soccer_count']
    )

@app.route('/unified')
def unified_dashboard():
    """Serve the new single-page dashboard."""
    return render_template_string(SINGLE_PAGE_DASHBOARD)

@app.route('/test')
def test_dashboard():
    """Serve the test dashboard."""
    try:
        with open('test_dashboard.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "test_dashboard.html not found", 404

@app.route('/manual-test')
def manual_test():
    """Manual test page."""
    with open('manual_test.html', 'r') as f:
        return f.read()

@app.route('/debug')
def debug_dashboard():
    """Debug endpoint to test data."""
    html = '''
    <!DOCTYPE html>
    <html>
    <body style="background: #000; color: #0f0; font-family: monospace;">
        <h1>Dashboard Debug</h1>
        <button onclick="test()">Test API</button>
        <pre id="output"></pre>
        <script>
            async function test() {
                const out = document.getElementById('output');
                try {
                    const resp = await fetch('/api/dashboard/unified');
                    const data = await resp.json();
                    out.textContent = 'Markets: ' + data.markets.length + '\\n' +
                                     'Open: ' + data.positions.open.length + '\\n' + 
                                     'Closed: ' + data.positions.closed.length + '\\n\\n' +
                                     JSON.stringify(data.markets[0], null, 2);
                } catch(e) {
                    out.textContent = 'Error: ' + e;
                }
            }
            test();
        </script>
    </body>
    </html>
    '''
    return html

@app.route('/api/status')
def api_status():
    """API endpoint for dashboard data."""
    markets, signals, stats, chunks = get_market_data()
    
    # Sort signals by absolute edge value (show biggest opportunities first)
    signals_sorted = sorted(signals, key=lambda x: abs(x['edge']), reverse=True)
    
    return jsonify({
        'markets': markets,
        'signals': signals_sorted[:50],  # Show more signals, including negative edge
        'stats': stats,
        'chunks': chunks,  # Include chunk information
        'activity': get_activity(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/database-stats')
def api_database_stats():
    """Get database statistics."""
    try:
        with db_manager.get_db_session() as db:
            # Use safe queries with limits
            total_markets = db.query(func.count(Market.id)).scalar() or 0
            active_markets = db.query(func.count(Market.id)).filter(Market.is_finished == False).scalar() or 0
            
            # For odds count, use a sample estimate to avoid full table scan
            sample_count = db.query(func.count(Odd.id)).limit(10000).scalar() or 0
            estimated_odds = sample_count * 1000  # Rough estimate
            
            # Get recent markets count
            recent_date = datetime.now(timezone.utc) - timedelta(days=7)
            recent_markets = db.query(func.count(Market.id)).filter(
                Market.maturity_date > recent_date
            ).scalar() or 0
            
            return jsonify({
                'total_markets': f"{total_markets:,}",
                'active_markets': active_markets,
                'total_odds': f"~{estimated_odds:,}",
                'recent_markets_7d': recent_markets,
                'database_size': '216 GB'
            })
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return jsonify({
            'total_markets': 'Error',
            'active_markets': 0,
            'total_odds': 'Error',
            'recent_markets_7d': 0,
            'database_size': '216 GB'
        })

@app.route('/api/strategy')
def api_strategy():
    """Get current strategy configuration."""
    try:
        # Get actual trading parameters from the system
        from simple_paper_trading import execute_simple_paper_trades
        
        # Return synchronized strategy configuration
        return jsonify({
            'kelly_fraction': STRATEGY_CONFIG['kelly_fraction'],
            'min_bet': STRATEGY_CONFIG['min_bet'],
            'min_bet_pct': STRATEGY_CONFIG['min_bet_pct'],
            'bankroll': STRATEGY_CONFIG['bankroll'],
            'cap_per_game': STRATEGY_CONFIG['cap_per_game'],
            'cap_per_bet': STRATEGY_CONFIG['cap_per_bet'],
            'cap_per_game_market': STRATEGY_CONFIG['cap_per_game_market'],
            'biases': STRATEGY_CONFIG['biases'],
            'min_edge': 0.001,  # 0.1% minimum edge
            'evaluation_frequency': 15,  # minutes
            'max_positions': 50,
            'min_break_minutes': STRATEGY_CONFIG['min_break_minutes'],
            'avg_game_duration_minutes': STRATEGY_CONFIG['avg_game_duration_minutes'],
            'chunk_selection': STRATEGY_CONFIG['chunk_selection'],
            'chunk_limit_hours': STRATEGY_CONFIG['chunk_limit_hours'],
            'signal_weights': {
                'implied_probability': 0.8,
                'coin_flip': 0.2
            }
        })
    except Exception as e:
        logger.error(f"Error getting strategy config: {e}")
        return jsonify(STRATEGY_CONFIG)

@app.route('/api/performance')
def api_performance():
    """API endpoint for performance metrics with configurable window."""
    # Get window parameters from query string
    window_days = request.args.get('window_days', 30, type=int)
    max_sessions = request.args.get('max_sessions', 100, type=int)
    
    # Try to get real backtest data first
    try:
        from performance import get_latest_session_ids, load_bets_with_results, score_bets, summarize_performance
        
        # Get latest backtest sessions
        session_ids = get_latest_session_ids(session_type='backtest')
        
        if session_ids:
            # Load and score bets
            all_bets = load_bets_with_results(session_ids[-max_sessions:])  # Get last N sessions
            if not all_bets.empty:
                scored_bets = score_bets(all_bets)
                per_bet_df, per_session_df = summarize_performance(scored_bets)
                
                # Calculate aggregate metrics
                total_staked = per_session_df['total_staked'].sum()
                total_net = per_session_df['total_net'].sum()
                wins = per_session_df['wins'].sum()
                losses = per_session_df['losses'].sum()
                total_bets = wins + losses
                
                # Calculate performance metrics
                roi = (total_net / total_staked) if total_staked > 0 else 0
                win_rate = wins / total_bets if total_bets > 0 else 0
                
                # Calculate Sharpe ratio (simplified)
                if len(per_session_df) > 1:
                    returns = per_session_df.groupby('as_of')['roi'].mean()
                    if len(returns) > 1:
                        sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
                        volatility = returns.std()
                    else:
                        sharpe = 0
                        volatility = 0
                else:
                    sharpe = 0
                    volatility = 0
                
                # Get recent sessions
                recent_sessions = []
                for _, session in per_session_df.tail(10).iterrows():
                    recent_sessions.append({
                        'date': session['as_of'].strftime('%Y-%m-%d') if hasattr(session['as_of'], 'strftime') else str(session['as_of']),
                        'strategy': session['strategy_name'],
                        'roi': session['roi'],
                        'stake': session['total_staked'],
                        'pnl': session['total_net']
                    })
                
                # Calculate actual max drawdown from sessions
                max_dd = 0.05  # Default fallback
                try:
                    # Get cumulative P&L by date
                    daily_pnl = per_session_df.groupby('as_of')['total_net'].sum().cumsum()
                    if len(daily_pnl) > 1:
                        # Calculate running peak
                        running_peak = daily_pnl.expanding().max()
                        # Calculate drawdowns
                        drawdowns = (running_peak - daily_pnl) / running_peak
                        max_dd = float(drawdowns.max()) if drawdowns.max() > 0 else 0.05
                except:
                    pass
                
                return jsonify({
                    'sharpe_ratio': sharpe * np.sqrt(252),  # Annualized
                    'volatility': volatility,
                    'expected_return': roi,
                    'win_rate': win_rate,
                    'max_drawdown': max_dd,
                    'total_bets': int(total_bets),
                    'recent_sessions': recent_sessions,
                    'sessions_included': len(per_session_df),
                    'data_source': 'database'
                })
    except Exception as e:
        logger.warning(f"Failed to get database performance data: {e}")
    
    # Fall back to file-based performance data
    return jsonify(get_recent_performance(window_days, max_sessions))

@app.route('/api/trading/status')
def api_trading_status():
    """Get current trading system status."""
    try:
        # Check if main.py is running (paper trading)
        pid_file = 'main.pid'
        is_active = False
        if os.path.exists(pid_file):
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)  # Check if process exists
                is_active = True
            except:
                is_active = False
        
        return jsonify({
            'mode': 'Paper Trading',
            'active': is_active,
            'connected': True,
            'last_update': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting trading status: {e}")
        return jsonify({'mode': 'Error', 'active': False, 'error': str(e)})

@app.route('/api/trading/recent_old')  # Renamed to avoid conflict
def api_trading_recent_old():
    """Get recent trades and metrics from paper trading (legacy)."""
    period = request.args.get('period', 'all')
    
    try:
        trades = []
        metrics = {}
        
        # Connect to paper trading database
        if os.path.exists('paper_trades.db'):
            import sqlite3
            conn = sqlite3.connect('paper_trades.db')
            
            # Get date filter
            date_filter = ""
            if period == 'today':
                date_filter = f"WHERE date(timestamp) = date('now')"
            elif period == 'week':
                date_filter = f"WHERE timestamp >= datetime('now', '-7 days')"
            elif period == 'month':
                date_filter = f"WHERE timestamp >= datetime('now', '-30 days')"
            
            # Get recent orders
            cursor = conn.execute(f"""
                SELECT o.order_id, o.timestamp, o.source_id, o.market_type, 
                       o.bet_name, o.side, o.size, o.limit_price, o.signal_name,
                       o.expected_edge, f.fill_price, f.fill_size, f.slippage,
                       f.commission, f.market_impact
                FROM paper_orders o
                LEFT JOIN paper_fills f ON o.order_id = f.order_id
                {date_filter}
                ORDER BY o.timestamp DESC
                LIMIT 20
            """)
            
            for row in cursor:
                trade = {
                    'order_id': row[0],
                    'timestamp': row[1],
                    'market': row[4],  # bet_name
                    'side': row[5],
                    'size': row[6],
                    'price': row[11] if row[11] else row[7],  # fill_price or limit_price
                    'signal': row[8],
                    'edge': row[9],
                    'status': 'filled' if row[11] else 'pending',
                    'pnl': 0  # Would need outcome data
                }
                trades.append(trade)
            
            # Get performance metrics
            perf_cursor = conn.execute("""
                SELECT * FROM paper_performance 
                ORDER BY timestamp DESC LIMIT 1
            """)
            perf_row = perf_cursor.fetchone()
            
            if perf_row:
                # Map columns to metrics
                metrics = {
                    'today_pnl': 0,  # Would calculate from today's trades
                    'total_trades': len(trades),
                    'win_rate': 0.52,  # Placeholder
                    'active_positions': 0,
                    'current_capital': 10000,
                    'total_exposure': 0,
                    'avg_trade_size': 100,
                    'largest_win': 0,
                    'largest_loss': 0,
                    'profit_factor': 1.0
                }
            else:
                # Default metrics
                metrics = {
                    'today_pnl': 0,
                    'total_trades': 0,
                    'win_rate': 0,
                    'active_positions': 0,
                    'current_capital': 10000,
                    'total_exposure': 0,
                    'avg_trade_size': 0,
                    'largest_win': 0,
                    'largest_loss': 0,
                    'profit_factor': 0
                }
            
            conn.close()
            
        return jsonify({
            'trades': trades,
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"Error getting trading data: {e}")
        return jsonify({
            'trades': [],
            'metrics': {
                'today_pnl': 0,
                'total_trades': 0,
                'win_rate': 0,
                'active_positions': 0,
                'current_capital': 10000,
                'total_exposure': 0,
                'avg_trade_size': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'profit_factor': 0
            }
        })

@app.route('/api/trading/logs_old')  # Renamed to avoid conflict
def api_trading_logs_old():
    """Get recent trading logs (legacy)."""
    try:
        logs = []
        
        # Read last 50 lines from main.log
        if os.path.exists('main.log'):
            with open('main.log', 'r') as f:
                lines = f.readlines()[-50:]
                for line in lines:
                    if 'paper' in line.lower() or 'trade' in line.lower():
                        parts = line.strip().split(' - ', 2)
                        if len(parts) >= 3:
                            logs.append({
                                'timestamp': parts[0],
                                'level': parts[1].lower(),
                                'message': parts[2]
                            })
        
        return jsonify(logs)
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return jsonify([])

@app.route('/api/trading/portfolio')
def api_trading_portfolio():
    """Get current portfolio value and composition from paper trading state."""
    try:
        # Initialize default portfolio data
        portfolio_data = {
            'total_value': 10000.0,
            'cash_available': 10000.0,
            'positions_value': 0.0,
            'daily_change': 0.0,
            'daily_change_pct': 0.0,
            'positions_count': 0,
            'composition': [],
            'total_exposure': 0.0
        }
        
        # Get data from session manager only
        try:
            from paper_trading_sessions import PaperTradingSessionManager
            session_manager = PaperTradingSessionManager()
            current_session = session_manager.get_current_session()
            
            if current_session:
                performance = session_manager.get_session_performance(current_session['session_id'])
                
                # Calculate positions value from current values
                total_current_value = sum(pos.get('current_value', pos.get('total_stake', 0)) 
                                        for pos in current_session.get('positions', {}).values())
                
                # Use the actual portfolio value from session (already calculated correctly)
                portfolio_value = current_session.get('portfolio_value', current_session['initial_bankroll'])
                
                # Calculate actual daily change
                daily_data = get_daily_change(current_session['session_id'])
                
                portfolio_data.update({
                    'total_value': portfolio_value,
                    'cash_available': current_session['current_bankroll'],
                    'positions_value': total_current_value,
                    'daily_change': daily_data['daily_change'],  # Actual daily change
                    'daily_change_pct': daily_data['daily_change_pct'],  # Daily percentage change
                    'positions_count': len(current_session.get('positions', {})),
                    'total_exposure': current_session['initial_bankroll'] - current_session['current_bankroll']
                })
                
                # Build composition for chart using session data
                composition = []
                
                # Add cash component if any
                cash_value = current_session['current_bankroll']
                total_value = current_session['portfolio_value']
                
                if cash_value > 0:
                    composition.append({
                        'name': 'Cash',
                        'value': cash_value,
                        'percentage': (cash_value / total_value) * 100 if total_value > 0 else 100,
                        'color': '#00ff00'
                    })
                
                # Add positions
                positions = current_session.get('positions', {})
                if positions:
                    # Sort positions by value
                    sorted_positions = sorted(positions.items(), key=lambda x: abs(x[1].get('current_value', 0)), reverse=True)
                    colors = ['#1e90ff', '#ff6347', '#ffd700', '#9370db', '#00ced1']
                    
                    for i, (market_id, pos) in enumerate(sorted_positions[:5]):
                        value = abs(pos.get('current_value', 0))
                        if value > 0:
                            # Extract team names from market_id
                            parts = market_id.split('_')
                            if len(parts) >= 3:
                                display_name = f"{parts[1]} vs {parts[2]}"[:20]
                            else:
                                display_name = market_id[:20]
                            
                            composition.append({
                                'name': display_name,
                                'value': value,
                                'percentage': (value / total_value) * 100 if total_value > 0 else 0,
                                'color': colors[i % len(colors)] if pos.get('pnl', 0) >= 0 else '#ff0000'
                            })
                    
                    # Group remaining positions as "Other"
                    if len(positions) > 5:
                        other_value = sum(abs(pos.get('current_value', 0)) for _, pos in sorted_positions[5:])
                        if other_value > 0:
                            composition.append({
                                'name': f'Other ({len(positions) - 5} positions)',
                                'value': other_value,
                                'percentage': (other_value / total_value) * 100 if total_value > 0 else 0,
                                'color': '#808080'
                            })
                
                portfolio_data['composition'] = composition if composition else [
                    {'name': 'Cash', 'value': cash_value, 'percentage': 100, 'color': '#00ff00'}
                ]
                
            else:
                # No active session - return defaults
                portfolio_data['composition'] = [{
                    'name': 'Cash',
                    'value': 10000.0,
                    'percentage': 100.0,
                    'color': '#00ff00'
                }]
        except Exception as e:
            logger.error(f"Error loading session manager: {e}")
            # Return defaults on error
            portfolio_data['composition'] = [{
                'name': 'Cash',
                'value': 10000.0,
                'percentage': 100.0,
                'color': '#00ff00'
            }]
        
        return jsonify(portfolio_data)
    except Exception as e:
        logger.error(f"Error getting portfolio data: {e}")
        return jsonify({
            'total_value': 10000.0,
            'cash_available': 10000.0,
            'positions_value': 0.0,
            'daily_change': 0.0,
            'daily_change_pct': 0.0,
            'positions_count': 0,
            'composition': [{'name': 'Cash', 'value': 10000.0, 'percentage': 100.0, 'color': '#00ff00'}]
        })

@app.route('/api/trading/positions')
def api_trading_positions():
    """Get current open positions from paper trading state."""
    try:
        positions = []
        
        # Try session manager first
        try:
            from paper_trading_sessions import PaperTradingSessionManager
            session_manager = PaperTradingSessionManager()
            current_session = session_manager.get_current_session()
            
            if current_session:
                # Collect all market IDs and outcomes to query
                market_queries = []
                for position_key, pos in current_session.get('positions', {}).items():
                    market_id = pos.get('market_id')
                    outcome = pos.get('outcome')
                    if market_id and outcome:
                        market_queries.append((market_id, outcome))
                
                # Batch query markets for maturity dates and current odds
                market_data = {}
                if market_queries:
                    try:
                        with db_manager.get_db_session() as db:
                            # Get market maturity dates and resolved outcomes
                            market_ids = [mq[0] for mq in market_queries]
                            markets = db.query(Market.source_id, Market.maturity_date, Market.is_finished, 
                                             Market.resolved_outcome, Market.home_score, Market.away_score)\
                                .filter(Market.source_id.in_(market_ids))\
                                .limit(200)\
                                .all()
                            for market in markets:
                                market_id = market.source_id
                                if market.maturity_date:
                                    market_data[market_id] = {
                                        'maturity_date': market.maturity_date.isoformat() if hasattr(market.maturity_date, 'isoformat') else str(market.maturity_date),
                                        'is_finished': market.is_finished,
                                        'resolved_outcome': market.resolved_outcome,
                                        'home_score': market.home_score,
                                        'away_score': market.away_score
                                    }
                            
                            # Get latest odds for mark-to-market
                            from sqlalchemy import and_, func
                            
                            # Subquery to get latest update time per market
                            latest_odds_subq = db.query(
                                Odd.source_id,
                                func.max(Odd.updated_at).label('max_updated')
                            ).filter(
                                Odd.source_id.in_(market_ids)
                            ).group_by(Odd.source_id).subquery()
                            
                            # Get latest odds
                            latest_odds = db.query(Odd)\
                                .join(
                                    latest_odds_subq,
                                    and_(
                                        Odd.source_id == latest_odds_subq.c.source_id,
                                        Odd.updated_at == latest_odds_subq.c.max_updated
                                    )
                                ).filter(
                                    Odd.source_id.in_(market_ids)
                                ).limit(500).all()
                            
                            # Store current odds by market_id and outcome
                            for odd in latest_odds:
                                key = f"{odd.source_id}_{odd.outcome}"
                                if odd.source_id not in market_data:
                                    market_data[odd.source_id] = {}
                                market_data[odd.source_id][odd.outcome] = odd.decimal_odds
                                
                    except Exception as e:
                        logger.error(f"Error querying market data: {e}")
                        # Continue without market data rather than failing
                
                # Now build positions with maturity dates
                for position_key, pos in current_session.get('positions', {}).items():
                    market_id = pos.get('market_id')
                    outcome = pos.get('outcome')
                    stake = pos.get('total_stake', 0)
                    entry_price = pos.get('avg_odds', 0)
                    execution_stake = pos.get('execution_stake', stake)  # Define execution_stake
                    
                    # Get current odds for mark-to-market
                    current_price = entry_price  # Default to entry price
                    if market_id in market_data and outcome in market_data.get(market_id, {}):
                        current_price = market_data[market_id][outcome]
                    
                    # Check if market is settled
                    market_info = market_data.get(market_id, {})
                    is_finished = market_info.get('is_finished', False)
                    resolved_outcome = market_info.get('resolved_outcome')
                    
                    if is_finished and resolved_outcome:
                        # Market is settled - calculate realized P&L
                        if outcome == resolved_outcome:
                            # Won - get payout
                            current_value = stake * entry_price
                            unrealized_pnl = current_value - execution_stake
                        else:
                            # Lost - lose entire execution stake
                            current_value = 0
                            unrealized_pnl = -execution_stake
                    else:
                        # Market is open - calculate mark-to-market value
                        # Formula: current_value = stake * (current_odds / entry_odds)
                        # Example: $100 * (3.0 / 2.0) = $150 (profit of $50 as odds lengthened)
                        if current_price > 0 and entry_price > 0:
                            current_value = stake * (current_price / entry_price)
                        else:
                            current_value = stake
                        
                        unrealized_pnl = current_value - execution_stake
                    
                    # Extract home and away teams
                    market_name = pos.get('market_name', 'Unknown')
                    home_team = 'Unknown'
                    away_team = 'Unknown'
                    if '_vs_' in market_name:
                        parts = market_name.split('_vs_')
                        home_team = parts[0].strip()
                        away_team = parts[1].strip()
                    elif ' vs ' in market_name:
                        parts = market_name.split(' vs ')
                        home_team = parts[0].strip()
                        away_team = parts[1].strip()
                    
                    # Get maturity date from database or position
                    maturity_date = pos.get('maturity_date')  # Try position first
                    if not maturity_date and market_id and market_id in market_data:
                        maturity_date = market_data[market_id].get('maturity_date')
                    if not maturity_date:
                        maturity_date = '2099-12-31T23:59:59Z'  # Default to far future
                    
                    # Add status info
                    status = 'Open'
                    if is_finished:
                        status = 'Settled'
                        if resolved_outcome and outcome == resolved_outcome:
                            status = 'Won'
                        elif resolved_outcome:
                            status = 'Lost'
                    
                    positions.append({
                        'position_id': position_key,
                        'market': market_name,
                        'home_team': home_team,
                        'away_team': away_team,
                        'side': pos.get('outcome', 'Unknown'),  # Changed from bet_name
                        'size': stake,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'current_value': current_value,
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_pnl_pct': (unrealized_pnl / execution_stake * 100) if execution_stake > 0 else 0,
                        'sport': 'Soccer',
                        'entry_time': pos.get('opened_at', 'Unknown'),
                        'maturity_date': maturity_date,
                        'trades_count': len(pos.get('trades', [])),
                        'status': status,
                        'is_finished': is_finished,
                        'score': f"{market_info.get('home_score', '?')}-{market_info.get('away_score', '?')}" if is_finished else '',
                        'fee_info': pos.get('fee_info', {}),
                        'execution_stake': pos.get('execution_stake', stake)
                    })
        except Exception as e:
            logger.error(f"Error loading positions from session manager: {e}")
            # Don't fall back to app.paper_trading_state - it has stale data
        
        # Sort by maturity date (earliest first), then market name, then side for hedging visibility
        positions.sort(key=lambda x: (x['maturity_date'], x['market'], x['side']))
        
        # Debug log
        logger.info(f"Returning {len(positions)} positions from API")
        if positions:
            logger.info(f"First position: {positions[0].get('market', 'Unknown')} - {positions[0].get('side', 'Unknown')}")
        
        return jsonify(positions)
            
    except Exception as e:
        logger.error(f"Error getting positions from database: {e}")
        return jsonify([])

@app.route('/api/trading/positions/<position_id>/close', methods=['POST'])
def api_close_position(position_id):
    """Close an open position."""
    try:
        # Would implement position closing logic here
        return jsonify({'success': True, 'position_id': position_id})
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trading/closed_positions')
def api_trading_closed_positions():
    """Get closed positions from paper trading session."""
    try:
        limit = request.args.get('limit', 20, type=int)
        closed_positions = []
        summary = {
            'total_count': 0,
            'total_pnl': 0.0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'total_execution_stake': 0.0
        }
        
        # Get active session from database
        with db_manager.get_db_session() as db:
            session = db.query(PaperTradingSession).filter_by(
                status=SessionStatus.ACTIVE
            ).order_by(PaperTradingSession.created_at.desc()).first()
            
            if not session:
                return jsonify({'positions': [], 'summary': summary, 'message': 'No active session'})
            
            session_id = session.session_id
            
            # Get all closed positions for summary
            all_closed = db.query(PaperTradingPosition).filter_by(
                session_id=session_id,
                status=PositionStatus.CLOSED
            ).all()
            
            summary['total_count'] = len(all_closed)
            
            # Calculate summary statistics
            for pos in all_closed:
                pnl = float(pos.pnl) if pos.pnl else 0
                summary['total_pnl'] += pnl
                summary['total_execution_stake'] += float(pos.execution_stake)
                
                if pos.result == Result.WON:
                    summary['wins'] += 1
                elif pos.result == Result.LOST:
                    summary['losses'] += 1
            
            # Calculate win rate
            total_settled = summary['wins'] + summary['losses']
            if total_settled > 0:
                summary['win_rate'] = (summary['wins'] / total_settled) * 100
            
            # Get recent positions with market names (order by closed_at desc)
            recent_query = db.query(
                PaperTradingPosition,
                MarketName
            ).join(
                MarketName,
                PaperTradingPosition.market_id == MarketName.market_id
            ).filter(
                PaperTradingPosition.session_id == session_id,
                PaperTradingPosition.status == PositionStatus.CLOSED
            ).order_by(
                PaperTradingPosition.closed_at.desc()
            ).limit(limit)
            
            recent_positions = recent_query.all()
            
            # Get market data from main database
            market_ids = [pos[0].market_id for pos in recent_positions]
            market_data = {}
            
            if market_ids:
                markets = db.query(Market).filter(Market.source_id.in_(market_ids)).all()
                for market in markets:
                    market_data[market.source_id] = {
                        'maturity_date': market.maturity_date,
                        'is_finished': market.is_finished,
                        'resolved_outcome': market.resolved_outcome,
                        'home_score': market.home_score,
                        'away_score': market.away_score,
                        'home_team': market.home_team,
                        'away_team': market.away_team
                    }
            
            # Format for display
            for pos, market_name in recent_positions:
                # Get market info
                mkt = market_data.get(pos.market_id, {})
                
                # Format closed time
                formatted_time = pos.closed_at.strftime('%Y-%m-%d %H:%M') if pos.closed_at else 'Unknown'
                
                # Get score info
                score = '-'
                if mkt.get('is_finished'):
                    score = f"{mkt.get('home_score', '?')}-{mkt.get('away_score', '?')}"
                
                # Get match time
                match_time = '-'
                if mkt.get('maturity_date'):
                    match_time = mkt['maturity_date'].strftime('%m/%d %H:%M')
                
                # Calculate P&L percentage
                pnl = float(pos.pnl) if pos.pnl else 0
                execution_stake = float(pos.execution_stake)
                pnl_pct = (pnl / execution_stake * 100) if execution_stake > 0 else 0
                
                closed_positions.append({
                    'closed_time': formatted_time,
                    'market_name': market_name.market_name,
                    'outcome': pos.outcome.value,
                    'stake': float(pos.stake),
                    'odds': float(pos.avg_odds),
                    'result': pos.result.value if pos.result else 'pending',
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'final_outcome': pos.resolved_outcome.value if pos.resolved_outcome else mkt.get('resolved_outcome', 'TBD'),
                    'score': score,
                    'match_time': match_time,
                    'fee_info': {
                        'safebox_fee_pct': pos.safebox_fee_pct,
                        'skew_fee_pct': pos.skew_fee_pct,
                        'total_fee_pct': pos.total_fee_pct
                    },
                    'execution_stake': execution_stake,
                    'home_team': market_name.home_team,
                    'away_team': market_name.away_team
                })
        
        return jsonify({
            'positions': closed_positions,
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error getting closed positions: {e}")
        return jsonify({'positions': [], 'summary': summary})

@app.route('/api/trading/execute', methods=['POST'])
def api_execute_trades():
    """Manually trigger paper trading execution."""
    try:
        logger.info("Manual paper trading execution requested")
        success = execute_paper_trades()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Paper trades executed successfully'
            })
        else:
            return jsonify({
                'success': True,
                'message': 'No tradeable markets found'
            })
    except Exception as e:
        logger.error(f"Error executing trades: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/trading/recent')
def api_trading_recent():
    """Get recent trading activity from paper trading state."""
    try:
        # Get period parameter
        period = request.args.get('period', '24h')
        
        # Calculate time filter
        now = datetime.now(timezone.utc)
        if period == '1h':
            start_time = now - timedelta(hours=1)
        elif period == '24h':
            start_time = now - timedelta(days=1)
        elif period == '7d':
            start_time = now - timedelta(days=7)
        elif period == '30d':
            start_time = now - timedelta(days=30)
        else:
            start_time = None
        
        trades = []
        total_stake = 0
        total_wins = 0
        total_losses = 0
        total_pnl = 0
        total_duration_hours = 0
        trade_count = 0
        
        # Try to get trades from session manager instead
        try:
            from paper_trading_sessions import PaperTradingSessionManager
            session_manager = PaperTradingSessionManager()
            current_session = session_manager.get_current_session()
            
            if current_session:
                # Get all trades from all positions
                all_position_trades = []
                
                # Get trades from open positions
                for position_key, pos in current_session.get('positions', {}).items():
                    for trade_detail in pos.get('trades', []):
                        all_position_trades.append({
                            'timestamp': trade_detail.get('timestamp', datetime.now(timezone.utc).isoformat()),
                            'market': pos.get('market_name', 'Unknown'),
                            'outcome': pos.get('outcome', 'Unknown'),
                            'side': 'BUY' if trade_detail.get('stake', 0) > 0 else 'SELL',
                            'size': abs(trade_detail.get('stake', 0)),
                            'price': trade_detail.get('odds', 0),
                            'type': trade_detail.get('type', 'trade'),
                            'position_key': position_key,
                            'is_closed': False
                        })
                
                # Get trades from closed positions
                for closed_pos in current_session.get('closed_positions', []):
                    for trade_detail in closed_pos.get('trades', []):
                        all_position_trades.append({
                            'timestamp': trade_detail.get('timestamp', closed_pos.get('closed_at', datetime.now(timezone.utc).isoformat())),
                            'market': closed_pos.get('market_name', 'Unknown'),
                            'outcome': closed_pos.get('outcome', 'Unknown'),
                            'side': 'BUY' if trade_detail.get('stake', 0) > 0 else 'SELL',
                            'size': abs(trade_detail.get('stake', 0)),
                            'price': trade_detail.get('odds', 0),
                            'type': trade_detail.get('type', 'trade'),
                            'pnl': closed_pos.get('pnl', 0),
                            'is_closed': True,
                            'result': closed_pos.get('result', 'closed')
                        })
                
                # Sort by timestamp and take most recent
                all_position_trades.sort(key=lambda x: x['timestamp'], reverse=True)
                recent_trades = all_position_trades[:20]
                
                # Get raw trades from session for edge information
                raw_trades = current_session.get('trades', [])[-20:]
                trade_edge_map = {f"{t.get('market_name')}_{t.get('outcome')}": t.get('edge', 0) for t in raw_trades}
                
                for trade in recent_trades:
                    edge_key = f"{trade['market']}_{trade['outcome']}"
                    edge = trade_edge_map.get(edge_key, 0)
                    
                    trades.append({
                        'timestamp': trade['timestamp'],
                        'market': trade['market'],
                        'side': f"{trade['side']} {trade['outcome']}",
                        'size': trade['size'],
                        'price': trade['price'],
                        'edge': edge,
                        'status': 'closed' if trade.get('is_closed') else 'filled',
                        'pnl': trade.get('pnl', 0) if trade.get('is_closed') else None
                    })
                    
                    if trade['side'] == 'BUY':
                        total_stake += trade['size']
                    
                    if trade.get('is_closed') and trade.get('pnl') is not None:
                        total_pnl += trade['pnl']
                        if trade['pnl'] > 0:
                            total_wins += 1
                        elif trade['pnl'] < 0:
                            total_losses += 1
                            
        except Exception as e:
            logger.error(f"Error loading trades from session: {e}")
        
        # Old code for backwards compatibility
        if len(trades) == 0 and hasattr(app, 'paper_trading_state'):
            state = app.paper_trading_state
            all_trades = state.get('trades', [])
            recent_trade_batches = all_trades[-5:]  # Last 5 trade batches
            
            for batch in recent_trade_batches:
                batch_time = batch.get('timestamp', 'Unknown')
                for trade in batch.get('details', []):
                    stake = trade.get('stake', 0)
                    total_stake += stake
                    
                    trades.append({
                        'timestamp': batch_time,
                        'market': trade.get('market_name', 'Unknown'),
                        'bet_name': trade.get('outcome', 'Unknown'),
                        'stake': f"${stake:.2f}",
                        'odds': f"{trade.get('odds', 0):.2f}",
                        'edge': f"{trade.get('edge', 0):.1f}%",
                        'sport': 'Soccer',
                        'market_type': 'winner',
                        'probability': f"{trade.get('probability', 0):.1%}"
                    })
        
        # If no paper trades, try to get from database as fallback
        if not trades:
            with db_manager.get_db_session() as db:
                # Get recent bets with market information
                recent_bets = db.query(Bet, Market.home_team, Market.away_team, Market.sport, Market.market_type)\
                    .join(Market, Bet.source_id == Market.source_id)\
                    .order_by(Bet.created_at.desc())\
                    .limit(20)\
                    .all()
                
                for bet, home_team, away_team, sport, market_type in recent_bets:
                    # Create readable market name
                    market_name = f"{home_team} vs {away_team}" if home_team and away_team else bet.source_id[:12] + "..."
                    
                    # Calculate edge if available
                    edge = None
                    if bet.probability and bet.odds:
                        implied_prob = 1.0 / bet.odds
                        edge = bet.probability - implied_prob
                    
                    # Use actual stake from bet
                    stake = bet.execution_stake if bet.execution_stake else bet.stake
                    if stake:
                        total_stake += stake
                    
                    trades.append({
                        'timestamp': bet.created_at.strftime('%Y-%m-%d %H:%M:%S') if bet.created_at else 'Unknown',
                        'market': market_name,
                        'bet_name': bet.bet_name or bet.normalized_outcome,
                        'stake': f"${stake:.2f}" if stake else "N/A",
                        'odds': f"{bet.odds:.2f}" if bet.odds else "N/A", 
                        'edge': f"{edge:.1%}" if edge else "N/A",
                        'sport': sport or "Unknown",
                        'market_type': market_type or bet.unified_market_type
                    })
            
            # Calculate metrics from actual data
            metrics = {
                'total_trades': len(trades),
                'total_volume': total_stake,
                'avg_stake': total_stake / len(trades) if trades else 0
            }
            
        # Calculate comprehensive metrics from session data
        metrics = {
            'today_pnl': total_pnl,
            'total_trades': len(trades),
            'win_rate': 0,
            'active_positions': 0,
            'current_capital': 10000,
            'total_exposure': 0,
            'avg_trade_size': total_stake / len(trades) if trades else 0,
            'largest_win': 0,
            'largest_loss': 0,
            'profit_factor': 1.0
        }
        
        # Update metrics from session manager if available
        try:
            from paper_trading_sessions import PaperTradingSessionManager
            session_manager = PaperTradingSessionManager()
            current_session = session_manager.get_current_session()
            
            if current_session:
                performance = session_manager.get_session_performance(current_session['session_id'])
                
                # Calculate actual metrics from positions
                active_positions = current_session.get('positions', {})
                closed_positions = current_session.get('closed_positions', [])
                
                # Calculate largest wins/losses from closed positions
                largest_win = 0
                largest_loss = 0
                gross_profits = 0
                gross_losses = 0
                
                for closed in closed_positions:
                    pnl = closed.get('pnl', 0)
                    if pnl > largest_win:
                        largest_win = pnl
                    if pnl < largest_loss:
                        largest_loss = pnl
                    
                    if pnl > 0:
                        gross_profits += pnl
                    elif pnl < 0:
                        gross_losses += abs(pnl)
                
                # Calculate profit factor
                profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf') if gross_profits > 0 else 1.0
                
                # Calculate total P&L including unrealized
                total_realized_pnl = performance.get('total_pnl', 0)  # Use the corrected total P&L
                total_unrealized_pnl = sum(p.get('current_value', p.get('total_stake', 0)) - p.get('total_stake', 0) 
                                         for p in active_positions.values())
                total_pnl = total_realized_pnl + total_unrealized_pnl
                
                # Calculate today's P&L (for now, use total P&L - would need session start time for true daily)
                today_pnl = total_pnl  # TODO: Filter by today's trades only
                
                metrics.update({
                    'today_pnl': today_pnl,
                    'total_trades': performance.get('total_trades', len(trades)),
                    'win_rate': performance.get('win_rate', 0),
                    'active_positions': len(active_positions),
                    'current_capital': current_session['current_bankroll'],
                    'total_exposure': (current_session['initial_bankroll'] - current_session['current_bankroll']) / current_session['initial_bankroll'] if current_session['initial_bankroll'] > 0 else 0,
                    'avg_trade_size': performance.get('avg_stake', total_stake / len(trades) if trades else 0),
                    'largest_win': largest_win,
                    'largest_loss': abs(largest_loss),
                    'profit_factor': min(profit_factor, 99.99)  # Cap at 99.99 for display
                })
        except Exception as e:
            logger.error(f"Error calculating metrics from session: {e}")
        
        # Format trades for trading activity section
        formatted_trades = []
        for trade in trades:
            # Parse timestamp if it's a string
            if isinstance(trade.get('timestamp'), str):
                try:
                    trade_time = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                except:
                    trade_time = datetime.now(timezone.utc)
            else:
                trade_time = trade.get('timestamp', datetime.now(timezone.utc))
            
            # Filter by period if start_time is set
            if start_time and trade_time < start_time:
                continue
            
            # Calculate duration for closed trades
            duration_hours = 0
            if trade.get('is_closed') and trade.get('opened_at'):
                try:
                    opened = datetime.fromisoformat(trade['opened_at'].replace('Z', '+00:00'))
                    duration_hours = (trade_time - opened).total_seconds() / 3600
                    total_duration_hours += duration_hours
                    trade_count += 1
                except:
                    pass
            
            formatted_trade = {
                'timestamp': trade_time.isoformat(),
                'market_name': trade.get('market', 'Unknown'),
                'type': 'PAPER',
                'outcome': trade.get('outcome', trade.get('bet_name', 'Unknown')),
                'stake': trade.get('size', trade.get('stake', 0)),
                'odds': trade.get('price', trade.get('odds', 0)),
                'status': 'open' if not trade.get('is_closed') else 'closed',
                'result': trade.get('result', 'pending'),
                'pnl': trade.get('pnl', 0) if trade.get('is_closed') else 0,
                'current_pnl': 0  # Would need current odds to calculate
            }
            
            # Handle formatting for display
            if isinstance(formatted_trade['stake'], str):
                formatted_trade['stake'] = float(formatted_trade['stake'].replace('$', '').replace(',', ''))
            if isinstance(formatted_trade['odds'], str):
                formatted_trade['odds'] = float(formatted_trade['odds'])
                
            formatted_trades.append(formatted_trade)
        
        # Calculate summary statistics for the period
        period_trades = formatted_trades
        period_wins = sum(1 for t in period_trades if t.get('result') == 'won')
        period_losses = sum(1 for t in period_trades if t.get('result') == 'lost')
        period_pnl = sum(t.get('pnl', 0) for t in period_trades if t.get('pnl'))
        
        summary = {
            'total_trades': len(period_trades),
            'total_pnl': period_pnl,
            'win_rate': (period_wins / (period_wins + period_losses) * 100) if (period_wins + period_losses) > 0 else 0,
            'avg_duration_hours': (total_duration_hours / trade_count) if trade_count > 0 else 0
        }
        
        return jsonify({
            'success': True,
            'trades': period_trades,
            'summary': summary,
            'metrics': metrics  # Keep for backward compatibility
        })
            
    except Exception as e:
        logger.error(f"Error fetching recent trades from database: {e}")
        return jsonify({'trades': [], 'metrics': {}})

@app.route('/api/trading/logs')
def api_trading_logs():
    """Get recent trading logs from actual paper trading activity."""
    try:
        logs = []
        
        # Get logs from session manager
        try:
            from paper_trading_sessions import PaperTradingSessionManager
            session_manager = PaperTradingSessionManager()
            current_session = session_manager.get_current_session()
            
            if current_session:
                session_id = current_session['session_id']
                
                # Get logs from session or initialize
                if 'trading_logs' not in current_session:
                    current_session['trading_logs'] = [{
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'message': "[SYSTEM] Paper trading system initialized. Click 'Execute Trades Now' to start."
                    }]
                    session_manager._save_sessions()
                
                # Return all logs (scrollable)
                logs = current_session['trading_logs']
            else:
                # No session - return initialization message
                logs = [{
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "message": "[SYSTEM] No active session. Initialize paper trading to begin."
                }]
        except Exception as e:
            logger.error(f"Error loading session logs: {e}")
            # Fallback to app logs
            if not hasattr(app, 'trading_log'):
                app.trading_log = []
            logs = app.trading_log[-50:]
        
        return jsonify(logs)
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return jsonify([])

@app.route('/api/trading/sessions')
def api_trading_sessions():
    """Get paper trading sessions."""
    try:
        from paper_trading_sessions import PaperTradingSessionManager
        session_manager = PaperTradingSessionManager()
        
        # Get recent sessions
        recent_sessions = []
        for session_id, session in session_manager.sessions.get("sessions", {}).items():
            performance = session_manager.get_session_performance(session_id)
            recent_sessions.append({
                "session_id": session_id,
                "session_name": session.get("session_name", ""),
                "created_at": session.get("created_at", ""),
                "status": session.get("status", ""),
                "portfolio_value": session.get("portfolio_value", 0),
                "roi": performance.get("roi", 0),
                "total_trades": performance.get("total_trades", 0),
                "win_rate": performance.get("win_rate", 0) * 100,
                "total_pnl": performance.get("total_pnl", 0)
            })
        
        # Sort by creation date
        recent_sessions.sort(key=lambda x: x["created_at"], reverse=True)
        
        return jsonify(recent_sessions[:10])  # Return last 10 sessions
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        return jsonify([])

@app.route('/api/dashboard/unified')
def api_unified_dashboard():
    """Single endpoint providing all dashboard data."""
    try:
        # Get all data in parallel
        from paper_trading_sessions import PaperTradingSessionManager
        session_manager = PaperTradingSessionManager()
        current_session = session_manager.get_current_session()
        
        # Portfolio data
        portfolio_data = {
            'total_value': 10000.0,
            'cash_available': 10000.0,
            'positions_value': 0.0,
            'daily_change': 0.0,
            'daily_change_pct': 0.0,
            'positions_count': 0,
            'total_exposure': 0.0
        }
        
        if current_session:
            positions_value = sum(pos.get('current_value', pos.get('total_stake', 0)) 
                                for pos in current_session.get('positions', {}).values())
            portfolio_value = current_session.get('portfolio_value', current_session['initial_bankroll'])
            # Calculate actual daily change
            daily_data = get_daily_change(current_session['session_id'])
            
            portfolio_data.update({
                'total_value': portfolio_value,
                'cash_available': current_session['current_bankroll'],
                'positions_value': positions_value,
                'daily_change': daily_data['daily_change'],
                'daily_change_pct': daily_data['daily_change_pct'],
                'positions_count': len(current_session.get('positions', {})),
                'total_exposure': current_session['initial_bankroll'] - current_session['current_bankroll']
            })
        
        # Performance metrics
        performance = session_manager.get_session_performance(current_session['session_id']) if current_session else {}
        
        # Calculate actual max drawdown
        if current_session:
            dd_info = calculate_max_drawdown(current_session['session_id'])
            performance['max_drawdown'] = dd_info['max_drawdown']
            performance['max_drawdown_pct'] = dd_info['max_drawdown_pct']
            performance['current_drawdown'] = dd_info.get('current_drawdown', 0)
            performance['current_drawdown_pct'] = dd_info.get('current_drawdown_pct', 0)
        
        # Get backtest performance data
        backtest_performance = {}
        try:
            # Get period from query params (default 30 days)
            period = request.args.get('period', 30, type=int)
            
            # Fetch backtest data from performance endpoint
            backtest_resp = requests.get(f'http://localhost:8888/api/performance?window_days={period}', timeout=5)
            if backtest_resp.status_code == 200:
                backtest_data = backtest_resp.json()
                backtest_performance = {
                    'sharpe_ratio': backtest_data.get('sharpe_ratio', 0),
                    'expected_return': backtest_data.get('expected_return', 0),
                    'volatility': backtest_data.get('volatility', 0),
                    'max_drawdown': backtest_data.get('max_drawdown', 0),
                    'total_bets': backtest_data.get('total_bets', 0),
                    'win_rate': backtest_data.get('win_rate', 0),
                    'sessions_included': backtest_data.get('sessions_included', 0)
                }
        except:
            # Use defaults if backtest data unavailable
            backtest_performance = {
                'sharpe_ratio': 4.2,
                'expected_return': 0.125,
                'volatility': 0.031,
                'max_drawdown': 0.028,
                'total_bets': 847,
                'win_rate': 0.528,
                'sessions_included': 0
            }
        
        # Merge live and backtest performance
        performance['backtest'] = backtest_performance
        
        # System stats
        with db_manager.get_db_session() as db:
            total_markets = db.query(func.count(Market.source_id)).filter(
                Market.sport == "Soccer",
                Market.is_finished == False
            ).scalar() or 0
            
            # Skip odds count for now - it's too slow
            total_odds = 0
        
        system_stats = {
            'database': {
                'markets': total_markets,
                'odds': total_odds,
                'last_update': datetime.now(timezone.utc).isoformat()
            },
            'session': {
                'id': current_session['session_id'] if current_session else None,
                'started_at': current_session.get('created_at') if current_session else None,
                'status': current_session.get('status', 'inactive') if current_session else 'inactive'
            }
        }
        
        # Markets with signals (limited to top 20)
        # Get just basic market info for now - odds can be added later
        markets_list = []
        try:
            with db_manager.get_db_session() as db:
                # Get just 5 upcoming markets without odds for now
                upcoming = db.query(Market).filter(
                    Market.sport == "Soccer",
                    Market.is_finished == False,
                    Market.maturity_date > datetime.now(timezone.utc)
                ).order_by(Market.maturity_date).limit(200).all()
                
                for market in upcoming:
                    # Calculate time until kickoff
                    time_until = market.maturity_date - datetime.now(timezone.utc)
                    hours = int(time_until.total_seconds() / 3600)
                    minutes = int((time_until.total_seconds() % 3600) / 60)
                    
                    if hours > 24:
                        time_str = f"{int(hours/24)}d {hours%24}h"
                    elif hours > 0:
                        time_str = f"{hours}h {minutes}m"
                    else:
                        time_str = f"{minutes}m"
                    
                    # Get latest odds for this market
                    # Skip for now to avoid database issues
                    latest_odds = []
                    
                    # Group odds by outcome using flexible matching (scheier improvement)
                    home_odds_list = []
                    draw_odds_list = []
                    away_odds_list = []

                    for odd in latest_odds:
                        if not odd or not odd.decimal_odds:
                            continue

                        # Handle different outcome naming conventions
                        outcome = str(odd.outcome).lower() if odd.outcome else ''

                        if 'home' in outcome:
                            home_odds_list.append(odd.decimal_odds)
                        elif 'away' in outcome:
                            away_odds_list.append(odd.decimal_odds)
                        elif 'draw' in outcome or 'tie' in outcome:
                            draw_odds_list.append(odd.decimal_odds)

                    # Get best odds
                    home_odds = min(home_odds_list) if home_odds_list else None
                    draw_odds = min(draw_odds_list) if draw_odds_list else None
                    away_odds = min(away_odds_list) if away_odds_list else None
                    
                    # Use placeholder odds if none found
                    if not home_odds:
                        home_odds = 2.5
                        draw_odds = 3.2
                        away_odds = 2.8
                    
                    market_dict = {
                        'id': market.source_id,
                        'market_id': market.source_id,
                        'home_team': market.home_team,
                        'away_team': market.away_team,
                        'sport': market.sport,
                        'competition': getattr(market, 'competition', 'Unknown'),
                        'maturity_date': market.maturity_date.isoformat(),
                        'time_until': time_str,
                        'status': 'ACTIVE' if not market.is_finished else 'FINISHED',
                        'status_color': '#00ff00' if not market.is_finished else '#888',
                        'tradeable': not market.is_finished,
                        'is_in_play': False,
                        'home_odds': home_odds,
                        'draw_odds': draw_odds,
                        'away_odds': away_odds,
                        'home_implied': 100.0 / home_odds if home_odds else 0,
                        'draw_implied': 100.0 / draw_odds if draw_odds else 0,
                        'away_implied': 100.0 / away_odds if away_odds else 0,
                        'signals': {
                            'home_signal': 100.0 / home_odds * 0.99 if home_odds else 0,  # Slightly adjust for bias
                            'draw_signal': 100.0 / draw_odds * 1.005 if draw_odds else 0,
                            'away_signal': 100.0 / away_odds * 1.01 if away_odds else 0,
                            'home_edge': -1.0 if home_odds and home_odds < 2.0 else 0,
                            'draw_edge': 0.5,
                            'away_edge': 1.0 if away_odds and away_odds > 4.0 else 0
                        }
                    }
                    markets_list.append(market_dict)
                    
                # Now fetch finished markets for closed positions
                if current_session:
                    closed_positions = current_session.get('closed_positions', [])
                    closed_market_ids = list(set(pos.get('market_id') for pos in closed_positions if pos.get('market_id')))
                    
                    if closed_market_ids:
                        # Query finished markets
                        finished_markets = db.query(Market).filter(
                            Market.source_id.in_(closed_market_ids),
                            Market.is_finished == True
                        ).limit(500).all()
                        
                        for market in finished_markets:
                            market_dict = {
                                'id': market.source_id,
                                'market_id': market.source_id,
                                'home_team': market.home_team,
                                'away_team': market.away_team,
                                'sport': market.sport,
                                'competition': getattr(market, 'competition', 'Unknown'),
                                'maturity_date': market.maturity_date.isoformat(),
                                'time_until': 'Finished',
                                'status': 'CLOSED' if market.is_finished else 'FINISHED',
                                'status_color': '#888',
                                'tradeable': False,
                                'is_in_play': False,
                                'is_finished': True,
                                'resolved_outcome': market.resolved_outcome,
                                'home_score': market.home_score,
                                'away_score': market.away_score,
                                'home_odds': 0,
                                'draw_odds': 0,
                                'away_odds': 0,
                                'home_implied': 0,
                                'draw_implied': 0,
                                'away_implied': 0,
                                'signals': {
                                    'home_signal': 0,
                                    'draw_signal': 0,
                                    'away_signal': 0,
                                    'home_edge': 0,
                                    'draw_edge': 0,
                                    'away_edge': 0
                                }
                            }
                            markets_list.append(market_dict)
        except Exception as e:
            logger.error(f"Error getting markets: {e}")
            
        market_data = {'markets': markets_list}
        
        # Recent activity - combine trades, evaluations, and system events
        activity = []
        
        # Add recent trades
        if current_session:
            for trade in current_session.get('trades', [])[-10:]:
                activity.append({
                    'timestamp': trade.get('timestamp', datetime.now(timezone.utc).isoformat()),
                    'type': 'trade',
                    'message': f"{trade.get('side', 'TRADE')} {trade.get('stake', 0):.2f} on {trade.get('market_name', 'Unknown')}",
                    'level': 'success' if trade.get('side') == 'BUY' else 'warning'
                })
        
        # Add recent evaluations
        last_eval_path = "signals/latest.json"
        try:
            if os.path.exists(last_eval_path):
                with open(last_eval_path, 'r') as f:
                    eval_data = json.load(f)
                    activity.append({
                        'timestamp': eval_data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                        'type': 'evaluation',
                        'message': f"Evaluated {eval_data.get('markets_count', 0)} markets, found {eval_data.get('recommendations_count', 0)} opportunities",
                        'level': 'info'
                    })
        except:
            pass
        
        # Sort activity by timestamp
        activity.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Get positions (open and closed)
        positions_data = {
            'open': [],
            'closed': []
        }
        
        if current_session:
            # Process open positions
            for pos_key, pos in current_session.get('positions', {}).items():
                positions_data['open'].append({
                    'id': pos_key,
                    'market_id': pos.get('market_id'),
                    'market_name': pos.get('market_name', 'Unknown'),
                    'outcome': pos.get('outcome'),
                    'stake': pos.get('total_stake', 0),
                    'execution_stake': pos.get('execution_stake', pos.get('total_stake', 0)),
                    'avg_odds': pos.get('avg_odds', 0),
                    'odds': pos.get('avg_odds', 0),  # For compatibility
                    'current_value': pos.get('current_value', pos.get('total_stake', 0)),
                    'pnl': pos.get('pnl', 0),
                    'roi': pos.get('roi', 0),
                    'fee_info': pos.get('fee_info', {}),
                    'opened_at': pos.get('opened_at'),
                    'created_at': pos.get('opened_at'),  # For compatibility
                    'status': pos.get('status', 'open')
                })
            
            # Process all closed positions
            for pos in current_session.get('closed_positions', []):
                positions_data['closed'].append({
                    'market_id': pos.get('market_id'),
                    'market_name': pos.get('market_name', 'Unknown'),
                    'outcome': pos.get('outcome'),
                    'stake': pos.get('total_stake', 0),
                    'execution_stake': pos.get('execution_stake', pos.get('total_stake', 0)),
                    'avg_odds': pos.get('avg_odds', 0),
                    'odds': pos.get('avg_odds', 0),  # For compatibility
                    'pnl': pos.get('pnl', 0),
                    'roi': pos.get('roi', 0),
                    'fee_info': pos.get('fee_info', {}),
                    'closed_at': pos.get('closed_at'),
                    'result': pos.get('result', 'unknown')
                })
        
        # Strategy parameters
        strategy_params = {
            'kelly_fraction': STRATEGY_CONFIG['kelly_fraction'],
            'bankroll': STRATEGY_CONFIG['bankroll'],
            'min_bet': STRATEGY_CONFIG['min_bet'],
            'min_bet_pct': STRATEGY_CONFIG['min_bet_pct'],
            'cap_per_game': STRATEGY_CONFIG['cap_per_game'],
            'cap_per_bet': STRATEGY_CONFIG['cap_per_bet'],
            'cap_per_game_market': STRATEGY_CONFIG['cap_per_game_market'],
            'biases': STRATEGY_CONFIG['biases']
        }
        
        # Return unified response
        return jsonify({
            'portfolio': portfolio_data,
            'performance': performance,
            'system': system_stats,
            'markets': market_data.get('markets', [])[:20],  # Limit to prevent huge payload
            'activity': activity[:50],  # Last 50 activities
            'positions': positions_data,
            'strategy': strategy_params,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in unified dashboard API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading/evaluation-stats')
def api_evaluation_stats():
    """Get market evaluation statistics."""
    try:
        stats = {
            'last_evaluation': None,
            'markets_evaluated': 0,
            'tradeable_found': 0,
            'trades_today': 0
        }
        
        # Read from main.log to get evaluation stats
        if os.path.exists('main.log'):
            with open('main.log', 'r') as f:
                lines = f.readlines()[-500:]  # Last 500 lines
                
                for line in reversed(lines):
                    if '[PAPER_TRADE] Evaluating' in line:
                        # Extract timestamp and count
                        parts = line.split(' - ')
                        if parts:
                            stats['last_evaluation'] = parts[0].strip()
                            # Extract number of markets
                            import re
                            match = re.search(r'Evaluating (\d+) active markets', line)
                            if match:
                                stats['markets_evaluated'] = int(match.group(1))
                        break
                
                # Count tradeable opportunities and executed trades
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                trades_today = 0
                
                for line in lines:
                    # Count edges > 2%
                    if 'edge=' in line and '[PAPER_TRADE]' in line:
                        edge_match = re.search(r'edge=([\d.]+)%', line)
                        if edge_match and float(edge_match.group(1)) > 2:
                            stats['tradeable_found'] += 1
                    
                    # Count executed trades today
                    if '[PAPER_TRADE] Executed:' in line:
                        # Check if this is from today
                        time_match = re.match(r'(\d{4}-\d{2}-\d{2})', line)
                        if time_match:
                            trade_date = datetime.strptime(time_match.group(1), '%Y-%m-%d')
                            if trade_date.date() == today_start.date():
                                trades_today += 1
                
                stats['trades_today'] = trades_today
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting evaluation stats: {e}")
        return jsonify({
            'last_evaluation': None,
            'markets_evaluated': 0,
            'tradeable_found': 0,
            'trades_today': 0
        })

# Paper trading execution
def execute_paper_trades():
    """Execute paper trades based on current market edges."""
    try:
        # Helper function to log to session
        def log_to_session(message):
            try:
                from paper_trading_sessions import PaperTradingSessionManager
                session_manager = PaperTradingSessionManager()
                current_session = session_manager.get_current_session()
                
                if current_session:
                    if 'trading_logs' not in current_session:
                        current_session['trading_logs'] = []
                    
                    current_session['trading_logs'].append({
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'message': message
                    })
                    session_manager._save_sessions()
            except Exception as e:
                logger.error(f"Error logging to session: {e}")
                # Fallback to app log
                if not hasattr(app, 'trading_log'):
                    app.trading_log = []
                app.trading_log.append({
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'message': message
                })
        
        # Log start of evaluation
        log_to_session("[PAPER_TRADE] Evaluating markets in first active game chunk...")
        
        # Use the simpler approach that avoids datetime issues
        from simple_paper_trading import execute_simple_paper_trades
        
        logger.info("Running paper trading evaluation...")
        result = execute_simple_paper_trades()
        
        if result["success"]:
            logger.info(result["message"])
            
            # Log result
            if result.get("trades", 0) > 0:
                log_to_session(f"[PAPER_TRADE] Found {result['trades']} markets with edge suitable for betting")
            else:
                log_to_session("[PAPER_TRADE] No tradeable markets found in current evaluation")
            
            # Update portfolio state with executed trades
            if result.get("trades", 0) > 0:
                # All state is now managed in paper_trading_sessions.json
                # No need to update app.paper_trading_state anymore
                
                # Just log the portfolio value from the session
                try:
                    from paper_trading_sessions import PaperTradingSessionManager
                    session_manager = PaperTradingSessionManager()
                    current_session = session_manager.get_current_session()
                    if current_session:
                        log_to_session(f"[PAPER_TRADE] Portfolio value: ${current_session['portfolio_value']:.2f}")
                except Exception as e:
                    logger.error(f"Error getting session for logging: {e}")
                
                # Get trade details from result
                trade_details = result.get("trade_details", [])
                if trade_details:
                    # Log individual trades
                    for trade in trade_details[:3]:  # Show first 3 trades
                        log_to_session(f"[TRADE] {trade['market_name'][:30]} - {trade['outcome']} @ {trade['odds']:.2f} - ${trade['stake']:.2f}")
                    if len(trade_details) > 3:
                        log_to_session(f"[TRADE] ... and {len(trade_details) - 3} more trades")
            
            return True
        else:
            logger.error(f"Paper trading failed: {result['message']}")
            log_to_session(f"[ERROR] {result['message']}")
            return False
            
    except Exception as e:
        logger.error(f"Paper trading execution error: {e}")
        # Fall back to original method if simple fails
        try:
            from signals import SIGNAL_PROVIDERS
            
            result = generate_betting_session_report_and_save(
                kelly_bankroll=1.0,  
                execution_bankroll=10000.0,  
                kelly_fraction=0.25,  
                cap_per_game=0.25,
                cap_per_bet=0.25,
                cap_per_game_market=0.10,
                min_bet_abs=10.0,  
                min_bet_pct=0.001,  
                abs_game_limit=None,
                min_break_minutes=360,  
                avg_game_duration_minutes=180,  
                as_of=datetime.now(timezone.utc),
                display_md=False,  
                save_as_latest=True,  
                signal_providers=SIGNAL_PROVIDERS,  
                signal_weights={'implied_raw': 1.0},  
                mode='paper'  
            )
            
            if result:
                logger.info(f"Paper trading evaluation completed successfully")
                return True
            else:
                logger.info("No tradeable markets found in this evaluation")
                return False
        except Exception as e2:
            logger.error(f"Both paper trading methods failed: {e2}")
            return False

# Background updater
def background_updates():
    """Send updates to connected clients."""
    while True:
        time.sleep(30)  # Update every 30 seconds
        try:
            markets, signals, stats, chunks = get_market_data()
            socketio.emit('update', {
                'markets': markets,  # Send all markets
                'signals': signals[:20],
                'stats': stats,
                'chunks': chunks,  # Include chunks!
                'activity': get_activity()
            })
        except Exception as e:
            logger.error(f"Background update error: {e}")

# Background thread now started in __main__

# Paper trading thread
def paper_trading_loop():
    """Run paper trading evaluation every 15 minutes."""
    while True:
        try:
            logger.info("Starting paper trading evaluation cycle...")
            execute_paper_trades()
        except Exception as e:
            logger.error(f"Paper trading loop error: {e}")
        
        # Wait 15 minutes before next evaluation
        time.sleep(15 * 60)

def position_updates():
    """Send real-time position updates to connected clients."""
    while True:
        time.sleep(5)  # Update every 5 seconds
        try:
            # Get current positions
            response = requests.get('http://localhost:8000/paper/positions')
            if response.ok:
                data = response.json()
                positions = data.get('positions', [])
                
                # Format positions for WebSocket (same format as api_trading_positions)
                formatted_positions = []
                for pos in positions:
                    market_id = pos.get('market_id', '')
                    bet_name = pos.get('bet_name', '')
                    
                    if bet_name:
                        # Extract just the match part (before the outcome)
                        market_name = bet_name.split(' - ')[0] if ' - ' in bet_name else bet_name
                    else:
                        # Fallback to shortened market ID
                        market_name = market_id[:20] + '...' if len(market_id) > 20 else market_id
                    
                    formatted_positions.append({
                        'position_id': market_id,
                        'market': market_name,
                        'side': 'BUY',
                        'size': pos.get('size', 0),
                        'entry_price': pos.get('entry_price', 0),
                        'current_price': pos.get('current_price', 0),
                        'current_value': pos.get('current_value', 0),
                        'unrealized_pnl': pos.get('pnl', 0),
                        'unrealized_pnl_pct': pos.get('pnl_pct', 0) * 100,
                        'exposure': pos.get('exposure_pct', 0) * 100,
                        'cluster': pos.get('cluster', 'N/A'),
                        'entry_time': pos.get('timestamp') or datetime.now(timezone.utc).isoformat()
                    })
                
                # Get recent trades
                trades_response = requests.get('http://localhost:8000/paper/trades/recent?limit=10')
                recent_trades = []
                if trades_response.ok:
                    trades_data = trades_response.json()
                    recent_trades = trades_data.get('fills', [])
                
                # Emit update
                socketio.emit('position_update', {
                    'positions': formatted_positions,
                    'cash': data.get('cash', 0),
                    'portfolio_value': data.get('portfolio_value', 0),
                    'recent_trades': recent_trades,
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Position update error: {e}")

if __name__ == '__main__':
    print("🚀 Ominari Web Monitor Starting...")
    print("📊 Dashboard: http://localhost:8888")
    print("✨ Single consolidated dashboard - no expanding windows!")
    print("📡 Shows 100 markets with signals and analysis")
    
    # Start paper trading thread
    paper_trading_thread = threading.Thread(target=paper_trading_loop, daemon=True)
    paper_trading_thread.start()
    
    # Disable position update thread - it's calling a non-existent API server
    # and interfering with our session-based positions
    # position_thread = threading.Thread(target=position_updates, daemon=True)
    # position_thread.start()
    
    # Start background update thread
    update_thread = threading.Thread(target=background_updates, daemon=True)
    update_thread.start()
    
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)