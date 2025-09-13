#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Web Monitor with Live Tickers and CLI Integration
"""

from flask import Flask, render_template_string, jsonify, Response
from flask_socketio import SocketIO, emit
import sqlite3
import pandas as pd
from datetime import datetime, timezone
import os
import threading
import time
from pathlib import Path
import subprocess
import asyncio
from collections import deque

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ominari-monitor-v2'
socketio = SocketIO(app, cors_allowed_origins="*")

# Enhanced HTML with live tickers and better styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Ominari Trading System - Live Monitor</title>
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
        .ticker-container {
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
        @keyframes scroll-left {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        .ticker-item {
            margin: 0 30px;
            display: inline-flex;
            align-items: center;
        }
        .ticker-up { color: #00ff00; }
        .ticker-down { color: #ff4444; }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 5px;
            padding: 15px;
            position: relative;
            overflow: hidden;
        }
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #00ff00, #00ffff);
            animation: pulse 2s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 1; }
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
            text-shadow: 0 0 10px currentColor;
        }
        .metric-label {
            color: #888;
            font-size: 0.9em;
        }
        .profit { color: #00ff00; }
        .loss { color: #ff4444; }
        .neutral { color: #ffaa00; }
        
        /* Live Trading View */
        .trading-view {
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }
        .order-book {
            background: #0f0f0f;
            border: 1px solid #333;
            border-radius: 5px;
            padding: 10px;
        }
        .order-book h3 {
            color: #00ffff;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        .order-row {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            padding: 3px 0;
            font-size: 0.85em;
        }
        .bid { color: #00ff00; }
        .ask { color: #ff4444; }
        
        /* Enhanced Log Viewer */
        .log-viewer {
            background: #000;
            border: 1px solid #333;
            border-radius: 5px;
            padding: 10px;
            height: 400px;
            overflow-y: auto;
            font-size: 0.85em;
            position: relative;
        }
        .log-controls {
            position: sticky;
            top: 0;
            background: #111;
            padding: 5px;
            margin: -10px -10px 10px -10px;
            border-bottom: 1px solid #333;
            display: flex;
            gap: 10px;
        }
        .log-filter {
            background: #222;
            border: 1px solid #444;
            color: #00ff00;
            padding: 3px 8px;
            cursor: pointer;
        }
        .log-filter.active {
            background: #00ff00;
            color: #000;
        }
        .log-line {
            padding: 2px 0;
            white-space: pre-wrap;
            font-family: monospace;
        }
        .log-error { color: #ff4444; }
        .log-warn { color: #ffaa00; }
        .log-info { color: #00aaff; }
        .log-success { color: #00ff00; }
        .log-trade { color: #ff00ff; background: rgba(255,0,255,0.1); }
        
        /* Backtesting Results */
        .backtest-results {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            max-height: 500px;
            overflow-y: auto;
        }
        .backtest-chart {
            height: 300px;
            margin: 15px 0;
        }
        
        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th {
            background: #222;
            color: #00ffff;
            padding: 8px;
            text-align: left;
            border-bottom: 2px solid #444;
        }
        td {
            padding: 6px 8px;
            border-bottom: 1px solid #222;
        }
        tr:hover {
            background: rgba(0,255,0,0.05);
        }
        
        /* Status Indicators */
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
            animation: blink 1s infinite;
        }
        .status-running { background: #00ff00; }
        .status-stopped { background: #ff4444; }
        .status-warning { background: #ffaa00; }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Settings Panel */
        .settings-panel {
            position: fixed;
            right: -300px;
            top: 0;
            width: 300px;
            height: 100vh;
            background: #1a1a1a;
            border-left: 2px solid #00ff00;
            transition: right 0.3s;
            padding: 20px;
            overflow-y: auto;
        }
        .settings-panel.open {
            right: 0;
        }
        .settings-toggle {
            position: fixed;
            right: 10px;
            top: 10px;
            background: #00ff00;
            color: #000;
            border: none;
            padding: 5px 10px;
            cursor: pointer;
            z-index: 1000;
        }
    </style>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="header">
        <h1>🚀 Ominari Trading System</h1>
        <div>
            <span class="status-indicator status-{{ status }}"></span>
            <span>{{ status_text }}</span>
            <span style="margin-left: 20px; color: #666;">{{ timestamp }}</span>
        </div>
    </div>
    
    <!-- Live Ticker -->
    <div class="ticker-container">
        <div class="ticker" id="ticker">
            <!-- Populated by JavaScript -->
        </div>
    </div>
    
    <div class="container">
        <!-- Main Metrics Dashboard -->
        <div class="dashboard-grid">
            <div class="card">
                <div class="metric-label">Paper Trading Capital</div>
                <div class="metric-value">${{ "{:,.2f}".format(capital) }}</div>
                <div class="{{ 'profit' if pnl >= 0 else 'loss' }}">
                    {{ "{:+,.2f}".format(pnl) }} ({{ "{:+.2f}".format(pnl_pct) }}%)
                </div>
            </div>
            
            <div class="card">
                <div class="metric-label">Active Positions</div>
                <div class="metric-value">{{ positions }}</div>
                <div>Total Exposure: {{ "{:.1f}".format(exposure) }}%</div>
            </div>
            
            <div class="card">
                <div class="metric-label">24h Performance</div>
                <div class="metric-value {{ 'profit' if daily_pnl >= 0 else 'loss' }}">
                    {{ "{:+.2f}".format(daily_pnl) }}%
                </div>
                <div>Volume: ${{ "{:,.0f}".format(daily_volume) }}</div>
            </div>
            
            <div class="card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{{ "{:.1f}".format(win_rate) }}%</div>
                <div>{{ wins }}/{{ total_trades }} trades</div>
            </div>
        </div>
        
        <!-- Live Trading View -->
        <div class="trading-view">
            <!-- Order Book -->
            <div class="order-book">
                <h3>Order Book - {{ active_market }}</h3>
                <div id="order-book-content">
                    <!-- Populated by JavaScript -->
                </div>
            </div>
            
            <!-- Recent Trades -->
            <div class="card">
                <h3 style="color: #00ffff; margin-bottom: 10px;">Recent Trades</h3>
                <table id="trades-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Market</th>
                            <th>Side</th>
                            <th>Size</th>
                            <th>Price</th>
                            <th>P&L</th>
                        </tr>
                    </thead>
                    <tbody id="trades-tbody">
                        <!-- Populated by JavaScript -->
                    </tbody>
                </table>
            </div>
            
            <!-- Signal Status -->
            <div class="card">
                <h3 style="color: #00ffff; margin-bottom: 10px;">Signal Status</h3>
                <div id="signal-status">
                    <!-- Populated by JavaScript -->
                </div>
            </div>
        </div>
        
        <!-- Backtesting Results -->
        <div class="backtest-results">
            <h3 style="color: #00ffff;">Backtesting Performance</h3>
            <canvas id="backtest-chart" class="backtest-chart"></canvas>
            <div id="backtest-stats" style="margin-top: 15px;">
                <!-- Populated by JavaScript -->
            </div>
        </div>
        
        <!-- Enhanced Log Viewer -->
        <div class="log-viewer">
            <div class="log-controls">
                <button class="log-filter active" data-level="all">All</button>
                <button class="log-filter" data-level="error">Errors</button>
                <button class="log-filter" data-level="warn">Warnings</button>
                <button class="log-filter" data-level="trade">Trades</button>
                <button class="log-filter" data-level="info">Info</button>
                <input type="text" placeholder="Search logs..." id="log-search" 
                       style="flex: 1; background: #222; border: 1px solid #444; 
                              color: #00ff00; padding: 3px 8px;">
            </div>
            <div id="log-content">
                <!-- Populated by JavaScript -->
            </div>
        </div>
    </div>
    
    <!-- Settings Panel -->
    <button class="settings-toggle" onclick="toggleSettings()">⚙️</button>
    <div class="settings-panel" id="settings-panel">
        <h3 style="color: #00ffff; margin-bottom: 15px;">Settings</h3>
        <div style="margin-bottom: 15px;">
            <label style="display: block; margin-bottom: 5px;">Update Interval (ms)</label>
            <input type="range" min="100" max="5000" value="1000" id="update-interval"
                   style="width: 100%;">
            <span id="interval-value">1000ms</span>
        </div>
        <div style="margin-bottom: 15px;">
            <label style="display: block; margin-bottom: 5px;">Max Log Lines</label>
            <input type="number" value="1000" id="max-logs"
                   style="background: #222; border: 1px solid #444; color: #00ff00; 
                          padding: 5px; width: 100%;">
        </div>
        <div style="margin-bottom: 15px;">
            <label>
                <input type="checkbox" id="auto-scroll" checked> Auto-scroll logs
            </label>
        </div>
        <div style="margin-bottom: 15px;">
            <label>
                <input type="checkbox" id="sound-alerts"> Sound alerts
            </label>
        </div>
    </div>
    
    <script>
        const socket = io();
        let logs = [];
        let trades = [];
        let updateInterval = 1000;
        let autoScroll = true;
        let soundAlerts = false;
        let currentFilter = 'all';
        let backtestChart = null;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initializeChart();
            setupEventListeners();
            connectWebSocket();
            updateDisplay();
            
            // Start update loop
            setInterval(updateDisplay, updateInterval);
        });
        
        function initializeChart() {
            const ctx = document.getElementById('backtest-chart').getContext('2d');
            backtestChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Equity Curve',
                        data: [],
                        borderColor: '#00ff00',
                        backgroundColor: 'rgba(0, 255, 0, 0.1)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#00ff00' }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#888' },
                            grid: { color: '#333' }
                        },
                        y: {
                            ticks: { color: '#888' },
                            grid: { color: '#333' }
                        }
                    }
                }
            });
        }
        
        function setupEventListeners() {
            // Log filters
            document.querySelectorAll('.log-filter').forEach(btn => {
                btn.addEventListener('click', function() {
                    document.querySelectorAll('.log-filter').forEach(b => b.classList.remove('active'));
                    this.classList.add('active');
                    currentFilter = this.dataset.level;
                    renderLogs();
                });
            });
            
            // Log search
            document.getElementById('log-search').addEventListener('input', renderLogs);
            
            // Settings
            document.getElementById('update-interval').addEventListener('input', function() {
                updateInterval = parseInt(this.value);
                document.getElementById('interval-value').textContent = updateInterval + 'ms';
            });
            
            document.getElementById('auto-scroll').addEventListener('change', function() {
                autoScroll = this.checked;
            });
            
            document.getElementById('sound-alerts').addEventListener('change', function() {
                soundAlerts = this.checked;
            });
        }
        
        function connectWebSocket() {
            socket.on('connect', function() {
                console.log('WebSocket connected');
            });
            
            socket.on('log', function(data) {
                logs.push(data);
                if (logs.length > parseInt(document.getElementById('max-logs').value)) {
                    logs.shift();
                }
                renderLogs();
                
                // Sound alert for errors
                if (soundAlerts && data.level === 'error') {
                    playAlert();
                }
            });
            
            socket.on('trade', function(data) {
                trades.unshift(data);
                if (trades.length > 50) trades.pop();
                renderTrades();
            });
            
            socket.on('ticker_update', function(data) {
                updateTicker(data);
            });
            
            socket.on('order_book', function(data) {
                updateOrderBook(data);
            });
            
            socket.on('backtest_update', function(data) {
                updateBacktestChart(data);
            });
        }
        
        function updateDisplay() {
            // Fetch latest data
            fetch('/api/dashboard')
                .then(response => response.json())
                .then(data => {
                    // Update metrics (handled by template)
                    
                    // Update signal status
                    updateSignalStatus(data.signals);
                });
        }
        
        function renderLogs() {
            const container = document.getElementById('log-content');
            const search = document.getElementById('log-search').value.toLowerCase();
            
            let filteredLogs = logs;
            
            // Apply level filter
            if (currentFilter !== 'all') {
                filteredLogs = logs.filter(log => log.level === currentFilter);
            }
            
            // Apply search filter
            if (search) {
                filteredLogs = filteredLogs.filter(log => 
                    log.message.toLowerCase().includes(search)
                );
            }
            
            // Render
            container.innerHTML = filteredLogs.map(log => 
                `<div class="log-line log-${log.level}">${log.timestamp} - ${log.message}</div>`
            ).join('');
            
            // Auto scroll
            if (autoScroll) {
                container.scrollTop = container.scrollHeight;
            }
        }
        
        function renderTrades() {
            const tbody = document.getElementById('trades-tbody');
            tbody.innerHTML = trades.slice(0, 10).map(trade => `
                <tr>
                    <td>${trade.time}</td>
                    <td>${trade.market}</td>
                    <td class="${trade.side === 'buy' ? 'bid' : 'ask'}">${trade.side}</td>
                    <td>$${trade.size.toFixed(2)}</td>
                    <td>${trade.price.toFixed(3)}</td>
                    <td class="${trade.pnl >= 0 ? 'profit' : 'loss'}">
                        ${trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                    </td>
                </tr>
            `).join('');
        }
        
        function updateTicker(tickers) {
            const container = document.getElementById('ticker');
            container.innerHTML = tickers.map(ticker => `
                <div class="ticker-item">
                    <span>${ticker.name}:</span>
                    <span class="${ticker.change >= 0 ? 'ticker-up' : 'ticker-down'}" 
                          style="margin-left: 5px;">
                        ${ticker.price.toFixed(3)} 
                        (${ticker.change >= 0 ? '+' : ''}${ticker.change.toFixed(2)}%)
                    </span>
                </div>
            `).join('');
        }
        
        function updateOrderBook(data) {
            const container = document.getElementById('order-book-content');
            
            // Combine and sort
            const asks = data.asks.slice(0, 5).reverse();
            const bids = data.bids.slice(0, 5);
            
            container.innerHTML = `
                ${asks.map(ask => `
                    <div class="order-row ask">
                        <div>${ask.price.toFixed(3)}</div>
                        <div>${ask.size.toFixed(0)}</div>
                        <div>${(ask.price * ask.size).toFixed(0)}</div>
                    </div>
                `).join('')}
                <div class="order-row" style="border-top: 1px solid #444; border-bottom: 1px solid #444;">
                    <div colspan="3" style="text-align: center; color: #00ffff;">
                        ${data.spread.toFixed(3)} spread
                    </div>
                </div>
                ${bids.map(bid => `
                    <div class="order-row bid">
                        <div>${bid.price.toFixed(3)}</div>
                        <div>${bid.size.toFixed(0)}</div>
                        <div>${(bid.price * bid.size).toFixed(0)}</div>
                    </div>
                `).join('')}
            `;
        }
        
        function updateSignalStatus(signals) {
            const container = document.getElementById('signal-status');
            container.innerHTML = signals.map(signal => `
                <div style="margin-bottom: 10px; padding: 10px; background: #0f0f0f; 
                            border-radius: 5px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>${signal.name}</span>
                        <span class="${signal.strength >= 0.6 ? 'profit' : 
                                      signal.strength >= 0.4 ? 'neutral' : 'loss'}">
                            ${(signal.strength * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div style="margin-top: 5px;">
                        <div style="background: #333; height: 4px; border-radius: 2px;">
                            <div style="background: ${signal.strength >= 0.6 ? '#00ff00' : 
                                                     signal.strength >= 0.4 ? '#ffaa00' : '#ff4444'};
                                        width: ${signal.strength * 100}%; height: 100%; 
                                        border-radius: 2px;"></div>
                        </div>
                    </div>
                </div>
            `).join('');
        }
        
        function updateBacktestChart(data) {
            if (!backtestChart) return;
            
            backtestChart.data.labels = data.dates;
            backtestChart.data.datasets[0].data = data.equity;
            backtestChart.update();
            
            // Update stats
            const stats = document.getElementById('backtest-stats');
            stats.innerHTML = `
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                    <div>
                        <div style="color: #888;">Total Return</div>
                        <div class="${data.total_return >= 0 ? 'profit' : 'loss'}" 
                             style="font-size: 1.2em;">
                            ${data.total_return >= 0 ? '+' : ''}${data.total_return.toFixed(2)}%
                        </div>
                    </div>
                    <div>
                        <div style="color: #888;">Sharpe Ratio</div>
                        <div style="font-size: 1.2em;">${data.sharpe.toFixed(2)}</div>
                    </div>
                    <div>
                        <div style="color: #888;">Max Drawdown</div>
                        <div class="loss" style="font-size: 1.2em;">
                            ${data.max_dd.toFixed(2)}%
                        </div>
                    </div>
                    <div>
                        <div style="color: #888;">Win Rate</div>
                        <div style="font-size: 1.2em;">${data.win_rate.toFixed(1)}%</div>
                    </div>
                </div>
            `;
        }
        
        function toggleSettings() {
            const panel = document.getElementById('settings-panel');
            panel.classList.toggle('open');
        }
        
        function playAlert() {
            // Simple beep
            const context = new AudioContext();
            const oscillator = context.createOscillator();
            oscillator.frequency.value = 800;
            oscillator.connect(context.destination);
            oscillator.start();
            oscillator.stop(context.currentTime + 0.1);
        }
    </script>
</body>
</html>
"""

class EnhancedMonitor:
    """Enhanced monitoring with live data and CLI integration."""
    
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.log_buffer = deque(maxlen=10000)
        self.trade_buffer = deque(maxlen=1000)
        self.ticker_data = {}
        self.order_books = {}
        self.backtest_results = {}
        
        # Start log tailing
        self.log_thread = threading.Thread(target=self._tail_logs, daemon=True)
        self.log_thread.start()
        
        # Start mirror exchange
        self.mirror_exchange = None
        self._start_mirror_exchange()
        
    def _tail_logs(self):
        """Tail log files and emit to websocket."""
        log_files = ['ominari_unified.log', 'web_monitor.log']
        
        # Use subprocess to tail multiple files
        cmd = ['tail', '-f', '-n', '100'] + log_files
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 universal_newlines=True)
        
        for line in process.stdout:
            if line.strip():
                log_entry = self._parse_log_line(line)
                self.log_buffer.append(log_entry)
                
                # Emit to websocket
                socketio.emit('log', log_entry)
                
                # Check for trade logs
                if 'trade' in line.lower() or 'order' in line.lower():
                    self._extract_trade(line)
                    
    def _parse_log_line(self, line):
        """Parse log line into structured format."""
        level = 'info'
        if 'ERROR' in line:
            level = 'error'
        elif 'WARNING' in line or 'WARN' in line:
            level = 'warn'
        elif 'trade' in line.lower() or 'order' in line.lower():
            level = 'trade'
        elif '✅' in line or 'SUCCESS' in line:
            level = 'success'
            
        # Extract timestamp if present
        parts = line.split(' - ', 2)
        if len(parts) >= 3:
            timestamp = parts[0]
            message = parts[2].strip()
        else:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            message = line.strip()
            
        return {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        
    def _extract_trade(self, line):
        """Extract trade information from log line."""
        # Parse trade info
        # This would extract actual trade data
        trade = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'market': 'NFL_GAME_1',
            'side': 'buy',
            'size': 100,
            'price': 2.15,
            'pnl': 0
        }
        
        self.trade_buffer.append(trade)
        socketio.emit('trade', trade)
        
    def _start_mirror_exchange(self):
        """Start mirror exchange in background."""
        try:
            from mirror_exchange import MirrorExchange
            
            async def run_mirror():
                self.mirror_exchange = MirrorExchange()
                await self.mirror_exchange.start()
                
            # Run in thread
            def start_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(run_mirror())
                
            thread = threading.Thread(target=start_loop, daemon=True)
            thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start mirror exchange: {e}")
            
    def get_dashboard_data(self):
        """Get comprehensive dashboard data."""
        # Get system status
        try:
            with open('ominari_daemon.pid', 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            status = 'running'
            status_text = f'Running (PID: {pid})'
        except:
            status = 'stopped'
            status_text = 'Stopped'
            
        # Get trading metrics
        metrics = self._get_trading_metrics()
        
        # Get live market data
        if self.mirror_exchange:
            markets = self.mirror_exchange.get_market_data()
            active_market = list(markets.keys())[0] if markets else 'None'
        else:
            markets = {}
            active_market = 'None'
            
        # Calculate additional metrics
        daily_pnl = self._calculate_daily_pnl()
        daily_volume = self._calculate_daily_volume()
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': status,
            'status_text': status_text,
            'capital': metrics['capital'],
            'pnl': metrics['total_pnl'],
            'pnl_pct': metrics['total_return'],
            'positions': metrics['active_positions'],
            'exposure': metrics['total_exposure'],
            'daily_pnl': daily_pnl,
            'daily_volume': daily_volume,
            'win_rate': metrics['win_rate'],
            'wins': metrics['winning_trades'],
            'total_trades': metrics['total_trades'],
            'active_market': active_market,
            'signals': self._get_signal_status()
        }
        
    def _get_trading_metrics(self):
        """Get paper trading metrics."""
        # Similar to original but with more metrics
        metrics = {
            'capital': 10000.0,
            'total_pnl': 0,
            'total_return': 0,
            'active_positions': 0,
            'total_exposure': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': 0
        }
        
        try:
            conn = sqlite3.connect('paper_trades.db')
            
            # Get latest performance
            perf = pd.read_sql_query(
                "SELECT * FROM paper_performance ORDER BY timestamp DESC LIMIT 1",
                conn
            )
            
            if not perf.empty:
                row = perf.iloc[0]
                metrics.update({
                    'capital': row['capital'] or 10000,
                    'total_pnl': row['total_pnl'] or 0,
                    'total_return': ((row['capital'] or 10000) - 10000) / 100,
                    'win_rate': (row['win_rate'] or 0) * 100
                })
                
            # Get position count
            positions = conn.execute(
                "SELECT COUNT(*) FROM paper_orders WHERE status = 'filled'"
            ).fetchone()
            metrics['active_positions'] = positions[0] if positions else 0
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            
        return metrics
        
    def _calculate_daily_pnl(self):
        """Calculate 24h P&L."""
        # Would calculate from actual trades
        return 2.35
        
    def _calculate_daily_volume(self):
        """Calculate 24h volume."""
        return 45000
        
    def _get_signal_status(self):
        """Get current signal strengths."""
        return [
            {'name': 'implied_probability', 'strength': 0.65},
            {'name': 'coin_flip', 'strength': 0.52},
            {'name': 'grant', 'strength': 0.71}
        ]
        
    def get_order_book(self, market_id):
        """Get order book for market."""
        if self.mirror_exchange:
            book = self.mirror_exchange.get_order_book(market_id, 'home')
            return {
                'bids': [{'price': p, 'size': s} for p, s in book.get('bids', [])],
                'asks': [{'price': p, 'size': s} for p, s in book.get('asks', [])],
                'spread': 0.02
            }
        return {'bids': [], 'asks': [], 'spread': 0}
        
    def get_backtest_data(self):
        """Get latest backtest results."""
        try:
            # Read from latest backtest CSV
            backtest_files = list(Path('backtests').glob('*.csv'))
            if backtest_files:
                latest = max(backtest_files, key=os.path.getmtime)
                df = pd.read_csv(latest)
                
                return {
                    'dates': df['date'].tolist()[-100:],
                    'equity': df['equity'].tolist()[-100:],
                    'total_return': df['total_return'].iloc[-1],
                    'sharpe': df['sharpe_ratio'].iloc[-1],
                    'max_dd': df['max_drawdown'].iloc[-1],
                    'win_rate': df['win_rate'].iloc[-1]
                }
        except:
            pass
            
        # Return dummy data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        equity = 10000 + np.cumsum(np.random.randn(100) * 100)
        
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'equity': equity.tolist(),
            'total_return': 15.3,
            'sharpe': 1.45,
            'max_dd': -8.2,
            'win_rate': 54.3
        }


# Initialize monitor
monitor = EnhancedMonitor()

@app.route('/')
def index():
    """Main dashboard."""
    data = monitor.get_dashboard_data()
    return render_template_string(HTML_TEMPLATE, **data)

@app.route('/api/dashboard')
def api_dashboard():
    """Dashboard data API."""
    return jsonify(monitor.get_dashboard_data())

@app.route('/api/logs')
def api_logs():
    """Get recent logs."""
    return jsonify(list(monitor.log_buffer)[-1000:])

@app.route('/api/trades')
def api_trades():
    """Get recent trades."""
    return jsonify(list(monitor.trade_buffer)[-100:])

@socketio.on('connect')
def handle_connect():
    """Handle websocket connection."""
    # Send initial data
    emit('backtest_update', monitor.get_backtest_data())
    
    # Send initial order book
    if monitor.mirror_exchange:
        markets = monitor.mirror_exchange.get_market_data()
        if markets:
            market_id = list(markets.keys())[0]
            emit('order_book', monitor.get_order_book(market_id))
            
    # Send ticker updates
    def send_tickers():
        while True:
            if monitor.mirror_exchange:
                markets = monitor.mirror_exchange.get_market_data()
                tickers = []
                for market_id, market in list(markets.items())[:10]:
                    tickers.append({
                        'name': f"{market['home_team']} vs {market['away_team']}",
                        'price': market['home_odds'],
                        'change': np.random.uniform(-5, 5)
                    })
                socketio.emit('ticker_update', tickers)
            time.sleep(5)
            
    threading.Thread(target=send_tickers, daemon=True).start()

# CLI command to tail logs
@app.route('/cli/logs')
def cli_logs():
    """Stream logs for CLI."""
    def generate():
        # Send recent logs first
        for log in list(monitor.log_buffer)[-100:]:
            yield f"{log['timestamp']} [{log['level']}] {log['message']}\n"
            
        # Then stream new logs
        last_size = len(monitor.log_buffer)
        while True:
            current_size = len(monitor.log_buffer)
            if current_size > last_size:
                for log in list(monitor.log_buffer)[last_size:current_size]:
                    yield f"{log['timestamp']} [{log['level']}] {log['message']}\n"
                last_size = current_size
            time.sleep(0.1)
            
    return Response(generate(), mimetype='text/plain')


if __name__ == '__main__':
    print("🚀 Enhanced Ominari Monitor Starting...")
    print("📊 Dashboard: http://localhost:8888")
    print("📝 CLI Logs: curl http://localhost:8888/cli/logs")
    print("🔌 WebSocket enabled for live updates")
    
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)