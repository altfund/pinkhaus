#!/usr/bin/env python3
"""
Working dashboard with REAL odds from database
"""

import os
import asyncio
import logging
from datetime import datetime, timezone
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
from rate_limiter import setup_rate_limiting, rate_limit, ws_rate_limit, api_limiter
from cache_manager import cache, market_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

# Import dashboard configuration
from dashboard_config import ALLOWED_SPORTS, ALLOWED_LEAGUES, ALLOWED_NATIONS, DASHBOARD_SETTINGS, get_display_league
from models import Market, Odd, Bet, BettingSession
from config.bankroll_config import BankrollConfig

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ominari-blockchain-trading-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Apply rate limiting middleware
app = setup_rate_limiting(app)

# Dashboard HTML
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Ominari Blockchain Trading - Real Odds</title>
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
        
        .markets-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 2px;
        }
        
        .markets-table thead {
            position: sticky;
            top: -15px;
            background: #111;
            z-index: 10;
        }
        
        .markets-table th {
            text-align: left;
            padding: 8px;
            border-bottom: 2px solid #00ff00;
            font-size: 12px;
            text-transform: uppercase;
            color: #00ff00;
        }
        
        .markets-table tbody tr {
            background: #1a1a1a;
            transition: background 0.2s;
        }
        
        .markets-table tbody tr:hover {
            background: #252525;
        }
        
        .markets-table tbody tr.game-separator {
            border-top: 3px solid #111;
        }
        
        .markets-table td {
            padding: 8px;
            font-size: 13px;
            border-bottom: 1px solid #222;
        }
        
        .team-cell {
            font-weight: bold;
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .sport-league-cell {
            font-size: 11px;
            color: #888;
        }
        
        .odds-cell {
            text-align: center;
            font-weight: bold;
        }
        
        .odds-value {
            color: #00ff00;
            font-size: 14px;
        }
        
        .odds-value.high {  /* Odds > 5.0 */
            color: #ff8c00;  /* Dark orange - longshot */
        }
        
        .odds-value.medium-high {  /* Odds 3.0 - 5.0 */
            color: #ffaa00;  /* Orange */
        }
        
        .odds-value.medium {  /* Odds 2.0 - 3.0 */
            color: #00ff00;  /* Green - balanced */
        }
        
        .odds-value.low {  /* Odds < 2.0 */
            color: #00ccff;  /* Cyan - favorite */
        }
        
        .odds-value.very-low {  /* Odds < 1.5 */
            color: #ff00ff;  /* Magenta - heavy favorite */
        }
        
        .implied-prob {
            color: #666;
            font-size: 10px;
            display: block;
        }
        
        .edge-cell {
            text-align: center;
            font-weight: bold;
            font-size: 12px;
        }
        
        .edge-positive {
            color: #00ff00;
        }
        
        .edge-negative {
            color: #ff4444;
        }
        
        .edge-high {  /* Edge > 5% */
            color: #00ff00;
            font-weight: bold;
            text-shadow: 0 0 5px #00ff00;
        }
        
        .edge-medium {  /* Edge 2-5% */
            color: #88ff00;
        }
        
        .edge-low {  /* Edge 0-2% */
            color: #ccff00;
        }
        
        .position-active {  /* Has position */
            background: #333300;
            border: 1px solid #ffcc00;
            padding: 2px 4px;
            border-radius: 3px;
        }
        
        .position-size {
            color: #ffcc00;
            font-weight: bold;
            font-size: 11px;
        }
        
        .prob-favorite {  /* > 50% probability */
            color: #00ccff;
            font-weight: bold;
        }
        
        .prob-normal {  /* 25-50% probability */
            color: #888;
        }
        
        .prob-longshot {  /* < 25% probability */
            color: #666;
            font-style: italic;
        }
        
        .position-header {
            text-align: center;
            font-size: 11px;
        }
        
        .links-cell {
            text-align: center;
        }
        
        .market-link {
            color: #00ff00;
            text-decoration: none;
            padding: 2px 6px;
            border: 1px solid #00ff00;
            border-radius: 3px;
            font-size: 11px;
            transition: all 0.2s;
            margin: 0 2px;
        }
        
        .market-link:hover {
            background: #00ff00;
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
        
        .odds-summary {
            margin-top: 10px;
            padding: 10px;
            background: #0a0a0a;
            border-radius: 4px;
            font-size: 11px;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🔗 Ominari Blockchain Trading</h1>
            <div>Live Markets with Real Odds</div>
        </div>
        <div id="connection-status">⚡ Connecting...</div>
    </div>
    
    <div class="main-container">
        <div class="markets-section">
            <h2>📈 Live Markets - Real Odds <span style=\"font-size: 14px; color: #888;\">(${allowed_sports})</span></h2>
            <div id="markets-container">Loading markets...</div>
        </div>
        
        <div class="controls-section">
            <div>
                <h3>📊 System Stats</h3>
                <div class="stat-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="total-markets">0</div>
                        <div class="stat-label">Total Markets</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="real-odds">0</div>
                        <div class="stat-label">Real Odds</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="bankroll">$0</div>
                        <div class="stat-label">Bankroll</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="odds-range">-</div>
                        <div class="stat-label">Odds Range</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="total-positions">$0</div>
                        <div class="stat-label">Total Positions</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="active-bets">0</div>
                        <div class="stat-label">Active Bets</div>
                    </div>
                </div>
            </div>
            
            <div class="odds-summary">
                <h4>Odds Distribution</h4>
                <div id="odds-dist">Loading...</div>
            </div>
            
            <div class="position-summary">
                <h4>Position Summary</h4>
                <div id="position-dist" style="font-size: 13px;">
                    <div style="padding: 4px; margin: 2px 0;">Home: $<span id="pos-home" style="color: #ffcc00; font-weight: bold;">0</span></div>
                    <div style="padding: 4px; margin: 2px 0;">Draw: $<span id="pos-draw" style="color: #ffcc00; font-weight: bold;">0</span></div>
                    <div style="padding: 4px; margin: 2px 0;">Away: $<span id="pos-away" style="color: #ffcc00; font-weight: bold;">0</span></div>
                </div>
            </div>
            
            <div style="margin-top: 20px; padding: 10px; background: #1a1a1a; border-radius: 8px;">
                <h4 style="margin-bottom: 10px;">Color Legend</h4>
                <div style="font-size: 11px; line-height: 1.8;">
                    <div><span style="color: #00ccff;">●</span> Favorite (odds < 2.0)</div>
                    <div><span style="color: #00ff00;">●</span> Balanced (odds 2.0-3.0)</div>
                    <div><span style="color: #ffaa00;">●</span> Underdog (odds 3.0-5.0)</div>
                    <div><span style="color: #ff8c00;">●</span> Longshot (odds > 5.0)</div>
                    <div style="margin-top: 5px;"><span style="color: #00ff00; text-shadow: 0 0 5px #00ff00;">●</span> High Edge (> 5%)</div>
                    <div><span style="color: #ffcc00; font-weight: bold;">$</span> Active Position</div>
                </div>
            </div>
            
            <div>
                <h3>🎮 Controls</h3>
                <button id="refresh-btn" class="button" style="width: 100%;">Refresh Real Data</button>
            </div>
            
            <div>
                <h3>📊 Recent Paper Trades</h3>
                <div id="trades-container" style="max-height: 300px; overflow-y: auto; background: #1a1a1a; padding: 10px; border-radius: 8px;">
                    <table class="trades-table" style="width: 100%; font-size: 11px;">
                        <thead>
                            <tr style="border-bottom: 1px solid #333;">
                                <th style="text-align: left; padding: 5px;">Match</th>
                                <th style="padding: 5px;">Outcome</th>
                                <th style="padding: 5px;">Stake</th>
                                <th style="padding: 5px;">Odds</th>
                                <th style="padding: 5px;">Status</th>
                                <th style="padding: 5px;">P&L</th>
                            </tr>
                        </thead>
                        <tbody id="trades-tbody">
                            <tr><td colspan="6" style="text-align: center; padding: 20px; color: #666;">No trades yet...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div>
                <h3>📝 Activity Log</h3>
                <div class="log-section" id="activity-log">
                    <div class="log-entry">System initialized - fetching real odds</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io({
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionAttempts: 10,
            timeout: 20000
        });
        
        let reconnectAttempts = 0;
        
        socket.on('connect', function() {
            document.getElementById('connection-status').textContent = '✅ Connected';
            addLog('Connected - requesting real odds data');
            socket.emit('request_dashboard_data');
            reconnectAttempts = 0;
        });
        
        socket.on('disconnect', function() {
            document.getElementById('connection-status').textContent = '❌ Disconnected';
            addLog('Disconnected from server');
        });
        
        socket.on('reconnect_attempt', function(attemptNumber) {
            document.getElementById('connection-status').textContent = `🔄 Reconnecting... (${attemptNumber})`;
            reconnectAttempts = attemptNumber;
        });
        
        socket.on('reconnect_failed', function() {
            document.getElementById('connection-status').textContent = '❌ Connection Failed';
            addLog('Failed to reconnect after ' + reconnectAttempts + ' attempts');
        });
        
        socket.on('dashboard_update', function(data) {
            console.log('Dashboard update received:', data);
            updateDashboard(data);
        });
        
        document.getElementById('refresh-btn').onclick = function() {
            socket.emit('request_dashboard_data');
            addLog('Refreshing real odds data...');
        };
        
        function updateDashboard(data) {
            if (data.error) {
                addLog('Error: ' + data.error);
                return;
            }
            
            // Update stats
            document.getElementById('total-markets').textContent = data.stats?.total_markets || 0;
            document.getElementById('real-odds').textContent = data.stats?.real_odds_count || 0;
            document.getElementById('bankroll').textContent = '$' + (data.trading_status?.bankroll || 10000).toLocaleString();
            document.getElementById('odds-range').textContent = data.stats?.odds_range || '-';
            document.getElementById('total-positions').textContent = '$' + (data.stats?.total_positions || 0).toFixed(0);
            document.getElementById('active-bets').textContent = data.stats?.active_bets || 0;
            
            // Store positions globally for table rendering
            window.activePositions = data.positions || {};
            
            // Update odds distribution
            if (data.odds_distribution) {
                let distHtml = '';
                data.odds_distribution.slice(0, 5).forEach(item => {
                    distHtml += `<div>${item.odds.toFixed(2)}: ${item.count} markets</div>`;
                });
                document.getElementById('odds-dist').innerHTML = distHtml;
            }
            
            // Update position distribution
            if (data.stats?.positions_by_outcome) {
                document.getElementById('pos-home').textContent = data.stats.positions_by_outcome.Home.toFixed(0);
                document.getElementById('pos-draw').textContent = data.stats.positions_by_outcome.Draw.toFixed(0);
                document.getElementById('pos-away').textContent = data.stats.positions_by_outcome.Away.toFixed(0);
            }
            
            // Update markets
            if (data.markets && data.markets.length > 0) {
                updateMarkets(data.markets);
                addLog(`Loaded ${data.markets.length} markets with real odds`);
            }
        }
        
        function updateMarkets(markets) {
            const container = document.getElementById('markets-container');
            
            // Group markets by game
            const marketsByGame = {};
            markets.forEach(market => {
                const gameKey = `${market.home_team}-${market.away_team}-${market.sport}-${market.maturity_date}`;
                if (!marketsByGame[gameKey]) {
                    marketsByGame[gameKey] = {
                        home_team: market.home_team,
                        away_team: market.away_team,
                        sport: market.sport,
                        league: market.league,
                        nation: market.nation,
                        maturity_date: market.maturity_date,
                        markets: {},
                        hasLinks: market.overtime_link || market.blockchain_link
                    };
                }
                // Map outcome names
                const positionMap = {
                    'Home': 'home',
                    'Draw': 'draw', 
                    'Away': 'away'
                };
                const position = positionMap[market.position] || market.position.toLowerCase();
                
                // Store by position for easy access
                marketsByGame[gameKey].markets[position] = market;
            });
            
            // Create table HTML
            let tableHTML = `
                <table class="markets-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Match</th>
                            <th>League</th>
                            <th class="position-header">Home<br><small>1</small></th>
                            <th class="position-header">Draw<br><small>X</small></th>
                            <th class="position-header">Away<br><small>2</small></th>
                            <th class="position-header">Edge<br><small>%</small></th>
                            <th>📎</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            // Add rows for each game
            Object.values(marketsByGame).slice(0, 25).forEach((game, index) => {
                const homeOdds = game.markets['home'];
                const drawOdds = game.markets['draw'];
                const awayOdds = game.markets['away'];
                
                // Calculate edge if we have a signal system (placeholder for now)
                const edge = ''; // Would calculate from signal vs odds
                
                // Format match time
                const matchDate = new Date(game.maturity_date);
                const timeStr = formatMatchTime(matchDate);
                
                // Calculate max edge and total position
                const homeEdge = homeOdds?.edge || 0;
                const drawEdge = drawOdds?.edge || 0;
                const awayEdge = awayOdds?.edge || 0;
                const maxEdge = Math.max(homeEdge, drawEdge, awayEdge);
                
                const totalPosition = (homeOdds?.position_size || 0) + (drawOdds?.position_size || 0) + (awayOdds?.position_size || 0);
                
                tableHTML += `
                    <tr class="${index > 0 ? 'game-separator' : ''}">
                        <td class="time-cell" style="font-size: 11px; color: #888;">
                            ${timeStr}
                        </td>
                        <td class="team-cell">
                            <div style="font-size: 12px;">${game.home_team}</div>
                            <div style="font-size: 12px;">${game.away_team}</div>
                        </td>
                        <td class="sport-league-cell">
                            <div style="font-size: 11px;">${game.league}</div>
                            <div style="font-size: 10px; color: #666;">${game.nation}</div>
                        </td>
                        <td class="odds-cell">
                            ${formatOdds(homeOdds)}
                        </td>
                        <td class="odds-cell">
                            ${formatOdds(drawOdds)}
                        </td>
                        <td class="odds-cell">
                            ${formatOdds(awayOdds)}
                        </td>
                        <td class="edge-cell" style="${maxEdge > 0 ? 'color: #00ff00;' : ''}">
                            ${maxEdge > 0 ? '+' + maxEdge.toFixed(1) + '%' : '-'}
                            ${totalPosition > 0 ? `<div style="font-size: 10px; color: #ffcc00;">$${totalPosition.toFixed(0)}</div>` : ''}
                        </td>
                        <td class="links-cell">
                            ${game.hasLinks ? '🔗' : ''}
                        </td>
                    </tr>
                `;
            });
            
            tableHTML += '</tbody></table>';
            container.innerHTML = tableHTML;
        }
        
        function formatOdds(market) {
            if (!market || !market.odds) return '<span style="color:#444">-</span>';
            
            const odds = market.odds;
            const impliedProb = market.implied_prob ? market.implied_prob : (1 / odds * 100);
            const position = market.position_size || 0;
            const edge = market.edge || 0;
            
            // Determine odds color class
            let oddsClass = 'odds-value';
            if (odds < 1.5) oddsClass += ' very-low';
            else if (odds < 2) oddsClass += ' low';
            else if (odds < 3) oddsClass += ' medium';
            else if (odds < 5) oddsClass += ' medium-high';
            else oddsClass += ' high';
            
            // Determine edge color class
            let edgeClass = '';
            if (edge > 5) edgeClass = 'edge-high';
            else if (edge > 2) edgeClass = 'edge-medium';
            else if (edge > 0) edgeClass = 'edge-low';
            else if (edge < 0) edgeClass = 'edge-negative';
            
            // Determine probability color class
            let probClass = 'prob-normal';
            if (impliedProb > 50) probClass = 'prob-favorite';
            else if (impliedProb < 25) probClass = 'prob-longshot';
            
            const isDefault = odds === 2.5 || odds === 2.8 || odds === 3.0;
            const hasPosition = position > 0;
            
            return `
                <div ${hasPosition ? 'class="position-active"' : ''}>
                    <span class="${oddsClass}" ${isDefault ? 'style="opacity: 0.5;"' : ''}>
                        ${odds.toFixed(3)}
                    </span>
                    ${edge !== 0 ? `<span class="${edgeClass}" style="font-size: 10px; margin-left: 4px;">${edge > 0 ? '+' : ''}${edge.toFixed(1)}%</span>` : ''}
                </div>
                <div style="font-size: 9px; margin-top: 2px;">
                    <span class="${probClass}">${impliedProb.toFixed(1)}%</span>
                    ${position > 0 ? `<span class="position-size"> $${position.toFixed(0)}</span>` : ''}
                </div>
            `;
        }
        
        function formatEdge(edge) {
            const edgeClass = edge > 0 ? 'edge-positive' : 'edge-negative';
            return `<div style="font-size: 10px;" class="${edgeClass}">${edge > 0 ? '+' : ''}${edge.toFixed(1)}%</div>`;
        }
        
        function formatMatchTime(date) {
            const now = new Date();
            const diffMs = date - now;
            const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
            const diffDays = Math.floor(diffHours / 24);
            
            // Show relative time for near matches
            if (diffMs < 0) {
                return '<span style="color: #ff0000; font-weight: bold;">LIVE</span>';
            } else if (diffHours < 1) {
                const diffMins = Math.floor(diffMs / (1000 * 60));
                return `<span style="color: #ffaa00; font-weight: bold;">${diffMins}m</span>`;
            } else if (diffHours < 24) {
                return `<span style="color: #ffcc00;">${diffHours}h</span>`;
            } else if (diffDays < 7) {
                return `${diffDays}d`;
            } else {
                // Show date for far future
                return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            }
        }
        
        function formatEdgeSummary(maxEdge, totalPosition) {
            let html = '';
            
            if (maxEdge > 0) {
                let edgeClass = '';
                if (maxEdge > 5) edgeClass = 'edge-high';
                else if (maxEdge > 2) edgeClass = 'edge-medium';
                else edgeClass = 'edge-low';
                
                html += `<div class="${edgeClass}">+${maxEdge.toFixed(1)}%</div>`;
            } else {
                html += '<div style="color: #444;">-</div>';
            }
            
            if (totalPosition > 0) {
                html += `<div class="position-size" style="font-size: 10px;">$${totalPosition.toFixed(0)}</div>`;
            }
            
            return html;
        }
        
        function addLog(message) {
            const log = document.getElementById('activity-log');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
        }
        
        // Fetch trades periodically
        function fetchTrades() {
            fetch('/api/trades')
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.trades.length > 0) {
                        const tbody = document.getElementById('trades-tbody');
                        tbody.innerHTML = data.trades.map(trade => {
                            const statusClass = trade.status === 'won' ? 'color: #00ff00;' : 
                                              trade.status === 'lost' ? 'color: #ff0000;' : 
                                              'color: #ffcc00;';
                            const pnlClass = trade.pnl > 0 ? 'color: #00ff00;' : 'color: #ff0000;';
                            
                            return `
                                <tr style="border-bottom: 1px solid #222;">
                                    <td style="padding: 5px; font-size: 10px;">${trade.match}</td>
                                    <td style="padding: 5px; text-align: center;">${trade.outcome}</td>
                                    <td style="padding: 5px; text-align: right;">$${trade.stake.toFixed(2)}</td>
                                    <td style="padding: 5px; text-align: center;">${trade.odds.toFixed(2)}</td>
                                    <td style="padding: 5px; text-align: center; ${statusClass}">${trade.status.toUpperCase()}</td>
                                    <td style="padding: 5px; text-align: right; ${pnlClass}">
                                        ${trade.pnl > 0 ? '+' : ''}$${trade.pnl.toFixed(2)}
                                    </td>
                                </tr>
                            `;
                        }).join('');
                        
                        addLog(`Loaded ${data.trades.length} trades`);
                    }
                })
                .catch(error => {
                    console.error('Error fetching trades:', error);
                });
        }
        
        // Fetch trades every 30 seconds
        fetchTrades();
        setInterval(fetchTrades, 30000);
        
        // Request initial data
        setTimeout(() => {
            if (socket.connected) {
                socket.emit('request_dashboard_data');
            }
        }, 1000);
    </script>
</body>
</html>
"""

# Import database components directly
try:
    from database_v2 import db_manager
    from models import Market, Odd, Bet, BettingSession
    from paper_trading_postgres_integrated import PaperTradingSessionManager
    from sqlalchemy import and_, or_, not_, func
    
    session_manager = PaperTradingSessionManager()
    components_loaded = True
    logger.info("✅ Components loaded")
except Exception as e:
    logger.error(f"Failed to load components: {e}")
    components_loaded = False
    session_manager = None

def calculate_edge(odds_by_outcome):
    """Calculate edge based on normalized implied probability vs fair odds"""
    edges = {}
    
    # Get implied probabilities
    total_prob = 0
    probs = {}
    for outcome, odd in odds_by_outcome.items():
        if odd and hasattr(odd, 'normalized_implied') and odd.normalized_implied:
            probs[outcome] = odd.normalized_implied
            total_prob += odd.normalized_implied
        else:
            probs[outcome] = 0
    
    # Calculate fair probabilities (removing margin)
    if total_prob <= 0:
        return {}
    
    fair_probs = {k: v / total_prob for k, v in probs.items()}
    
    # Calculate edge for each outcome
    for outcome, odd in odds_by_outcome.items():
        if odd and hasattr(odd, 'decimal_odds') and odd.decimal_odds and outcome in fair_probs and fair_probs[outcome] > 0:
            fair_odds = 1 / fair_probs[outcome]
            actual_odds = odd.decimal_odds
            edge = ((actual_odds / fair_odds) - 1) * 100
            edges[outcome] = round(edge, 2)
        else:
            edges[outcome] = 0
    
    return edges

def get_active_positions():
    """Get active positions from current betting sessions"""
    positions = {}
    
    try:
        with db_manager.get_db_session() as db:
            # Get recent paper/live trading sessions (last 24 hours)
            from datetime import datetime, timedelta
            recent_time = datetime.utcnow() - timedelta(hours=24)
            
            active_sessions = db.query(BettingSession).filter(
                BettingSession.session_type.in_(['paper', 'live']),
                BettingSession.created_at >= recent_time
            ).order_by(BettingSession.created_at.desc()).limit(10).all()
            
            if not active_sessions:
                return positions
            
            session_ids = [s.id for s in active_sessions]
            
            # Get all bets from these sessions grouped by market and outcome
            active_bets = db.query(
                Bet.source_id,
                Bet.normalized_outcome,
                func.sum(Bet.execution_stake).label('total_stake')
            ).filter(
                Bet.session_id.in_(session_ids)
            ).group_by(
                Bet.source_id,
                Bet.normalized_outcome
            ).all()
            
            # Organize by market
            for bet in active_bets:
                market_id = bet.source_id
                if market_id not in positions:
                    positions[market_id] = {}
                positions[market_id][bet.normalized_outcome] = float(bet.total_stake or 0)
    
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
    
    return positions

async def get_real_odds_data():
    """Get markets with REAL odds from database"""
    # Try to get from cache first
    cache_key = "real_odds_data"
    cached_data = cache.get(cache_key)
    if cached_data:
        return cached_data['markets'], cached_data['odds_distribution']
    
    markets = []
    odds_distribution = []
    
    try:
        with db_manager.get_db_session() as db:
            # First get unique markets
            market_query = db.query(Market).join(Odd, Market.source_id == Odd.source_id).filter(
                # Exclude default odds
                not_(and_(
                    Odd.decimal_odds.in_([2.5, 2.8, 3.0])
                ))
            )
            
            # Apply sport filter if configured
            if ALLOWED_SPORTS and ALLOWED_SPORTS != ['']:
                market_query = market_query.filter(Market.sport.in_(ALLOWED_SPORTS))
                logger.info(f"Filtering markets for sports: {ALLOWED_SPORTS}")
            
            # Apply league filter if configured
            if ALLOWED_LEAGUES and ALLOWED_LEAGUES != ['']:
                market_query = market_query.filter(Market.league_name.in_(ALLOWED_LEAGUES))
                logger.info(f"Filtering markets for leagues: {ALLOWED_LEAGUES}")
            
            # Apply nation filter if configured
            if ALLOWED_NATIONS and ALLOWED_NATIONS != ['']:
                market_query = market_query.filter(Market.nation.in_(ALLOWED_NATIONS))
                logger.info(f"Filtering markets for nations: {ALLOWED_NATIONS}")
            
            # Exclude "International Football" league as it contains mostly American Football
            market_query = market_query.filter(Market.league_name != 'International Football')
            
            market_query = market_query.distinct().order_by(Market.maturity_date.desc()).limit(DASHBOARD_SETTINGS['markets_limit'])
            
            market_results = market_query.all()
            
            # Get all positions from paper trading sessions
            positions = get_active_positions()
            
            # Get odds for all fetched markets
            market_ids = [m.source_id for m in market_results]
            
            # Format markets with all their odds
            for market in market_results:
                # Get all odds for this market
                market_odds = db.query(Odd).filter(
                    Odd.source_id == market.source_id,
                    not_(Odd.decimal_odds.in_([2.5, 2.8, 3.0]))
                ).all()
                
                # Group odds by outcome
                odds_by_outcome = {}
                for odd in market_odds:
                    odds_by_outcome[odd.outcome] = odd
                
                # Calculate edges for this market
                edges = calculate_edge(odds_by_outcome)
                
                # Get positions for this market
                market_positions = positions.get(market.source_id, {})
                
                # Create a market entry for each outcome
                for outcome, odd in odds_by_outcome.items():
                    # Generate links
                    overtime_link = None
                    blockchain_link = None
                
                    if market.source_id:
                        # These IDs (like v2_0x323032...) are hex-encoded internal IDs, not blockchain addresses
                        # They decode to date-based IDs like "2025092094118470"
                        # So we can't generate valid blockchain explorer links
                        
                        # We can try Overtime Markets links, but the URL structure might not support these IDs
                        overtime_link = f"https://overtimemarkets.xyz/markets/{market.source_id}"
                        
                        # No blockchain link since these aren't real blockchain addresses
                        blockchain_link = None
                    
                    # Map outcomes for display
                    outcome_position = market_positions.get(outcome, 0)
                    edge_value = edges.get(outcome, 0)
                    
                    markets.append({
                        'match_id': market.source_id,
                        'home_team': market.home_team,
                        'away_team': market.away_team,
                        'sport': market.sport,
                        'league': market.league_name or 'Unknown',
                        'nation': market.nation or 'Unknown',
                        'maturity_date': str(market.maturity_date),
                        'odds': float(odd.decimal_odds),
                        'position': odd.outcome,
                        'source': odd.source or 'db',
                        'blockchain_connected': bool(getattr(market, 'blockchain_id', None)),
                        'overtime_link': overtime_link,
                        'blockchain_link': blockchain_link,
                        'edge': edge_value,
                        'position_size': outcome_position,
                        'implied_prob': float(odd.normalized_implied) if odd.normalized_implied else 0
                    })
                
            # Get odds distribution
            odds_dist_query = db.query(
                Odd.decimal_odds,
                func.count(Odd.id).label('count')
            ).join(Market, Market.source_id == Odd.source_id).filter(
                not_(Odd.decimal_odds.in_([2.5, 2.8, 3.0]))
            )
            
            # Apply filters to odds distribution
            if ALLOWED_SPORTS and ALLOWED_SPORTS != ['']:
                odds_dist_query = odds_dist_query.filter(Market.sport.in_(ALLOWED_SPORTS))
            if ALLOWED_LEAGUES and ALLOWED_LEAGUES != ['']:
                odds_dist_query = odds_dist_query.filter(Market.league_name.in_(ALLOWED_LEAGUES))
            if ALLOWED_NATIONS and ALLOWED_NATIONS != ['']:
                odds_dist_query = odds_dist_query.filter(Market.nation.in_(ALLOWED_NATIONS))
                
            odds_dist = odds_dist_query.group_by(Odd.decimal_odds).order_by(func.count(Odd.id).desc()).limit(10).all()
            odds_distribution = [{'odds': float(o[0]), 'count': o[1]} for o in odds_dist]
            
            logger.info(f"Fetched {len(markets)} markets with real odds and edges")
            
            # Cache the results
            cache.set(cache_key, {
                'markets': markets,
                'odds_distribution': odds_distribution
            }, ttl=30)  # Cache for 30 seconds
            
    except Exception as e:
        logger.error(f"Error fetching real odds: {e}")
    
    return markets, odds_distribution

async def get_dashboard_data():
    """Get dashboard data with real odds"""
    # Get real odds markets
    markets, odds_distribution = await get_real_odds_data()
    
    # Calculate odds range
    if markets:
        all_odds = [m['odds'] for m in markets if m['odds'] > 0]
        if all_odds:
            odds_range = f"{min(all_odds):.2f} - {max(all_odds):.2f}"
        else:
            odds_range = "N/A"
    else:
        odds_range = "N/A"
    
    # Count real odds
    real_odds_count = sum(1 for m in markets if m['odds'] not in [2.5, 2.8, 3.0])
    
    # Get actual positions from database
    positions = get_active_positions()
    
    # Calculate position summary
    total_positions = 0
    active_bets = 0
    position_by_outcome = {'Home': 0, 'Draw': 0, 'Away': 0}
    
    # Count positions and total exposure
    for market_id, market_positions in positions.items():
        for outcome, stake in market_positions.items():
            if stake > 0:
                total_positions += stake
                active_bets += 1
                
                # Map outcome to display category
                if outcome == '1':
                    position_by_outcome['Home'] += stake
                elif outcome == 'X':
                    position_by_outcome['Draw'] += stake
                elif outcome == '2':
                    position_by_outcome['Away'] += stake
    
    # Get trading status from bankroll config
    try:
        bankroll_config = BankrollConfig()
        current_bankroll = bankroll_config.get_current_bankroll()
        perf_stats = bankroll_config.get_performance_stats()
        
        trading_status = {
            'status': 'Active' if bankroll_config.is_trading_enabled() else 'Paused',
            'bankroll': current_bankroll,
            'pnl': perf_stats['total_pnl'],
            'roi': perf_stats['roi'],
            'win_rate': perf_stats['win_rate']
        }
    except Exception as e:
        logger.error(f"Error loading bankroll config: {e}")
        # Fallback to database lookup
        trading_status = {'status': 'Active', 'bankroll': 10000}
        
        try:
            with db_manager.get_db_session() as db:
                # Get most recent paper trading session
                from datetime import datetime, timedelta
                latest_session = db.query(BettingSession).filter(
                    BettingSession.is_paper == True
                ).order_by(BettingSession.created_at.desc()).first()
                
                if latest_session:
                    # Calculate current bankroll based on initial + P&L
                    session_bets = db.query(Bet).filter(
                        Bet.betting_session_id == latest_session.id
                    ).all()
                    
                    total_pnl = 0
                    for bet in session_bets:
                        if bet.status == 'won':
                            total_pnl += (bet.payout or 0) - bet.stake
                        elif bet.status == 'lost':
                            total_pnl -= bet.stake
                    
                    current_bankroll = float(latest_session.bankroll) + total_pnl
                    trading_status = {
                        'status': 'Active',
                        'bankroll': current_bankroll
                    }
        except:
            # Final fallback
            pass
    
    return {
        'markets': markets[:50],  # Limit to 50 for display
        'trading_status': trading_status,
        'stats': {
            'total_markets': len(markets),
            'real_odds_count': real_odds_count,
            'odds_range': odds_range,
            'total_positions': total_positions,
            'active_bets': active_bets,
            'positions_by_outcome': position_by_outcome
        },
        'odds_distribution': odds_distribution,
        'positions': positions  # Include positions data
    }

@app.route('/')
def index():
    # Pass allowed sports and leagues to template
    filter_text = ', '.join(ALLOWED_SPORTS)
    if ALLOWED_LEAGUES:
        filter_text += f" - {', '.join(ALLOWED_LEAGUES)}"
    return render_template_string(DASHBOARD_HTML.replace('${allowed_sports}', filter_text))

@app.route('/api/trades')
@rate_limit(limiter=api_limiter)
def get_trades():
    """Get recent trades"""
    try:
        with db_manager.get_db_session() as db:
            # Get recent bets
            recent_bets = db.query(Bet).join(BettingSession).filter(
                BettingSession.session_type == 'paper'
            ).order_by(Bet.created_at.desc()).limit(20).all()
            
            trades = []
            for bet in recent_bets:
                # Extract match info from bet_name if possible
                match_name = bet.bet_name or f"Market {bet.source_id[:8]}..."
                trades.append({
                    'id': bet.id,
                    'match': match_name,
                    'outcome': bet.normalized_outcome,
                    'stake': float(bet.stake) if bet.stake else 0,
                    'odds': float(bet.odds) if bet.odds else 0,
                    'status': 'pending',  # Since we don't track status in Bet model
                    'placed_at': bet.created_at.isoformat() if bet.created_at else None,
                    'pnl': 0  # Would need separate tracking for actual P&L
                })
                
            return jsonify({'trades': trades, 'success': True})
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return jsonify({'trades': [], 'error': str(e)})

@app.route('/health')
@rate_limit(limiter=api_limiter)
def health():
    """Health check endpoint"""
    try:
        # Check database connection
        with db_manager.get_db_session() as db:
            market_count = db.query(Market).count()
            db_healthy = True
    except:
        market_count = 0
        db_healthy = False
    
    # Check session manager
    try:
        session_id = session_manager.get_current_session()
        session_healthy = True
    except:
        session_healthy = False
    
    health_data = {
        'status': 'healthy' if db_healthy else 'unhealthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'services': {
            'database': {
                'status': 'up' if db_healthy else 'down',
                'markets': market_count
            },
            'session_manager': {
                'status': 'up' if session_healthy else 'down'
            },
            'websocket': {
                'status': 'up',
                'clients': len(socketio.server.manager.rooms.get('/', {}).get('', set()))
            }
        }
    }
    
    return jsonify(health_data), 200 if db_healthy else 503

@app.route('/cache-stats')
@rate_limit(limiter=api_limiter)
def cache_stats():
    """Get cache statistics"""
    return jsonify(cache.get_stats())

@app.route('/metrics')
@rate_limit(limiter=api_limiter)
def metrics():
    """Prometheus-style metrics endpoint"""
    try:
        with db_manager.get_db_session() as db:
            metrics_data = []
            
            # Market metrics
            total_markets = db.query(Market).count()
            real_odds = db.query(Odd).filter(
                ~Odd.decimal_odds.in_([2.5, 2.8, 3.0])
            ).count()
            
            metrics_data.append(f"# HELP ominari_markets_total Total number of markets")
            metrics_data.append(f"# TYPE ominari_markets_total gauge")
            metrics_data.append(f"ominari_markets_total {total_markets}")
            
            metrics_data.append(f"# HELP ominari_real_odds_total Markets with real odds")
            metrics_data.append(f"# TYPE ominari_real_odds_total gauge")
            metrics_data.append(f"ominari_real_odds_total {real_odds}")
            
            # Session metrics
            if session_manager:
                try:
                    session_id = session_manager.get_current_session()
                    if session_id:
                        session = session_manager.get_session(session_id)
                        metrics_data.append(f"# HELP ominari_bankroll_current Current bankroll")
                        metrics_data.append(f"# TYPE ominari_bankroll_current gauge")
                        metrics_data.append(f"ominari_bankroll_current {float(session.get('current_bankroll', 0))}")
                except:
                    pass
            
            return '\n'.join(metrics_data), 200, {'Content-Type': 'text/plain'}
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return "# Error generating metrics", 500

@socketio.on('connect')
@ws_rate_limit()
def handle_connect():
    logger.info('Client connected')
    emit('connected', {'status': 'ok'})

@socketio.on('request_dashboard_data')
@ws_rate_limit()
def handle_request():
    logger.info('Dashboard data requested - fetching real odds')
    asyncio.run(send_update())

async def send_update():
    try:
        data = await get_dashboard_data()
        socketio.emit('dashboard_update', data)
        logger.info(f"Sent {len(data.get('markets', []))} markets with real odds")
    except Exception as e:
        logger.error(f"Update error: {e}")
        socketio.emit('dashboard_update', {'error': str(e)})

def start_automated_trading():
    """Start the automated trading system in background"""
    import subprocess
    import os
    
    logger.info("Starting integrated trading system...")
    
    # Start blockchain sync
    env = os.environ.copy()
    env['DATABASE_URL'] = os.getenv('DATABASE_URL', 'postgresql://ominari_user:ominari_2025_secure@localhost:5999/ominari_production')
    
    try:
        # Start integrated trading system (includes real odds + paper trading)
        trading_proc = subprocess.Popen(
            [sys.executable, 'integrated_trading_system.py'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Started integrated trading system (PID: {trading_proc.pid})")
        logger.info("✅ System includes: real odds fetching + paper trading + Discord notifications")
        
        # Start performance monitor
        monitor_proc = subprocess.Popen(
            [sys.executable, 'trading_performance_monitor.py'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Started performance monitor (PID: {monitor_proc.pid})")
        
        return True
    except Exception as e:
        logger.error(f"Failed to start integrated trading: {e}")
        return False

if __name__ == '__main__':
    logger.info("Starting Real Odds Dashboard on port 8888...")
    logger.info("Rate limiting enabled: 60 req/min general, 30 req/min API, 120 req/min WebSocket")
    logger.info("Caching enabled: 30s TTL for market data")
    logger.info("Fetching markets with actual odds from database...")
    
    # Start automated trading system
    if start_automated_trading():
        logger.info("✅ Automated trading system started successfully")
        logger.info("📊 Performance monitor available at http://localhost:8889")
    else:
        logger.warning("⚠️ Automated trading system failed to start - continuing without it")
    
    # Log cache and rate limit stats periodically
    def log_stats():
        while True:
            time.sleep(60)  # Every minute
            logger.info(f"Cache stats: {cache.get_stats()}")
            logger.info(f"Rate limiter active IPs: {len(api_limiter.requests)}")
    
    import threading
    import time
    import sys
    stats_thread = threading.Thread(target=log_stats, daemon=True)
    stats_thread.start()
    
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)