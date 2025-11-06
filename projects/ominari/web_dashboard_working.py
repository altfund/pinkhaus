#!/usr/bin/env python3
"""
Working Ominari Dashboard with embedded HTML
"""

import os
import asyncio
import logging
from datetime import datetime, timezone
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit

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

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ominari-blockchain-trading-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Dashboard HTML embedded directly
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Ominari Blockchain Trading Dashboard</title>
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
        
        .market-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 10px;
            transition: all 0.3s;
            position: relative;
        }
        
        .market-card.blockchain-connected {
            border-left: 3px solid #00ff00;
        }
        
        .market-card.has-edge {
            background: #1a2a1a;
            border-color: #00ff00;
        }
        
        .market-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
        }
        
        .match-teams {
            font-size: 16px;
        }
        
        .match-info {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        
        .sport-badge {
            background: #333;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
        }
        
        .blockchain-badge {
            background: #003300;
            color: #00ff00;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
        }
        
        .match-details {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #888;
            margin-bottom: 10px;
        }
        
        .market-rows {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        
        .market-row {
            display: grid;
            grid-template-columns: 2fr 2fr 1fr auto;
            align-items: center;
            gap: 10px;
            padding: 5px 8px;
            background: #222;
            border-radius: 4px;
            font-size: 13px;
        }
        
        .market-position {
            font-weight: bold;
        }
        
        .market-odds {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .odds-value {
            color: #00ff00;
            font-weight: bold;
            font-size: 14px;
        }
        
        .implied-prob {
            color: #888;
            font-size: 11px;
        }
        
        .market-edge {
            text-align: right;
            font-weight: bold;
            font-size: 12px;
        }
        
        .market-edge.positive {
            color: #00ff00;
        }
        
        .market-edge.negative {
            color: #ff4444;
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
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🔗 Ominari Blockchain Trading</h1>
            <div>Enhanced Dashboard with Real Data</div>
        </div>
        <div id="connection-status">⚡ Connecting...</div>
    </div>
    
    <div class="main-container">
        <div class="markets-section">
            <h2>📈 Live Markets</h2>
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
                        <div class="stat-value" id="blockchain-markets">0</div>
                        <div class="stat-label">Blockchain Connected</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="bankroll">$0</div>
                        <div class="stat-label">Bankroll</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="active-positions">0</div>
                        <div class="stat-label">Active Positions</div>
                    </div>
                </div>
            </div>
            
            <div>
                <h3>🎮 Controls</h3>
                <button id="refresh-btn" class="button" style="width: 100%;">Refresh Data</button>
            </div>
            
            <div>
                <h3>📝 Activity Log</h3>
                <div class="log-section" id="activity-log">
                    <div class="log-entry">System initialized</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        socket.on('connect', function() {
            document.getElementById('connection-status').textContent = '✅ Connected';
            addLog('Connected to server');
            socket.emit('request_dashboard_data');
        });
        
        socket.on('disconnect', function() {
            document.getElementById('connection-status').textContent = '❌ Disconnected';
            addLog('Disconnected from server');
        });
        
        socket.on('dashboard_update', function(data) {
            console.log('Dashboard update received:', data);
            updateDashboard(data);
        });
        
        document.getElementById('refresh-btn').onclick = function() {
            socket.emit('request_dashboard_data');
            addLog('Refreshing data...');
        };
        
        function updateDashboard(data) {
            if (data.error) {
                addLog('Error: ' + data.error);
                return;
            }
            
            // Update stats
            document.getElementById('total-markets').textContent = data.stats?.total_markets || 0;
            document.getElementById('blockchain-markets').textContent = data.stats?.blockchain_connected || 0;
            document.getElementById('bankroll').textContent = '$' + (data.trading_status?.bankroll || 0).toLocaleString();
            document.getElementById('active-positions').textContent = data.trading_status?.active_positions || 0;
            
            // Update markets
            if (data.markets && data.markets.length > 0) {
                updateMarkets(data.markets);
                addLog(`Loaded ${data.markets.length} markets`);
            }
        }
        
        function updateMarkets(markets) {
            const container = document.getElementById('markets-container');
            
            // Group markets by game
            const marketsByGame = {};
            markets.forEach(market => {
                const gameKey = `${market.home_team}-${market.away_team}-${market.sport}`;
                if (!marketsByGame[gameKey]) {
                    marketsByGame[gameKey] = {
                        home_team: market.home_team,
                        away_team: market.away_team,
                        sport: market.sport,
                        league: market.league,
                        maturity_date: market.maturity_date,
                        markets: []
                    };
                }
                marketsByGame[gameKey].markets.push(market);
            });
            
            // Display grouped markets
            container.innerHTML = Object.values(marketsByGame).slice(0, 10).map(game => {
                const hasBlockchain = game.markets.some(m => m.blockchain_connected);
                const hasEdge = game.markets.some(m => m.has_edge);
                
                const marketRows = game.markets.map(market => {
                    const oddsDisplay = market.odds ? market.odds.toFixed(3) : 'N/A';
                    const impliedProb = market.implied_prob ? (market.implied_prob * 100).toFixed(1) : 'N/A';
                    const edgeClass = market.edge > 0 ? 'positive' : market.edge < 0 ? 'negative' : '';
                    
                    return `
                        <div class="market-row">
                            <div class="market-position">${market.position}</div>
                            <div class="market-odds">
                                <span class="odds-value">${oddsDisplay}</span>
                                <span class="implied-prob">${impliedProb}%</span>
                            </div>
                            ${market.edge !== undefined && market.edge !== 0 ? 
                                `<div class="market-edge ${edgeClass}">${market.edge > 0 ? '+' : ''}${(market.edge * 100).toFixed(2)}%</div>` : 
                                '<div class="market-edge">-</div>'
                            }
                            ${market.blockchain_connected ? '🔗' : ''}
                        </div>
                    `;
                }).join('');
                
                return `
                    <div class="market-card ${hasBlockchain ? 'blockchain-connected' : ''} ${hasEdge ? 'has-edge' : ''}">
                        <div class="market-header">
                            <div class="match-teams">
                                <strong>${game.home_team}</strong> vs <strong>${game.away_team}</strong>
                            </div>
                            <div class="match-info">
                                <span class="sport-badge">${game.sport}</span>
                                ${hasBlockchain ? '<span class="blockchain-badge">🔗 Blockchain</span>' : ''}
                            </div>
                        </div>
                        <div class="match-details">
                            <div class="league-info">${game.league || 'Unknown League'}</div>
                        </div>
                        <div class="market-rows">
                            ${marketRows}
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        function addLog(message) {
            const log = document.getElementById('activity-log');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
        }
        
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

# Components
components_loaded = False
session_manager = None
unified_fetcher = None

try:
    from paper_trading_postgres_integrated import PaperTradingSessionManager
    session_manager = PaperTradingSessionManager()
    
    from unified_data_fetcher import UnifiedDataFetcher
    unified_fetcher = UnifiedDataFetcher(blockchain_first=True)
    
    components_loaded = True
    logger.info("✅ Components loaded")
except Exception as e:
    logger.error(f"Failed to load components: {e}")

async def get_dashboard_data():
    """Get dashboard data"""
    # Get markets
    markets = []
    if unified_fetcher:
        try:
            raw_markets = await unified_fetcher.fetch_all_markets()
            trading_markets = unified_fetcher.format_for_trading(raw_markets)
            markets = trading_markets[:50]  # Limit to 50
            logger.info(f"Fetched {len(trading_markets)} markets")
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
    
    # Format markets for display
    formatted_markets = []
    for market in markets:
        try:
            formatted_markets.append({
                'match_id': market.get('match_id', ''),
                'home_team': market.get('home_team', ''),
                'away_team': market.get('away_team', ''),
                'sport': market.get('sport', ''),
                'league': market.get('league', ''),
                'maturity_date': str(market.get('maturity_date', '')),
                'odds': float(market.get('odds', 0)),
                'position': market.get('position', market.get('outcome', '')),
                'blockchain_connected': bool(market.get('blockchain_connected', False) or market.get('blockchain_address')),
                'edge': float(market.get('edge', 0)),
                'implied_prob': float(market.get('odds', 0)) and 1/float(market.get('odds', 0)) or 0,
                'has_edge': False
            })
        except Exception as e:
            logger.error(f"Error formatting market: {e}")
    
    # Get trading status
    trading_status = {'status': 'Active', 'bankroll': 10000, 'active_positions': 0}
    if session_manager:
        try:
            session_id = session_manager.get_current_session()
            if session_id:
                session = session_manager.get_session(session_id)
                if session:
                    trading_status = {
                        'status': 'Active',
                        'session_id': session_id,
                        'bankroll': float(session.get('current_bankroll', 0)),
                        'active_positions': int(session.get('open_positions', 0))
                    }
        except Exception as e:
            logger.error(f"Error getting session: {e}")
    
    blockchain_connected = sum(1 for m in formatted_markets if m.get('blockchain_connected'))
    
    return {
        'markets': formatted_markets,
        'trading_status': trading_status,
        'stats': {
            'total_markets': len(formatted_markets),
            'blockchain_connected': blockchain_connected,
            'blockchain_ratio': (blockchain_connected / len(formatted_markets) * 100) if formatted_markets else 0
        }
    }

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('connected', {'status': 'ok'})

@socketio.on('request_dashboard_data')
def handle_request():
    logger.info('Dashboard data requested')
    asyncio.run(send_update())

async def send_update():
    try:
        data = await get_dashboard_data()
        socketio.emit('dashboard_update', data)
        logger.info(f"Sent {len(data.get('markets', []))} markets")
    except Exception as e:
        logger.error(f"Update error: {e}")
        socketio.emit('dashboard_update', {'error': str(e)})

if __name__ == '__main__':
    logger.info("Starting Working Dashboard on port 8888...")
    logger.info(f"Components: {components_loaded}")
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)