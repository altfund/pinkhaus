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
            text-transform: capitalize;
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
        
        .odds-value.high {
            color: #ffaa00;
        }
        
        .odds-value.low {
            color: #00ccff;
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
            <h2>📈 Live Markets - Real Odds</h2>
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
                </div>
            </div>
            
            <div class="odds-summary">
                <h4>Odds Distribution</h4>
                <div id="odds-dist">Loading...</div>
            </div>
            
            <div>
                <h3>🎮 Controls</h3>
                <button id="refresh-btn" class="button" style="width: 100%;">Refresh Real Data</button>
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
        const socket = io();
        
        socket.on('connect', function() {
            document.getElementById('connection-status').textContent = '✅ Connected';
            addLog('Connected - requesting real odds data');
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
            document.getElementById('bankroll').textContent = '$' + (data.trading_status?.bankroll || 0).toLocaleString();
            document.getElementById('odds-range').textContent = data.stats?.odds_range || '-';
            
            // Update odds distribution
            if (data.odds_distribution) {
                let distHtml = '';
                data.odds_distribution.slice(0, 5).forEach(item => {
                    distHtml += `<div>${item.odds.toFixed(2)}: ${item.count} markets</div>`;
                });
                document.getElementById('odds-dist').innerHTML = distHtml;
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
            container.innerHTML = Object.values(marketsByGame).slice(0, 15).map(game => {
                const hasBlockchain = game.markets.some(m => m.blockchain_connected);
                
                // Sort markets by position
                const sortedMarkets = game.markets.sort((a, b) => {
                    const order = ['home', 'draw', 'away'];
                    return order.indexOf(a.position.toLowerCase()) - order.indexOf(b.position.toLowerCase());
                });
                
                const marketRows = sortedMarkets.map(market => {
                    const odds = market.odds || 0;
                    const oddsDisplay = odds > 0 ? odds.toFixed(3) : 'N/A';
                    const impliedProb = odds > 0 ? (1 / odds * 100).toFixed(1) : 'N/A';
                    
                    // Color code odds
                    let oddsClass = '';
                    if (odds > 5) oddsClass = 'high';
                    else if (odds < 2) oddsClass = 'low';
                    
                    // Check if it's a default odd
                    const isDefault = odds === 2.5 || odds === 2.8 || odds === 3.0;
                    
                    return `
                        <div class="market-row">
                            <div class="market-position">${market.position}</div>
                            <div class="market-odds">
                                <span class="odds-value ${oddsClass}" style="${isDefault ? 'opacity: 0.5;' : ''}">${oddsDisplay}</span>
                                <span class="implied-prob">${impliedProb}%</span>
                            </div>
                            <div class="market-edge">
                                ${market.source || ''}
                            </div>
                            ${market.blockchain_connected ? '🔗' : ''}
                        </div>
                    `;
                }).join('');
                
                return `
                    <div class="market-card ${hasBlockchain ? 'blockchain-connected' : ''}">
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

# Import database components directly
try:
    from database_v2 import db_manager
    from models import Market, Odd
    from paper_trading_postgres_integrated import PaperTradingSessionManager
    from sqlalchemy import and_, or_, not_, func
    
    session_manager = PaperTradingSessionManager()
    components_loaded = True
    logger.info("✅ Components loaded")
except Exception as e:
    logger.error(f"Failed to load components: {e}")
    components_loaded = False
    session_manager = None

async def get_real_odds_data():
    """Get markets with REAL odds from database"""
    markets = []
    odds_distribution = []
    
    try:
        with db_manager.get_db_session() as db:
            # Get markets with non-default odds
            query = db.query(Market, Odd).join(Odd, Market.source_id == Odd.source_id).filter(
                # Exclude default odds
                not_(and_(
                    Odd.decimal_odds.in_([2.5, 2.8, 3.0])
                ))
            ).order_by(Market.maturity_date.desc()).limit(150)
            
            results = query.all()
            
            # Get odds distribution
            odds_dist = db.query(
                Odd.decimal_odds,
                func.count(Odd.id).label('count')
            ).filter(
                not_(Odd.decimal_odds.in_([2.5, 2.8, 3.0]))
            ).group_by(Odd.decimal_odds).order_by(func.count(Odd.id).desc()).limit(10).all()
            
            odds_distribution = [{'odds': float(o[0]), 'count': o[1]} for o in odds_dist]
            
            # Format markets
            for market, odd in results:
                markets.append({
                    'match_id': market.source_id,
                    'home_team': market.home_team,
                    'away_team': market.away_team,
                    'sport': market.sport,
                    'league': getattr(market, 'league', None) or 'Unknown',
                    'maturity_date': str(market.maturity_date),
                    'odds': float(odd.decimal_odds),
                    'position': odd.outcome,
                    'source': odd.source or 'db',
                    'blockchain_connected': bool(getattr(market, 'blockchain_id', None))
                })
                
            logger.info(f"Fetched {len(markets)} markets with real odds")
            
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
    
    # Get trading status
    trading_status = {'status': 'Active', 'bankroll': 10000}
    if session_manager:
        try:
            session_id = session_manager.get_current_session()
            if session_id:
                session = session_manager.get_session(session_id)
                if session:
                    trading_status = {
                        'status': 'Active',
                        'bankroll': float(session.get('current_bankroll', 0))
                    }
        except Exception as e:
            logger.error(f"Error getting session: {e}")
    
    return {
        'markets': markets[:50],  # Limit to 50 for display
        'trading_status': trading_status,
        'stats': {
            'total_markets': len(markets),
            'real_odds_count': real_odds_count,
            'odds_range': odds_range
        },
        'odds_distribution': odds_distribution
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

if __name__ == '__main__':
    logger.info("Starting Real Odds Dashboard on port 8888...")
    logger.info("Fetching markets with actual odds from database...")
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)