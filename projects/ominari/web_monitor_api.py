#!/usr/bin/env python3
"""
API-Connected Ominari Dashboard - Direct API calls without complex imports
"""

import os
import asyncio
import logging
import aiohttp
import json
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    'host': 'localhost',
    'port': 5999,
    'user': 'ominari_user',
    'password': 'ominari_2025_secure',
    'database': 'ominari_production'
}

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ominari-blockchain-trading-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Load HTML template
try:
    with open('web_monitor_blockchain.py', 'r') as f:
        content = f.read()
        start = content.find('BLOCKCHAIN_DASHBOARD_HTML = """') + len('BLOCKCHAIN_DASHBOARD_HTML = """')
        end = content.rfind('"""')
        end = content.rfind('"""', 0, end)
        DASHBOARD_HTML = content[start:end]
        logger.info("Dashboard HTML loaded successfully")
except Exception as e:
    logger.error(f"Failed to load dashboard HTML: {e}")
    DASHBOARD_HTML = "<h1>Error loading dashboard</h1>"

# API endpoints
UNIFIED_API_BASE = "http://localhost:8000/api/v1"
GRAPH_API = "https://overtimemarketsv2-thales-subgraph.thegraph.com/subgraphs/name/thales-markets/overtime-markets-v2"

async def fetch_api_markets():
    """Fetch markets from API"""
    try:
        async with aiohttp.ClientSession() as session:
            # Try unified API first
            try:
                async with session.get(f"{UNIFIED_API_BASE}/markets", timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get('markets', [])
            except:
                logger.info("Unified API not available, using direct queries")
            
            # Direct GraphQL query
            query = {
                "query": """
                {
                  markets(
                    first: 100
                    orderBy: maturityDate
                    orderDirection: asc
                    where: {
                      maturityDate_gte: "%d"
                      isResolved: false
                      isCanceled: false
                    }
                  ) {
                    id
                    address
                    gameId
                    sportId
                    typeId
                    maturityDate
                    homeTeam
                    awayTeam
                    isResolved
                    isPaused
                    positions {
                      id
                      side
                      line
                      odd
                    }
                    childMarkets {
                      id
                      typeId
                      positions {
                        id
                        side
                        line
                        odd
                      }
                    }
                  }
                }
                """ % int(datetime.now(timezone.utc).timestamp())
            }
            
            async with session.post(
                GRAPH_API,
                json=query,
                headers={'Content-Type': 'application/json'}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return format_graph_markets(data.get('data', {}).get('markets', []))
                    
        return []
    except Exception as e:
        logger.error(f"Error fetching API markets: {e}")
        return []

def format_graph_markets(graph_markets):
    """Format GraphQL markets to our standard format"""
    formatted = []
    
    sport_map = {
        '1': 'Soccer',
        '2': 'American Football',
        '3': 'Baseball',
        '4': 'Basketball',
        '5': 'Ice Hockey'
    }
    
    for market in graph_markets:
        # Main market
        base_market = {
            'match_id': market['gameId'],
            'blockchain_address': market['address'],
            'home_team': market['homeTeam'],
            'away_team': market['awayTeam'],
            'sport': sport_map.get(market['sportId'], 'Unknown'),
            'maturity_date': datetime.fromtimestamp(int(market['maturityDate']), tz=timezone.utc),
            'blockchain_connected': True,
            'has_blockchain_data': True
        }
        
        # Process positions
        for position in market.get('positions', []):
            if position.get('odd') and float(position['odd']) > 0:
                formatted_market = base_market.copy()
                formatted_market.update({
                    'position': get_position_name(position['side']),
                    'odds': float(position['odd']),
                    'blockchain_id': position['id'],
                    'implied_prob': 1 / float(position['odd']) if float(position['odd']) > 0 else 0
                })
                formatted.append(formatted_market)
    
    return formatted

def get_position_name(side):
    """Convert side number to position name"""
    return {
        '0': 'Home Win',
        '1': 'Away Win',
        '2': 'Draw'
    }.get(str(side), f'Position {side}')

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

async def get_trading_status():
    """Get trading status from database"""
    conn = get_db_connection()
    if not conn:
        return {
            'status': 'Database offline',
            'bankroll': 0,
            'active_positions': 0
        }
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get latest session
            cur.execute("""
                SELECT session_id, initial_bankroll, current_bankroll 
                FROM paper_trading_sessions 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            session = cur.fetchone()
            
            if not session:
                return {
                    'status': 'No session',
                    'bankroll': 0,
                    'active_positions': 0
                }
            
            # Get active positions count
            cur.execute("""
                SELECT COUNT(*) as count, COALESCE(SUM(stake), 0) as total_stake
                FROM paper_trades 
                WHERE session_id = %s 
                AND status = 'pending'
            """, (session['session_id'],))
            positions = cur.fetchone()
            
            return {
                'status': 'Active',
                'session_id': session['session_id'],
                'bankroll': float(session['current_bankroll']),
                'initial_bankroll': float(session['initial_bankroll']),
                'active_positions': positions['count'],
                'positions_value': float(positions['total_stake'])
            }
    except Exception as e:
        logger.error(f"Error getting trading status: {e}")
        return {
            'status': 'Error',
            'bankroll': 0,
            'active_positions': 0
        }
    finally:
        conn.close()

async def calculate_edges(markets):
    """Simple edge calculation based on market inefficiencies"""
    # Look for arbitrage opportunities and market discrepancies
    markets_by_game = {}
    
    for market in markets:
        game_key = f"{market['match_id']}"
        if game_key not in markets_by_game:
            markets_by_game[game_key] = []
        markets_by_game[game_key].append(market)
    
    # Simple edge: if total implied probability < 100%, there's positive edge
    for game_markets in markets_by_game.values():
        total_implied = sum(m.get('implied_prob', 0) for m in game_markets)
        
        if total_implied > 0 and len(game_markets) >= 2:
            # Distribute edge proportionally
            edge_per_market = (1 - total_implied) / len(game_markets) if total_implied < 1 else 0
            
            for market in game_markets:
                market['edge'] = edge_per_market
                market['has_edge'] = edge_per_market > 0.01  # 1% minimum

async def get_dashboard_data():
    """Get complete dashboard data"""
    # Fetch markets from API
    markets = await fetch_api_markets()
    
    # Calculate simple edges
    await calculate_edges(markets)
    
    # Get trading status
    trading_status = await get_trading_status()
    
    # Format for display
    formatted_markets = []
    for market in markets[:50]:  # Limit for performance
        formatted_markets.append({
            'match_id': market.get('match_id', ''),
            'home_team': market.get('home_team', ''),
            'away_team': market.get('away_team', ''),
            'sport': market.get('sport', ''),
            'league': market.get('league', 'Blockchain'),
            'maturity_date': (
                market.get('maturity_date').isoformat() 
                if hasattr(market.get('maturity_date'), 'isoformat')
                else str(market.get('maturity_date', ''))
            ),
            'odds': market.get('odds', 0),
            'position': market.get('position', ''),
            'blockchain_connected': True,
            'blockchain_address': market.get('blockchain_address', ''),
            'edge': market.get('edge', 0),
            'implied_prob': market.get('implied_prob', 0),
            'has_edge': market.get('has_edge', False)
        })
    
    blockchain_connected = len(markets)  # All from blockchain
    
    return {
        'markets': formatted_markets,
        'trading_status': trading_status,
        'stats': {
            'total_markets': len(markets),
            'blockchain_connected': blockchain_connected,
            'blockchain_ratio': 100.0,
            'chunk_hours': 2.0
        },
        'config': {
            'testnet_mode': False,
            'simulation_mode': False,
            'network': 'Blockchain (The Graph)'
        },
        'capital_state': {
            'available_cash': trading_status.get('bankroll', 0) - trading_status.get('positions_value', 0),
            'pending_stakes': 0,
            'in_play_exposure': trading_status.get('positions_value', 0),
            'settlement_pending': 0,
            'utilization_rate': (
                trading_status.get('positions_value', 0) / trading_status.get('bankroll', 1)
                if trading_status.get('bankroll', 0) > 0 else 0
            ),
            'value_at_risk': trading_status.get('positions_value', 0),
            'expected_value': 0
        }
    }

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    asyncio.run(send_dashboard_update())

@socketio.on('request_dashboard_data')
def handle_dashboard_request():
    asyncio.run(send_dashboard_update())

async def send_dashboard_update():
    """Send dashboard update"""
    try:
        data = await get_dashboard_data()
        socketio.emit('dashboard_update', data)
        logger.info(f"Sent update with {len(data['markets'])} markets")
    except Exception as e:
        logger.error(f"Error sending update: {e}")
        socketio.emit('dashboard_update', {'error': str(e)})

# Background updates
def background_updates():
    while True:
        try:
            asyncio.run(send_dashboard_update())
        except Exception as e:
            logger.error(f"Background update error: {e}")
        
        import time
        time.sleep(30)

if __name__ == '__main__':
    logger.info("Starting API-Connected Dashboard on port 8888...")
    logger.info("Connecting to blockchain via The Graph API...")
    
    # Start background thread
    import threading
    update_thread = threading.Thread(target=background_updates, daemon=True)
    update_thread.start()
    
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)