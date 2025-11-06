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
from decimal import Decimal
from flask_socketio import SocketIO, emit
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import threading
import time
import numpy as np
import pandas as pd

# Import database and models
from database_v2 import db_manager
from models import Market

# Import paper trading
from paper_trading_engine import PaperTradingEngine
from paper_trading_postgres_integrated import PaperTradingSessionManager

# Import enhanced edge calculator
from edge_calculator import EdgeCalculator

# Import portfolio trading engine
from portfolio_trading_engine import PortfolioTradingEngine
from stop_loss_manager import StopLossManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON encoder for datetime and Decimal
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if hasattr(obj, '__float__'):  # numpy types
            return float(obj)
        return super().default(obj)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ominari-trading-system-2024'
app.json_encoder = CustomJSONEncoder
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', json=json)

# Strategy configuration
STRATEGY_CONFIG = {
    'kelly_fraction': 0.25,
    'min_bet': 10,
    'min_bet_pct': 0.001,
    'bankroll': 10000,
    'cap_per_game': 0.02,  # Reduced from 25% to 2% per game
    'cap_per_bet': 0.01,   # Reduced from 25% to 1% per bet
    'cap_per_game_market': 0.005,  # Reduced from 10% to 0.5%
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

# Initialize paper trading and edge calculator
session_manager = PaperTradingSessionManager()
paper_engine = PaperTradingEngine()
edge_calculator = EdgeCalculator()

# Initialize portfolio trading engine
portfolio_engine = PortfolioTradingEngine(session_manager, edge_calculator, STRATEGY_CONFIG)

# Initialize stop loss manager
stop_loss_manager = StopLossManager(session_manager)
# Configure stop loss parameters
stop_loss_manager.set_stop_loss_config({
    'drawdown_pct': 10,  # Stop at 10% drawdown
    'time_window_minutes': 30,  # Monitor 30 min windows
    'max_daily_loss_pct': 15,  # Max 15% daily loss
    'consecutive_losses': 5,  # Stop after 5 losses
})

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

def get_market_data(sport_filter='All'):
    """Get market data directly from PostgreSQL with gap-based batching"""
    markets = []
    signals = []
    chunks = []
    
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get upcoming markets with odds from diverse sources (prioritize blockchain and overtime)
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
                    MAX(CASE WHEN o.outcome = 'home' THEN o.decimal_odds END) as home_odds,
                    MAX(CASE WHEN o.outcome = 'draw' THEN o.decimal_odds END) as draw_odds,
                    MAX(CASE WHEN o.outcome = 'away' THEN o.decimal_odds END) as away_odds,
                    -- Priority scoring: blockchain > overtime > api
                    CASE 
                        WHEN m.source LIKE '%blockchain%' THEN 1
                        WHEN m.source LIKE '%overtime%' THEN 2
                        ELSE 3
                    END as source_priority
                FROM market m
                LEFT JOIN odd o ON o.source_id = m.source_id
                WHERE 
                    m.source IN ('overtime_v2', 'blockchain_live', 'overtime_soccer', 'api_live', 'overtime_v2_public')
                    AND m.is_finished = FALSE
                    AND m.maturity_date > NOW()
                    -- Show all sports
                GROUP BY 
                    m.source_id,
                    m.home_team,
                    m.away_team,
                    m.sport,
                    m.league_name,
                    m.maturity_date,
                    m.is_finished,
                    m.source
                HAVING 
                    MAX(CASE WHEN o.outcome = 'home' THEN o.decimal_odds END) IS NOT NULL
                    OR MAX(CASE WHEN o.outcome = 'draw' THEN o.decimal_odds END) IS NOT NULL
                    OR MAX(CASE WHEN o.outcome = 'away' THEN o.decimal_odds END) IS NOT NULL
                ORDER BY source_priority ASC, m.maturity_date ASC
                LIMIT 100
                """
                
                cur.execute(query)
                rows = cur.fetchall()
                logger.info(f"Query returned {len(rows)} rows for sport {sport_filter}")
                
                # Debug: Analyze data diversity and sources
                home_odds_count = sum(1 for row in rows if row['home_odds'] is not None and row['home_odds'] > 0)
                draw_odds_count = sum(1 for row in rows if row['draw_odds'] is not None and row['draw_odds'] > 0)
                away_odds_count = sum(1 for row in rows if row['away_odds'] is not None and row['away_odds'] > 0)
                
                # Check data source diversity
                sources_used = set(row['source'] for row in rows)
                sports_used = set(row['sport'] for row in rows)
                
                # Check odds diversity
                all_odds = []
                for row in rows:
                    if row['home_odds']: all_odds.append(row['home_odds'])
                    if row['draw_odds']: all_odds.append(row['draw_odds'])
                    if row['away_odds']: all_odds.append(row['away_odds'])
                unique_odds = len(set(all_odds))
                
                logger.info(f"Data Quality: {len(rows)} markets from {len(sources_used)} sources ({', '.join(sources_used)})")
                logger.info(f"Sports: {len(sports_used)} types ({', '.join(sports_used)})")
                if len(all_odds) > 0:
                    logger.info(f"Odds diversity: {unique_odds} unique odds from {len(all_odds)} total ({unique_odds/len(all_odds)*100:.1f}% diverse)")
                else:
                    logger.info("No odds found to calculate diversity")
                logger.info(f"Odds distribution - Home: {home_odds_count}, Draw: {draw_odds_count}, Away: {away_odds_count}")
                
                # Show first few odds for transparency
                if rows:
                    for i, row in enumerate(rows[:3]):
                        logger.info(f"Sample {i+1}: {row['sport']} | {row['home_team']} vs {row['away_team']} | ({row['home_odds']}, {row['draw_odds']}, {row['away_odds']}) [{row['source']}]")
                
                # Process markets with gap-based batching
                if rows:
                    # Convert to list of markets with proper timing
                    all_markets = []
                    for row in rows:
                        kickoff_time = row['maturity_date']
                        # Ensure timezone-aware
                        if kickoff_time and kickoff_time.tzinfo is None:
                            kickoff_time = kickoff_time.replace(tzinfo=timezone.utc)
                        if kickoff_time:
                            all_markets.append({
                                'row': row,
                                'kickoff_time': kickoff_time
                            })
                    
                    # Sort by kickoff time
                    all_markets.sort(key=lambda x: x['kickoff_time'])
                    
                    # Create chunks based on time gaps
                    current_chunk = []
                    chunk_start_time = None
                    
                    for i, market_data in enumerate(all_markets):
                        if not current_chunk:
                            # Start new chunk
                            current_chunk.append(market_data)
                            chunk_start_time = market_data['kickoff_time']
                        else:
                            # Check gap from last market in chunk
                            last_kickoff = current_chunk[-1]['kickoff_time']
                            time_gap = (market_data['kickoff_time'] - last_kickoff).total_seconds() / 3600
                            
                            # If gap > 3 hours, start new chunk
                            if time_gap > 3:
                                # Save current chunk
                                chunks.append({
                                    'start_time': chunk_start_time,
                                    'end_time': current_chunk[-1]['kickoff_time'],
                                    'markets': current_chunk,
                                    'count': len(current_chunk)
                                })
                                # Start new chunk
                                current_chunk = [market_data]
                                chunk_start_time = market_data['kickoff_time']
                            else:
                                current_chunk.append(market_data)
                    
                    # Save last chunk
                    if current_chunk:
                        chunks.append({
                            'start_time': chunk_start_time,
                            'end_time': current_chunk[-1]['kickoff_time'],
                            'markets': current_chunk,
                            'count': len(current_chunk)
                        })
                    
                    # Log chunk information
                    logger.info(f"Created {len(chunks)} time-based chunks")
                    for i, chunk in enumerate(chunks):
                        logger.info(f"Chunk {i+1}: {chunk['count']} markets, {chunk['start_time']} to {chunk['end_time']}")
                    
                    # Process first chunk for live trading
                    if chunks and STRATEGY_CONFIG.get('chunk_selection', 'first') == 'first':
                        first_chunk = chunks[0]
                        logger.info(f"Selected first chunk with {first_chunk['count']} markets for trading")
                        for market_data in first_chunk['markets']:
                            row = market_data['row']
                
                # Build market list for edge calculation
                market_list = []
                for row in rows:
                    market = {
                        'id': row['source_id'],
                        'source_id': row['source_id'],
                        'market_id': row['source_id'],
                        'home_team': row['home_team'],
                        'away_team': row['away_team'],
                        'sport': row['sport'],
                        'league_name': row['league_name'],
                        'maturity_date': row['maturity_date'].isoformat() if row['maturity_date'] else None,
                        'is_finished': row['is_finished'],
                        'source': row['source'],
                        'home_odds': float(row['home_odds']) if row['home_odds'] else 0,
                        'draw_odds': float(row['draw_odds']) if row['draw_odds'] else 0,
                        'away_odds': float(row['away_odds']) if row['away_odds'] else 0
                    }
                    markets.append(market)
                    market_list.append(market)
                
                # Calculate enhanced edges using real signal providers
                logger.info(f"Calculating edges for {len(market_list)} markets using enhanced edge calculator")
                try:
                    # Create expanded market list for all positions (home/draw/away)
                    expanded_markets = []
                    for market in market_list:
                        if market['home_odds'] > 0:
                            expanded_markets.append({**market, 'position': 'home'})
                        if market['draw_odds'] > 0:
                            expanded_markets.append({**market, 'position': 'draw'})
                        if market['away_odds'] > 0:
                            expanded_markets.append({**market, 'position': 'away'})
                    
                    edge_results = edge_calculator.calculate_edges(expanded_markets)
                    
                    # Group edge results by market
                    edge_by_market = {}
                    for edge_result in edge_results:
                        market_id = edge_result['market_id']
                        if market_id not in edge_by_market:
                            edge_by_market[market_id] = {}
                        # Find position from expanded_markets
                        for em in expanded_markets:
                            if em['source_id'] == market_id and em.get('position'):
                                position = em['position']
                                edge_by_market[market_id][position] = {
                                    'edge': edge_result['edge'],
                                    'probability': edge_result['probability'],
                                    'confidence': edge_result['confidence']
                                }
                                break
                    
                    # Build signals with enhanced edge calculations
                    for market in market_list:
                        market_id = market['source_id']
                        edges = edge_by_market.get(market_id, {})
                        
                        home_odds = market['home_odds']
                        draw_odds = market['draw_odds']
                        away_odds = market['away_odds']
                        
                        home_edge = edges.get('home', {}).get('edge', 0)
                        draw_edge = edges.get('draw', {}).get('edge', 0)
                        away_edge = edges.get('away', {}).get('edge', 0)
                        
                        # Calculate Kelly stakes using enhanced calculator
                        home_stake = edge_calculator.calculate_kelly_stake(
                            home_edge, home_odds, STRATEGY_CONFIG['bankroll'],
                            STRATEGY_CONFIG['kelly_fraction'], STRATEGY_CONFIG['min_bet'],
                            STRATEGY_CONFIG['cap_per_bet']
                        ) if home_odds > 0 else 0
                        
                        draw_stake = edge_calculator.calculate_kelly_stake(
                            draw_edge, draw_odds, STRATEGY_CONFIG['bankroll'],
                            STRATEGY_CONFIG['kelly_fraction'], STRATEGY_CONFIG['min_bet'],
                            STRATEGY_CONFIG['cap_per_bet']
                        ) if draw_odds > 0 else 0
                        
                        away_stake = edge_calculator.calculate_kelly_stake(
                            away_edge, away_odds, STRATEGY_CONFIG['bankroll'],
                            STRATEGY_CONFIG['kelly_fraction'], STRATEGY_CONFIG['min_bet'],
                            STRATEGY_CONFIG['cap_per_bet']
                        ) if away_odds > 0 else 0
                        
                        signal = {
                            'home_odds': home_odds,
                            'home_edge': round(home_edge, 2),
                            'home_stake': round(home_stake, 2),
                            'home_confidence': edges.get('home', {}).get('confidence', 0.5),
                            'home_implied_prob': 1.0 / home_odds if home_odds > 0 else 0,
                            'draw_odds': draw_odds,
                            'draw_edge': round(draw_edge, 2),
                            'draw_stake': round(draw_stake, 2),
                            'draw_confidence': edges.get('draw', {}).get('confidence', 0.5),
                            'draw_implied_prob': 1.0 / draw_odds if draw_odds > 0 else 0,
                            'away_odds': away_odds,
                            'away_edge': round(away_edge, 2),
                            'away_stake': round(away_stake, 2),
                            'away_confidence': edges.get('away', {}).get('confidence', 0.5),
                            'away_implied_prob': 1.0 / away_odds if away_odds > 0 else 0
                        }
                        signals.append(signal)
                    
                    logger.info(f"Successfully created {len(signals)} signals with enhanced edge calculation")
                        
                except Exception as e:
                    logger.error(f"Error calculating enhanced edges: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    # Fallback to simple edge calculation
                    for market in market_list:
                        home_odds = market['home_odds']
                        draw_odds = market['draw_odds']
                        away_odds = market['away_odds']
                        
                        # Simple fallback edge calculation
                        home_edge = (1 / home_odds * 100 - 33.33) if home_odds > 0 else 0
                        draw_edge = (1 / draw_odds * 100 - 33.33) if draw_odds > 0 else 0
                        away_edge = (1 / away_odds * 100 - 33.33) if away_odds > 0 else 0
                        
                        signal = {
                            'home_odds': home_odds,
                            'home_edge': round(home_edge, 2),
                            'home_stake': 0,
                            'home_confidence': 0.5,
                            'home_implied_prob': 1.0 / home_odds if home_odds > 0 else 0,
                            'draw_odds': draw_odds,
                            'draw_edge': round(draw_edge, 2),
                            'draw_stake': 0,
                            'draw_confidence': 0.5,
                            'draw_implied_prob': 1.0 / draw_odds if draw_odds > 0 else 0,
                            'away_odds': away_odds,
                            'away_edge': round(away_edge, 2),
                            'away_stake': 0,
                            'away_confidence': 0.5,
                            'away_implied_prob': 1.0 / away_odds if away_odds > 0 else 0
                        }
                        signals.append(signal)
                    
                    logger.info(f"Created {len(signals)} signals using fallback edge calculation")
                    
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
    
    logger.info(f"Returning {len(markets)} markets and {len(signals)} signals")
    
    # Add chunk summary
    stats = {
        'total_chunks': len(chunks),
        'chunk_details': []
    }
    
    for i, chunk in enumerate(chunks):
        time_window = ''
        # Ensure timezone-aware comparison
        now = datetime.now(timezone.utc)
        start_time = chunk['start_time']
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        hours_to_start = (start_time - now).total_seconds() / 3600
        
        if hours_to_start < 0.5:
            time_window = 'Live/Settling'
        elif hours_to_start < 6:
            time_window = 'Active Trading'
        elif hours_to_start < 24:
            time_window = 'Upcoming'
        else:
            time_window = 'Future'
            
        stats['chunk_details'].append({
            'chunk_num': i + 1,
            'time_window': time_window,
            'market_count': chunk['count'],
            'start_time': chunk['start_time'].isoformat(),
            'hours_away': round(hours_to_start, 1)
        })
    
    return markets, signals, stats, chunks

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
        
        /* Confidence score styling */
        .confidence-high { color: #00ff00; font-weight: bold; }
        .confidence-medium { color: #ffaa00; }
        .confidence-low { color: #ff6666; }
        
        /* Enhanced edge visualization */
        .best-edge { 
            background: rgba(0, 255, 0, 0.1); 
            border: 1px solid #00ff00; 
            border-radius: 3px; 
            padding: 2px 4px; 
            font-size: 0.9em;
        }
        
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
        <div id="stop-indicator" style="display: none; background: #ff0000; color: #fff; padding: 8px 16px; border-radius: 4px; font-weight: bold;">
            ⛔ TRADING STOPPED
        </div>
        <div style="display: flex; gap: 15px; align-items: center;">
            <button id="stop-button" onclick="stopTrading()" style="background: #ff0000; color: #fff; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; cursor: pointer; font-size: 0.9em;">
                🛑 STOP
            </button>
            <button id="resume-button" onclick="resumeTrading()" style="display: none; background: #00ff00; color: #000; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; cursor: pointer; font-size: 0.9em;">
                ▶️ Resume
            </button>
            <div class="header-time" id="current-time">--:--:--</div>
        </div>
    </div>
    
    <!-- Risk Management Alert -->
    <div id="risk-alert" class="risk-alert" style="display: none;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>⚠️ RISK ALERT:</strong> <span id="risk-message">High exposure detected</span>
            </div>
            <button onclick="document.getElementById('risk-alert').style.display='none'" style="background: none; border: 1px solid #ff6666; color: #ff6666; padding: 4px 8px; border-radius: 3px; cursor: pointer;">✕</button>
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
            
            <!-- Signal Intelligence Card -->
            <div class="metric-card signal-card">
                <div class="metric-label">Signal Intelligence</div>
                <div class="metric-value" id="signal-accuracy" style="font-size: 1.5em;">3 Models</div>
                <div class="metric-sub">Enhanced Edge Calculation</div>
                <div style="margin-top: 10px; font-size: 0.85em;">
                    <div>📊 Implied: <span id="implied-weight" class="positive">1.0x</span></div>
                    <div>🔗 Blockchain: <span id="blockchain-weight" class="positive">1.5x</span></div>
                    <div>🕵️ Scout: <span id="scout-weight" class="positive">1.2x</span></div>
                    <div style="margin-top: 5px; color: #00ff00;">Avg Confidence: <span id="avg-confidence">0%</span></div>
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
                    <select id="sport-filter" onchange="updateSportFilter(this.value)" style="background: #222; color: #00ff00; border: 1px solid #333; padding: 5px 10px; border-radius: 4px; font-size: 0.9em;">
                        <option value="Soccer">⚽ Soccer</option>
                        <option value="Basketball">🏀 Basketball</option>
                        <option value="Football">🏈 Football</option>
                        <option value="Baseball">⚾ Baseball</option>
                        <option value="Hockey">🏒 Hockey</option>
                        <option value="Tennis">🎾 Tennis</option>
                        <option value="Golf">⛳ Golf</option>
                        <option value="Boxing">🥊 Boxing</option>
                        <option value="All">📊 All Sports</option>
                    </select>
                    
                    <select id="confidence-filter" onchange="updateConfidenceFilter(this.value)" style="background: #222; color: #00ff00; border: 1px solid #333; padding: 5px 10px; border-radius: 4px; font-size: 0.9em;">
                        <option value="all">🎯 All Confidence</option>
                        <option value="high">⭐ High (70%+)</option>
                        <option value="medium">✨ Medium (50-70%)</option>
                        <option value="low">💫 Low (30-50%)</option>
                        <option value="positive">➕ Positive Edge Only</option>
                    </select>
                    
                    <select id="edge-filter" onchange="updateEdgeFilter(this.value)" style="background: #222; color: #00ff00; border: 1px solid #333; padding: 5px 10px; border-radius: 4px; font-size: 0.9em;">
                        <option value="all">📈 All Edges</option>
                        <option value="5">🔥 Edge > 5%</option>
                        <option value="3">🎯 Edge > 3%</option>
                        <option value="2">✅ Edge > 2%</option>
                        <option value="1">➕ Edge > 1%</option>
                        <option value="0">📊 Edge > 0%</option>
                    </select>
                    <div style="font-size: 0.9em;">
                        <span style="color: #888;">Total:</span> <span id="total-matches" style="color: #00ff00;">0</span>
                        <span style="color: #888; margin-left: 15px;">Active:</span> <span id="active-positions" style="color: #00ffff;">0</span>
                        <span style="color: #666; margin-left: 20px;">Updated:</span> <span id="markets-update-time" style="color: #888;">--:--:--</span>
                    </div>
                </div>
            </div>
            
            <!-- Market Chunks Window -->
            <div id="chunks-display" style="margin: 10px 0; padding: 10px; background: #1a1a1a; border-radius: 5px; border: 1px solid #333;">
                <div style="font-size: 0.9em; color: #00ffff; margin-bottom: 5px;">📅 Market Time Windows:</div>
                <div id="chunks-list" style="display: flex; flex-wrap: wrap; gap: 8px;">
                    <!-- Chunks will be populated here -->
                </div>
            </div>
            
            <div style="overflow: auto; max-height: 350px;">
                <table class="spreadsheet-table markets-table" style="width: 100%; min-width: 1200px;">
                    <thead>
                        <tr style="border-bottom: 2px solid #444;">
                            <th style="padding: 6px; text-align: left;">Match</th>
                            <th style="padding: 6px; text-align: center;">Source</th>
                            <th style="padding: 6px; text-align: center;">Time</th>
                            <th style="padding: 6px; text-align: center;">Status</th>
                            <th style="padding: 6px; text-align: center;">H Odds</th>
                            <th style="padding: 6px; text-align: center;">H Edge</th>
                            <th style="padding: 6px; text-align: center;">H Conf</th>
                            <th style="padding: 6px; text-align: center;">H Stake</th>
                            <th style="padding: 6px; text-align: center;">D Odds</th>
                            <th style="padding: 6px; text-align: center;">D Edge</th>
                            <th style="padding: 6px; text-align: center;">D Conf</th>
                            <th style="padding: 6px; text-align: center;">D Stake</th>
                            <th style="padding: 6px; text-align: center;">A Odds</th>
                            <th style="padding: 6px; text-align: center;">A Edge</th>
                            <th style="padding: 6px; text-align: center;">A Conf</th>
                            <th style="padding: 6px; text-align: center;">A Stake</th>
                            <th style="padding: 6px; text-align: center;">Best</th>
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
                <button class="tab-btn" onclick="showTab('analytics')">Analytics</button>
            </div>
            
            <div id="open-positions" class="tab-content active">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                    <thead>
                        <tr style="border-bottom: 1px solid #333;">
                            <th style="padding: 8px; text-align: left;">Market</th>
                            <th style="padding: 8px; text-align: center;">Side</th>
                            <th style="padding: 8px; text-align: right;">Stake</th>
                            <th style="padding: 8px; text-align: center;">Odds</th>
                            <th style="padding: 8px; text-align: center;">Time</th>
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
            
            <div id="analytics-positions" class="tab-content">
                <div style="padding: 20px;">
                    <h4 style="color: #00ff00; margin-bottom: 15px;">📈 Advanced Performance Analytics</h4>
                    
                    <!-- Key Performance Metrics -->
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px;">
                        <div style="background: #1a1a1a; padding: 15px; border-radius: 8px; border: 1px solid #333;">
                            <div style="color: #888; font-size: 0.9em;">ROI</div>
                            <div style="font-size: 1.8em; color: #00ff00;" id="roi-detailed">0%</div>
                        </div>
                        <div style="background: #1a1a1a; padding: 15px; border-radius: 8px; border: 1px solid #333;">
                            <div style="color: #888; font-size: 0.9em;">Profit Factor</div>
                            <div style="font-size: 1.8em; color: #00ffff;" id="profit-factor-detailed">0.0</div>
                        </div>
                        <div style="background: #1a1a1a; padding: 15px; border-radius: 8px; border: 1px solid #333;">
                            <div style="color: #888; font-size: 0.9em;">Sharpe Ratio</div>
                            <div style="font-size: 1.8em; color: #ffff00;" id="sharpe-ratio">0.0</div>
                        </div>
                        <div style="background: #1a1a1a; padding: 15px; border-radius: 8px; border: 1px solid #333;">
                            <div style="color: #888; font-size: 0.9em;">Max Drawdown</div>
                            <div style="font-size: 1.8em; color: #ff6600;" id="max-drawdown">0%</div>
                        </div>
                    </div>
                    
                    <!-- Win/Loss Analysis -->
                    <div style="background: #111; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <h5 style="color: #00ffff; margin-bottom: 10px;">💰 Win/Loss Analysis</h5>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                            <div>
                                <div style="color: #888;">Avg Win</div>
                                <div style="color: #00ff00; font-size: 1.2em;" id="avg-win">$0</div>
                            </div>
                            <div>
                                <div style="color: #888;">Avg Loss</div>
                                <div style="color: #ff3333; font-size: 1.2em;" id="avg-loss">$0</div>
                            </div>
                            <div>
                                <div style="color: #888;">Win/Loss Ratio</div>
                                <div style="color: #00ffff; font-size: 1.2em;" id="win-loss-ratio">0.0</div>
                            </div>
                        </div>
                        <div style="margin-top: 10px; display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                            <div>
                                <div style="color: #888;">Max Consecutive Wins</div>
                                <div style="color: #00ff00;" id="consecutive-wins">0</div>
                            </div>
                            <div>
                                <div style="color: #888;">Max Consecutive Losses</div>
                                <div style="color: #ff3333;" id="consecutive-losses">0</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Performance by Sport -->
                    <div style="background: #111; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <h5 style="color: #00ffff; margin-bottom: 10px;">⚽ Performance by Sport</h5>
                        <div id="sport-performance" style="max-height: 200px; overflow-y: auto;">
                            <table style="width: 100%; font-size: 0.9em;">
                                <thead>
                                    <tr style="border-bottom: 1px solid #333;">
                                        <th style="padding: 8px; text-align: left;">Sport</th>
                                        <th style="padding: 8px; text-align: center;">Trades</th>
                                        <th style="padding: 8px; text-align: center;">Win Rate</th>
                                        <th style="padding: 8px; text-align: center;">P&L</th>
                                        <th style="padding: 8px; text-align: center;">Avg Return</th>
                                    </tr>
                                </thead>
                                <tbody id="sport-performance-tbody">
                                    <!-- Will be populated dynamically -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                    
                    <!-- Performance by Outcome -->
                    <div style="background: #111; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <h5 style="color: #00ffff; margin-bottom: 10px;">🎯 Performance by Outcome</h5>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                            <div id="home-performance" style="text-align: center;">
                                <div style="color: #888;">Home</div>
                                <div style="font-size: 1.2em; color: #00ff00;">Win Rate: <span id="home-win-rate">0%</span></div>
                                <div style="color: #888;">P&L: <span id="home-pnl">$0</span></div>
                            </div>
                            <div id="draw-performance" style="text-align: center;">
                                <div style="color: #888;">Draw</div>
                                <div style="font-size: 1.2em; color: #ffff00;">Win Rate: <span id="draw-win-rate">0%</span></div>
                                <div style="color: #888;">P&L: <span id="draw-pnl">$0</span></div>
                            </div>
                            <div id="away-performance" style="text-align: center;">
                                <div style="color: #888;">Away</div>
                                <div style="font-size: 1.2em; color: #00ffff;">Win Rate: <span id="away-win-rate">0%</span></div>
                                <div style="color: #888;">P&L: <span id="away-pnl">$0</span></div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Hourly Performance -->
                    <div style="background: #111; padding: 15px; border-radius: 8px;">
                        <h5 style="color: #00ffff; margin-bottom: 10px;">⏰ Performance by Hour</h5>
                        <div id="hourly-performance" style="font-size: 0.9em; color: #888;">
                            <!-- Will be populated with hourly data -->
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
        
        // Filter state
        let currentFilters = {
            sport: 'Soccer',
            confidence: 'all',
            edge: 'all',
            chunk: 'all'
        };
        
        let globalChunks = [];  // Store chunks for filtering
        
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
            console.log('Markets count:', data.markets ? data.markets.length : 'NO MARKETS');
            console.log('Signals count:', data.signals ? data.signals.length : 'NO SIGNALS');
            updateDashboard(data);
        });
        
        socket.on('activity', (item) => {
            addActivityItem(item);
        });
        
        socket.on('stop_status', (status) => {
            updateStopStatus(status);
        });
        
        // Update dashboard
        function updateDashboard(data) {
            // Update metrics
            updateMetrics(data);
            
            // Update chunks
            if (data.chunks) {
                updateChunks(data.chunks);
            }
            
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
            
            // Apply filters after updating markets
            if (data.markets) {
                applyFilters();
            }
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
            
            // Update Analytics Tab
            updateAnalytics(performance);
            
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
            
            // Risk Management Alert
            checkRiskAlert(portfolio);
        }
        
        function updateChunks(chunks) {
            const chunksList = document.getElementById('chunks-list');
            chunksList.innerHTML = '';
            
            // Store chunks globally for filtering
            globalChunks = chunks || [];
            
            if (!chunks || chunks.length === 0) {
                chunksList.innerHTML = '<span style="color: #666;">No chunks available</span>';
                return;
            }
            
            chunks.forEach((chunk, index) => {
                const isActive = index === 0; // First chunk is active for trading
                const chunkDiv = document.createElement('div');
                chunkDiv.style.cssText = `
                    padding: 8px 12px;
                    background: ${isActive ? '#003300' : '#1a1a1a'};
                    border: 1px solid ${isActive ? '#00ff00' : '#333'};
                    border-radius: 4px;
                    font-size: 0.8em;
                    min-width: 120px;
                    text-align: center;
                    cursor: pointer;
                    transition: all 0.2s;
                `;
                
                // Make chunk clickable
                chunkDiv.onclick = function() {
                    filterByChunk(index);
                };
                
                chunkDiv.onmouseenter = function() {
                    if (!isActive) {
                        this.style.background = '#003300';
                        this.style.borderColor = '#00ff00';
                    }
                };
                
                chunkDiv.onmouseleave = function() {
                    if (!isActive) {
                        this.style.background = '#1a1a1a';
                        this.style.borderColor = '#333';
                    }
                };
                
                const startTime = new Date(chunk.start_time);
                const hoursAway = parseFloat(chunk.hours_away) || 0;
                
                chunkDiv.innerHTML = `
                    <div style="color: ${isActive ? '#00ff00' : '#00ffff'}; font-weight: ${isActive ? 'bold' : 'normal'};">
                        ${isActive ? '🎯 ' : ''}Chunk ${index + 1}
                    </div>
                    <div style="color: #888; font-size: 0.9em;">
                        ${chunk.market_count} markets
                    </div>
                    <div style="color: ${hoursAway < 2 ? '#ff6666' : '#aaa'}; font-size: 0.8em;">
                        ${hoursAway.toFixed(1)}h away
                    </div>
                `;
                
                chunksList.appendChild(chunkDiv);
            });
        }
        
        function updateMarkets(markets, signals) {
            console.log('updateMarkets called with:', markets.length, 'markets and', signals ? signals.length : 'no signals');
            const tbody = document.getElementById('matches-tbody');
            tbody.innerHTML = '';
            
            let totalExposure = 0;
            let activeCount = 0;
            
            markets.forEach((market, i) => {
                const signal = signals ? signals[i] : {};
                const row = document.createElement('tr');
                
                // Match info with external links
                let matchContent = market.home_team + ' vs ' + market.away_team;
                let marketLink = '';
                
                // Generate appropriate link based on market source and ID format
                if (market.source && market.market_id) {
                    if (market.source.includes('overtime')) {
                        // Overtime markets - extract game ID from source_id if needed
                        if (market.market_id.startsWith('live_0x') || market.market_id.startsWith('v2_0x')) {
                            // Extract the hex part for Overtime V2 markets
                            const hexPart = market.market_id.split('_')[1];
                            marketLink = `https://overtimemarkets.xyz/markets/optimism/market/${hexPart}`;
                        }
                    } else if (market.source.includes('blockchain')) {
                        // Blockchain markets - check if it's a transaction hash
                        if (market.market_id.startsWith('0x') && market.market_id.length >= 42) {
                            // Ethereum address format - likely on Optimism
                            if (market.source.includes('optimism')) {
                                marketLink = `https://optimistic.etherscan.io/address/${market.market_id}`;
                            } else if (market.source.includes('arbitrum')) {
                                marketLink = `https://arbiscan.io/address/${market.market_id}`;
                            } else {
                                // Default to Optimism for Overtime markets
                                marketLink = `https://optimistic.etherscan.io/address/${market.market_id}`;
                            }
                        }
                    }
                    
                    // Add a small icon to indicate external link
                    if (marketLink) {
                        matchContent += ` <a href="${marketLink}" target="_blank" style="color: #00ffff; text-decoration: none; font-size: 0.8em;" title="View on ${market.source}">🔗</a>`;
                    }
                }
                
                let matchCell = '<td style="padding: 6px;">' + matchContent + '</td>';
                
                // Source info
                let sourceDisplay = '';
                if (market.source) {
                    // Simplify source display
                    if (market.source.includes('overtime')) {
                        sourceDisplay = 'Overtime';
                    } else if (market.source.includes('blockchain')) {
                        if (market.source.includes('optimism')) {
                            sourceDisplay = 'OP Chain';
                        } else if (market.source.includes('arbitrum')) {
                            sourceDisplay = 'Arb Chain';
                        } else {
                            sourceDisplay = 'Blockchain';
                        }
                    } else if (market.source === 'api_live_real') {
                        sourceDisplay = 'API';
                    } else {
                        sourceDisplay = market.source;
                    }
                }
                let sourceCell = '<td style="padding: 6px; text-align: center; color: #888; font-size: 0.85em;">' + sourceDisplay + '</td>';
                
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
                
                // Odds, edges, confidence, and stakes
                let cells = '';
                let bestEdge = -100;
                let bestOutcome = '';
                
                ['home', 'draw', 'away'].forEach(outcome => {
                    const odds = signal[outcome + '_odds'] || 0;
                    const edge = signal[outcome + '_edge'] || 0;
                    const confidence = signal[outcome + '_confidence'] || 0;
                    const stake = signal[outcome + '_stake'] || 0;
                    
                    const edgeClass = edge > 2 ? 'edge-positive' : 
                                    (edge < -2 ? 'edge-negative' : '');
                    
                    const confClass = confidence > 0.7 ? 'confidence-high' : 
                                    (confidence < 0.4 ? 'confidence-low' : 'confidence-medium');
                    
                    // Track best edge
                    if (edge > bestEdge) {
                        bestEdge = edge;
                        bestOutcome = outcome.charAt(0).toUpperCase();
                    }
                    
                    cells += '<td style="padding: 6px; text-align: center;">' + 
                            (odds > 0 ? odds.toFixed(2) : '-') + '</td>';
                    cells += '<td style="padding: 6px; text-align: center;" class="' + edgeClass + '">' + 
                            (edge !== 0 ? edge.toFixed(1) + '%' : '-') + '</td>';
                    cells += '<td style="padding: 6px; text-align: center; font-size: 0.85em;" class="' + confClass + '">' + 
                            (confidence > 0 ? (confidence * 100).toFixed(0) + '%' : '-') + '</td>';
                    cells += '<td style="padding: 6px; text-align: center;">' + 
                            (stake > 0 ? '$' + stake.toFixed(0) : '-') + '</td>';
                    
                    if (stake > 0) {
                        totalExposure += stake;
                        activeCount++;
                    }
                });
                
                // Best recommendation
                let bestCell = '<td style="padding: 6px; text-align: center; font-weight: bold;">';
                if (bestEdge > 2) {
                    bestCell += '<span class="edge-positive">' + bestOutcome + ' ' + bestEdge.toFixed(1) + '%</span>';
                } else if (bestEdge < -2) {
                    bestCell += '<span class="edge-negative">Avoid</span>';
                } else {
                    bestCell += '<span class="neutral">-</span>';
                }
                bestCell += '</td>';
                
                row.innerHTML = matchCell + sourceCell + timeCell + statusCell + cells + bestCell;
                
                // Store market data for filtering
                row.marketData = {
                    sport: market.sport || 'Soccer',
                    home_edge: signal.home_edge || 0,
                    draw_edge: signal.draw_edge || 0,
                    away_edge: signal.away_edge || 0,
                    home_confidence: signal.home_confidence || 0,
                    maturity_date: market.maturity_date,
                    draw_confidence: signal.draw_confidence || 0,
                    away_confidence: signal.away_confidence || 0,
                    best_edge: bestEdge
                };
                
                tbody.appendChild(row);
            });
            
            // Calculate average confidence
            let totalConfidence = 0;
            let confidenceCount = 0;
            markets.forEach((market, i) => {
                const signal = signals ? signals[i] : {};
                ['home', 'draw', 'away'].forEach(outcome => {
                    const confidence = signal[outcome + '_confidence'] || 0;
                    if (confidence > 0) {
                        totalConfidence += confidence;
                        confidenceCount++;
                    }
                });
            });
            
            const avgConfidence = confidenceCount > 0 ? (totalConfidence / confidenceCount * 100) : 0;
            document.getElementById('avg-confidence').textContent = avgConfidence.toFixed(0) + '%';
            
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
            if (positions.open) {
                Object.values(positions.open).forEach(pos => {
                    // Calculate time until match
                    let timeStr = '';
                    if (pos.maturity_date) {
                        const kickoff = new Date(pos.maturity_date);
                        const now = new Date();
                        const hoursToKickoff = (kickoff - now) / (1000 * 60 * 60);
                        if (hoursToKickoff > 0) {
                            timeStr = hoursToKickoff > 24 ? 
                                (hoursToKickoff / 24).toFixed(1) + 'd' : 
                                hoursToKickoff.toFixed(1) + 'h';
                        } else {
                            timeStr = 'Live';
                        }
                    }
                    
                    const row = document.createElement('tr');
                    row.innerHTML = '<td style="padding: 8px;">' + pos.market_name + '</td>' +
                        '<td style="padding: 8px; text-align: center;">' + pos.outcome.toUpperCase() + '</td>' +
                        '<td style="padding: 8px; text-align: right;">$' + pos.total_stake.toFixed(2) + '</td>' +
                        '<td style="padding: 8px; text-align: center;">' + pos.avg_odds.toFixed(2) + '</td>' +
                        '<td style="padding: 8px; text-align: center; color: ' + (timeStr === 'Live' ? '#ff3333' : '#888') + ';">' + timeStr + '</td>' +
                        '<td style="padding: 8px; text-align: right;">$' + pos.current_value.toFixed(2) + '</td>' +
                        '<td style="padding: 8px; text-align: right;" class="' + (pos.pnl >= 0 ? 'positive' : 'negative') + '">' +
                            (pos.pnl >= 0 ? '+' : '') + '$' + Math.abs(pos.pnl).toFixed(2) + '</td>';
                    openTbody.appendChild(row);
                });
            }
            
            // Closed positions
            if (positions.closed) {
                positions.closed.forEach(pos => {
                const row = document.createElement('tr');
                const resultStr = pos.result === 'won' ? 'Won' : 'Lost';
                const resultClass = pos.result === 'won' ? 'positive' : 'negative';
                
                row.innerHTML = '<td style="padding: 8px;">' + pos.market_name + '</td>' +
                    '<td style="padding: 8px; text-align: center;">' + pos.outcome.toUpperCase() + '</td>' +
                    '<td style="padding: 8px; text-align: right;">$' + pos.total_stake.toFixed(2) + '</td>' +
                    '<td style="padding: 8px; text-align: center;">' + pos.avg_odds.toFixed(2) + '</td>' +
                    '<td style="padding: 8px; text-align: center;" class="' + resultClass + '">' + resultStr + '</td>' +
                    '<td style="padding: 8px; text-align: right;" class="' + (pos.pnl >= 0 ? 'positive' : 'negative') + '">' +
                        (pos.pnl >= 0 ? '+' : '') + '$' + Math.abs(pos.pnl).toFixed(2) + '</td>';
                closedTbody.appendChild(row);
                
                totalPnl += pos.pnl;
                if (pos.result === 'won') wins++;
                else losses++;
                });
            }
            
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
            div.innerHTML = '<div class="activity-time">' + time.toLocaleTimeString() + '</div>' +
                '<div>' + item.message + '</div>';
            
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
        
        // Update analytics display
        function updateAnalytics(performance) {
            // Key metrics
            document.getElementById('roi-detailed').textContent = 
                (performance.roi || 0).toFixed(2) + '%';
            document.getElementById('roi-detailed').style.color = 
                performance.roi >= 0 ? '#00ff00' : '#ff3333';
            
            document.getElementById('profit-factor-detailed').textContent = 
                (performance.profit_factor || 0).toFixed(2);
            document.getElementById('sharpe-ratio').textContent = 
                (performance.sharpe_ratio || 0).toFixed(2);
            document.getElementById('max-drawdown').textContent = 
                (performance.max_drawdown || 0).toFixed(1) + '%';
            
            // Win/Loss analysis
            document.getElementById('avg-win').textContent = 
                '$' + (performance.avg_win || 0).toFixed(2);
            document.getElementById('avg-loss').textContent = 
                '$' + (performance.avg_loss || 0).toFixed(2);
            document.getElementById('win-loss-ratio').textContent = 
                (performance.win_loss_ratio || 0).toFixed(2);
            document.getElementById('consecutive-wins').textContent = 
                performance.consecutive_wins || 0;
            document.getElementById('consecutive-losses').textContent = 
                performance.consecutive_losses || 0;
            
            // Performance by sport
            const sportTbody = document.getElementById('sport-performance-tbody');
            sportTbody.innerHTML = '';
            const sportData = performance.trades_by_sport || {};
            for (const sport in sportData) {
                const data = sportData[sport];
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td style="padding: 8px;">${sport}</td>
                    <td style="padding: 8px; text-align: center;">${data.count}</td>
                    <td style="padding: 8px; text-align: center; color: ${data.win_rate > 50 ? '#00ff00' : '#ff3333'};">
                        ${data.win_rate.toFixed(1)}%
                    </td>
                    <td style="padding: 8px; text-align: center; color: ${data.total_pnl >= 0 ? '#00ff00' : '#ff3333'};">
                        $${data.total_pnl.toFixed(2)}
                    </td>
                    <td style="padding: 8px; text-align: center; color: ${data.avg_return >= 0 ? '#00ff00' : '#ff3333'};">
                        ${data.avg_return.toFixed(1)}%
                    </td>
                `;
                sportTbody.appendChild(row);
            }
            
            // Performance by outcome
            const outcomeData = performance.trades_by_outcome || {};
            ['home', 'draw', 'away'].forEach(outcome => {
                const data = outcomeData[outcome] || {win_rate: 0, total_pnl: 0};
                document.getElementById(outcome + '-win-rate').textContent = 
                    data.win_rate.toFixed(1) + '%';
                document.getElementById(outcome + '-win-rate').parentElement.style.color = 
                    data.win_rate > 50 ? '#00ff00' : (data.win_rate < 40 ? '#ff3333' : '#ffff00');
                document.getElementById(outcome + '-pnl').textContent = 
                    '$' + data.total_pnl.toFixed(2);
                document.getElementById(outcome + '-pnl').style.color = 
                    data.total_pnl >= 0 ? '#00ff00' : '#ff3333';
            });
            
            // Hourly performance
            const hourlyDiv = document.getElementById('hourly-performance');
            const hourlyData = performance.hourly_performance || {};
            let hourlyHtml = '<div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px;">';
            
            for (let hour = 0; hour < 24; hour++) {
                const data = hourlyData[hour] || {count: 0, win_rate: 0, total_pnl: 0};
                const hasData = data.count > 0;
                hourlyHtml += `
                    <div style="text-align: center; padding: 5px; background: ${hasData ? '#1a1a1a' : '#0a0a0a'}; border-radius: 4px;">
                        <div style="font-size: 0.8em;">${hour}:00</div>
                        <div style="color: ${data.total_pnl >= 0 ? '#00ff00' : '#ff3333'}; font-size: 0.9em;">
                            ${hasData ? data.count + ' trades' : '-'}
                        </div>
                    </div>
                `;
            }
            hourlyHtml += '</div>';
            hourlyDiv.innerHTML = hourlyHtml;
        }
        
        // Stop trading
        function stopTrading() {
            if(confirm('Are you sure you want to STOP all trading and close all positions?')) {
                socket.emit('stop_trading');
            }
        }
        
        // Filter functions
        function updateSportFilter(value) {
            currentFilters.sport = value;
            applyFilters();
        }
        
        function updateConfidenceFilter(value) {
            currentFilters.confidence = value;
            applyFilters();
        }
        
        function updateEdgeFilter(value) {
            currentFilters.edge = value;
            applyFilters();
        }
        
        function filterByChunk(chunkIndex) {
            currentFilters.chunk = chunkIndex;
            applyFilters();
            
            // Update visual state of chunks
            const chunkDivs = document.querySelectorAll('#chunks-list > div');
            chunkDivs.forEach((div, index) => {
                const isActive = index === chunkIndex;
                div.style.background = isActive ? '#003300' : '#1a1a1a';
                div.style.borderColor = isActive ? '#00ff00' : '#333';
            });
        }
        
        // Apply filters to markets display
        function applyFilters() {
            const rows = document.querySelectorAll('#matches-tbody tr');
            let visibleCount = 0;
            
            rows.forEach(row => {
                const marketData = row.marketData;
                if (!marketData) return;
                
                let show = true;
                
                // Sport filter
                if (currentFilters.sport !== 'All' && marketData.sport !== currentFilters.sport) {
                    show = false;
                }
                
                // Confidence filter
                if (show && currentFilters.confidence !== 'all') {
                    const maxConfidence = Math.max(
                        marketData.home_confidence || 0,
                        marketData.draw_confidence || 0,
                        marketData.away_confidence || 0
                    );
                    
                    switch(currentFilters.confidence) {
                        case 'high':
                            show = maxConfidence >= 0.7;
                            break;
                        case 'medium':
                            show = maxConfidence >= 0.5 && maxConfidence < 0.7;
                            break;
                        case 'low':
                            show = maxConfidence >= 0.3 && maxConfidence < 0.5;
                            break;
                        case 'positive':
                            show = Math.max(
                                marketData.home_edge || -100,
                                marketData.draw_edge || -100,
                                marketData.away_edge || -100
                            ) > 0;
                            break;
                    }
                }
                
                // Edge filter
                if (show && currentFilters.edge !== 'all') {
                    const maxEdge = Math.max(
                        marketData.home_edge || -100,
                        marketData.draw_edge || -100,
                        marketData.away_edge || -100
                    );
                    const threshold = parseFloat(currentFilters.edge);
                    show = maxEdge > threshold;
                }
                
                // Chunk filter
                if (show && currentFilters.chunk !== 'all' && globalChunks.length > 0) {
                    const chunkIndex = currentFilters.chunk;
                    if (chunkIndex >= 0 && chunkIndex < globalChunks.length) {
                        const chunk = globalChunks[chunkIndex];
                        const marketTime = new Date(marketData.maturity_date);
                        const chunkStart = new Date(chunk.start_time);
                        const chunkEnd = chunk.end_time ? new Date(chunk.end_time) : 
                            new Date(chunkStart.getTime() + 24 * 60 * 60 * 1000); // Default 24h chunk
                        
                        show = marketTime >= chunkStart && marketTime <= chunkEnd;
                    }
                }
                
                row.style.display = show ? 'table-row' : 'none';
                if (show) visibleCount++;
            });
            
            // Update visible count
            document.getElementById('total-matches').textContent = visibleCount;
        }
        
        // Resume trading
        function resumeTrading() {
            socket.emit('resume_trading');
        }
        
        // Update stop status
        function updateStopStatus(status) {
            const stopIndicator = document.getElementById('stop-indicator');
            const stopButton = document.getElementById('stop-button');
            const resumeButton = document.getElementById('resume-button');
            
            if (status.is_stopped) {
                stopIndicator.style.display = 'block';
                stopButton.style.display = 'none';
                resumeButton.style.display = status.can_resume ? 'block' : 'none';
                
                // Add stop alert
                const alertDiv = document.getElementById('risk-alert');
                const messageSpan = document.getElementById('risk-message');
                alertDiv.style.display = 'block';
                alertDiv.style.background = '#ff0000';
                messageSpan.textContent = `STOPPED: ${status.reason || 'Manual stop'} - ${status.positions_closed || 0} positions closed`;
            } else {
                stopIndicator.style.display = 'none';
                stopButton.style.display = 'block';
                resumeButton.style.display = 'none';
            }
        }
        
        // Risk Management Alert System
        function checkRiskAlert(portfolio) {
            const exposure = portfolio.exposure_pct || 0;
            const cash = portfolio.current_bankroll || 0;
            const alertDiv = document.getElementById('risk-alert');
            const messageSpan = document.getElementById('risk-message');
            
            let alertMessage = '';
            let showAlert = false;
            
            // Check for over-exposure (>200%)
            if (exposure > 200) {
                alertMessage = `CRITICAL: Portfolio exposure at ${exposure.toFixed(1)}% - severely over-leveraged!`;
                showAlert = true;
            }
            // Check for high exposure (>100%)
            else if (exposure > 100) {
                alertMessage = `WARNING: Portfolio exposure at ${exposure.toFixed(1)}% - over-leveraged!`;
                showAlert = true;
            }
            // Check for negative cash
            else if (cash < 0) {
                alertMessage = `WARNING: Negative cash balance of $${cash.toFixed(2)}`;
                showAlert = true;
            }
            
            if (showAlert) {
                messageSpan.textContent = alertMessage;
                alertDiv.style.display = 'block';
                
                // Add visual indicators
                if (exposure > 200) {
                    alertDiv.style.background = 'linear-gradient(135deg, #ff1744, #d50000)';
                    alertDiv.style.animation = 'pulse 1s infinite';
                } else {
                    alertDiv.style.background = 'linear-gradient(135deg, #ff6600, #ff3300)';
                    alertDiv.style.animation = 'none';
                }
            } else {
                alertDiv.style.display = 'none';
            }
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

@app.route('/debug')
def debug():
    """Debug endpoint to show raw data"""
    try:
        markets, signals, stats, chunks = get_market_data('Soccer')
        
        debug_html = f"""
        <html>
        <head><title>Debug - Raw Data</title></head>
        <body style="font-family: monospace; background: #000; color: #0f0; padding: 20px;">
        <h1>📊 Debug Data Output</h1>
        <h2>Markets: {len(markets)}</h2>
        <h2>Signals: {len(signals)}</h2>
        
        <h3>First 3 Markets:</h3>
        <pre>{str(markets[:3])}</pre>
        
        <h3>First 3 Signals:</h3>
        <pre>{str(signals[:3])}</pre>
        
        <h3>Stats:</h3>
        <pre>{str(stats)}</pre>
        
        <p><a href="/" style="color: #0ff;">← Back to Dashboard</a></p>
        </body>
        </html>
        """
        return debug_html
    except Exception as e:
        return f"<pre>Error: {str(e)}</pre>"

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('request_dashboard_data')
def handle_request_dashboard_data():
    """Send dashboard data to client"""
    emit('dashboard_update', get_dashboard_data())

@socketio.on('stop_trading')
def handle_stop_trading():
    """Handle manual stop request"""
    session_id = session_manager.get_current_session()
    if session_id:
        result = stop_loss_manager.manual_stop(session_id, "Manual stop button pressed")
        emit('activity', {
            'type': 'stop',
            'message': f"STOPPED: {result.get('positions_closed', 0)} positions closed. Total P&L: ${result.get('total_pnl', 0):.2f}",
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'severity': 'critical'
        })
        emit('stop_status', {
            'is_stopped': True,
            'can_resume': False,
            'reason': result.get('reason'),
            'positions_closed': result.get('positions_closed', 0)
        })
    emit('dashboard_update', get_dashboard_data())

@socketio.on('resume_trading')
def handle_resume_trading():
    """Handle resume trading request"""
    can_resume, reason = stop_loss_manager.can_resume_trading()
    if can_resume:
        stop_loss_manager.resume_trading()
        emit('activity', {
            'type': 'resume',
            'message': "Trading resumed",
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        emit('stop_status', {'is_stopped': False})
    else:
        emit('activity', {
            'type': 'warning',
            'message': f"Cannot resume: {reason}",
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    emit('dashboard_update', get_dashboard_data())

def get_dashboard_data(sport_filter='All'):
    """Get all dashboard data"""
    try:
        # Get current session
        session_id = session_manager.get_current_session()
        if not session_id:
            session_id = session_manager.create_session(initial_bankroll=STRATEGY_CONFIG['bankroll'])
        
        # Get full session data
        session = session_manager.get_session(session_id)
        if not session:
            logger.error("Failed to get session data")
            return {}
        
        # Get markets and signals
        markets, signals, stats, chunks = get_market_data(sport_filter)
        logger.info(f"Found {len(markets)} markets for {sport_filter}")
        
        # Get performance with enhanced analytics
        performance = session_manager.get_session_performance(session['session_id'])
        enhanced_analytics = session_manager.get_enhanced_performance_analytics(session['session_id'])
        
        # Merge enhanced analytics into performance
        performance.update(enhanced_analytics)
        
        # Get positions separately
        positions = session_manager.get_positions(session['session_id'])
        open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
        closed_positions = [p for p in positions if p['status'] in ['won', 'lost', 'settled']]
        
        # Format positions for frontend
        formatted_open = {}
        for i, pos in enumerate(open_positions):
            # Get market info if team names are empty
            market_name = f"{pos.get('home_team', '')} vs {pos.get('away_team', '')}"
            
            if market_name == " vs " or market_name.strip() == "vs":  # Empty names, look up from database
                with db_manager.get_db_session() as db:
                    market = db.query(Market).filter(
                        Market.source_id == pos['match_id']
                    ).first()
                    if market:
                        market_name = f"{market.home_team} vs {market.away_team}"
            
            # Get maturity date from market
            maturity_date = None
            if market_name == " vs " or market_name.strip() == "vs":
                # Already looked up market above
                if market:
                    maturity_date = market.maturity_date.isoformat() if market.maturity_date else None
            else:
                # Need to look up market for maturity date
                with db_manager.get_db_session() as db:
                    market = db.query(Market).filter(
                        Market.source_id == pos['match_id']
                    ).first()
                    if market:
                        maturity_date = market.maturity_date.isoformat() if market.maturity_date else None
            
            formatted_open[str(i)] = {
                'market_name': market_name,
                'outcome': pos.get('bet_on', ''),
                'total_stake': float(pos.get('stake', 0)),
                'avg_odds': float(pos.get('odds', 0)),
                'current_value': float(pos.get('stake', 0)),  # For pending, same as stake
                'pnl': 0,  # No P&L until settled
                'status': 'open',
                'maturity_date': maturity_date
            }
        
        formatted_closed = []
        for pos in closed_positions[-20:]:  # Last 20 closed
            market_name = f"{pos.get('home_team', '')} vs {pos.get('away_team', '')}"
            
            if market_name == " vs " or market_name.strip() == "vs":
                with db_manager.get_db_session() as db:
                    market = db.query(Market).filter(
                        Market.source_id == pos['match_id']
                    ).first()
                    if market:
                        market_name = f"{market.home_team} vs {market.away_team}"
            
            payout = float(pos.get('payout', 0))
            stake = float(pos.get('stake', 0))
            
            formatted_closed.append({
                'market_name': market_name,
                'outcome': pos.get('bet_on', ''),
                'total_stake': stake,
                'avg_odds': float(pos.get('odds', 0)),
                'payout': payout,
                'pnl': payout - stake,
                'result': 'won' if payout > stake else 'lost',
                'settled_at': pos.get('resolved_at', pos.get('placed_at', ''))
            })
        
        # Calculate exposure
        total_exposure = sum(float(p['stake']) for p in open_positions)
        exposure_pct = (total_exposure / STRATEGY_CONFIG['bankroll'] * 100) if STRATEGY_CONFIG['bankroll'] else 0
        
        # Format response with JSON serialization
        response_data = {
            'session': {
                'session_id': session['session_id'],
                'status': session.get('status', 'active')
            },
            'portfolio': {
                'portfolio_value': session.get('portfolio_value', STRATEGY_CONFIG['bankroll']),
                'current_bankroll': session.get('current_bankroll', STRATEGY_CONFIG['bankroll']),
                'positions': {str(i): p for i, p in enumerate(open_positions)},
                'positions_value': total_exposure,
                'exposure_pct': exposure_pct
            },
            'performance': performance,
            'markets': markets,
            'signals': signals,
            'chunks': chunks,
            'positions': {
                'open': formatted_open,
                'closed': formatted_closed
            },
            'strategy': STRATEGY_CONFIG
        }
        
        # Convert to JSON and back to ensure all types are serializable
        return json.loads(json.dumps(response_data, cls=CustomJSONEncoder))
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        return {}

def execute_portfolio_trades():
    """Execute portfolio-based trades using optimal allocation"""
    try:
        # Check stop status first
        stop_status = stop_loss_manager.get_stop_status()
        if stop_status['is_stopped']:
            return {'success': False, 'error': f"Trading stopped: {stop_status['stop_reason']}"}
            
        session_id = session_manager.get_current_session()
        if not session_id:
            return {'success': False, 'error': 'No active session'}
        
        session = session_manager.get_session(session_id)
        if not session:
            return {'success': False, 'error': 'Failed to get session data'}
        
        # Check if we're over-leveraged
        positions = session_manager.get_positions(session_id)
        open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
        current_exposure = sum(float(p['stake']) for p in open_positions)
        exposure_pct = (current_exposure / STRATEGY_CONFIG['bankroll'] * 100) if STRATEGY_CONFIG['bankroll'] else 0
        
        MAX_EXPOSURE_PCT = 30
        if exposure_pct > MAX_EXPOSURE_PCT:
            logger.warning(f"⚠️ Exposure at {exposure_pct:.1f}% - skipping new trades (max: {MAX_EXPOSURE_PCT}%)")
            return {'success': True, 'trades_made': 0, 'reason': 'exposure_limit'}
        
        # Get market data
        markets, signals, stats, chunks = get_market_data()
        
        # Use portfolio engine for optimization
        result = portfolio_engine.execute_portfolio_trades(
            session_id, 
            markets, 
            signals,
            STRATEGY_CONFIG['bankroll']
        )
        
        # Log trades if successful
        if result.get('success') and result.get('trades_made', 0) > 0:
            logger.info(f"✅ Portfolio rebalanced: {result.get('trades_made')} trades, "
                       f"${result.get('total_stake', 0):.2f} deployed, "
                       f"portfolio size: {result.get('portfolio_size', 0)}")
            
            # Emit activity for each trade
            positions = session_manager.get_positions(session_id)
            recent_trades = sorted(positions, key=lambda x: x.get('placed_at', ''), reverse=True)[:result['trades_made']]
            
            for trade in recent_trades:
                socketio.emit('activity', {
                    'type': 'trade',
                    'message': f"Portfolio: {trade['bet_on']} ${trade['stake']:.2f} on {trade['home_team']} vs {trade['away_team']} @ {trade['odds']:.2f}",
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
        
        return result
        
    except Exception as e:
        logger.error(f"Error in portfolio trading: {e}")
        return {'success': False, 'error': str(e)}

def execute_paper_trades():
    """Execute paper trades using evaluate_open_markets logic"""
    try:
        session_id = session_manager.get_current_session()
        if not session_id:
            return {'success': False, 'error': 'No active session'}
        
        session = session_manager.get_session(session_id)
        if not session:
            return {'success': False, 'error': 'Failed to get session data'}
        
        # Get existing positions to avoid duplicates
        existing_positions = session_manager.get_positions(session_id)
        existing_bets = set()
        for pos in existing_positions:
            if pos['status'] in ['pending', 'open']:
                existing_bets.add((pos['match_id'], pos['bet_on']))
        
        # Check current exposure
        open_positions = [p for p in existing_positions if p['status'] in ['pending', 'open']]
        current_exposure = sum(float(p['stake']) for p in open_positions)
        exposure_pct = (current_exposure / STRATEGY_CONFIG['bankroll'] * 100) if STRATEGY_CONFIG['bankroll'] else 0
        
        # Stop if over-leveraged
        MAX_EXPOSURE_PCT = 30  # Maximum 30% exposure
        if exposure_pct > MAX_EXPOSURE_PCT:
            logger.warning(f"⚠️ Exposure at {exposure_pct:.1f}% - skipping new trades (max: {MAX_EXPOSURE_PCT}%)")
            return {'success': True, 'trades_made': 0, 'reason': 'exposure_limit'}
        
        # Get market data
        markets, signals, stats, chunks = get_market_data()
        
        # Track exposure per game to avoid over-concentration
        game_exposures = {}
        for pos in open_positions:
            game_key = pos['match_id']
            if game_key not in game_exposures:
                game_exposures[game_key] = 0
            game_exposures[game_key] += float(pos['stake'])
        
        # Find trades to execute
        trades_to_execute = []
        max_per_game = STRATEGY_CONFIG['bankroll'] * STRATEGY_CONFIG['cap_per_game']  # 2% per game
        
        for i, market in enumerate(markets):
            signal = signals[i]
            market_id = market.get('market_id', market.get('source_id', ''))
            
            # Check current exposure for this game
            game_exposure = game_exposures.get(market_id, 0)
            if game_exposure >= max_per_game:
                continue  # Skip if already at max exposure for this game
            
            for outcome in ['home', 'draw', 'away']:
                # Skip if we already have a bet on this market/outcome
                if (market_id, outcome) in existing_bets:
                    continue
                    
                stake = signal.get(f'{outcome}_stake', 0)
                if stake > 0:
                    # Apply per-bet cap
                    stake = min(stake, STRATEGY_CONFIG['bankroll'] * STRATEGY_CONFIG['cap_per_bet'])
                    
                    # Check if adding this trade would exceed per-game limit
                    if game_exposure + stake > max_per_game:
                        stake = max_per_game - game_exposure
                        if stake < STRATEGY_CONFIG['min_bet']:
                            continue
                    
                    # Check if adding this trade would exceed total exposure limit
                    new_exposure_pct = ((current_exposure + stake) / STRATEGY_CONFIG['bankroll'] * 100)
                    if new_exposure_pct > MAX_EXPOSURE_PCT:
                        continue
                        
                    trades_to_execute.append({
                        'match_id': market_id,
                        'sport': market.get('sport', 'Soccer'),
                        'home_team': market.get('home_team', ''),
                        'away_team': market.get('away_team', ''),
                        'bet_type': 'moneyline',
                        'bet_on': outcome,
                        'odds': signal.get(f'{outcome}_odds', 0),
                        'stake': stake,
                        'signal_name': 'enhanced_edge',
                        'signal_value': signal.get(f'{outcome}_implied_prob', 0),
                        'edge': signal.get(f'{outcome}_edge', 0),
                        'kickoff_time': market.get('maturity_date')
                    })
                    current_exposure += stake
                    game_exposure += stake
                    game_exposures[market_id] = game_exposure
        
        # Record trades
        if trades_to_execute:
            session_manager.record_trades(session['session_id'], trades_to_execute)
            
            # Log each trade
            for trade in trades_to_execute:
                socketio.emit('activity', {
                    'type': 'trade',
                    'message': f"Placed {trade['bet_on']} ${trade['stake']:.2f} on {trade['home_team']} vs {trade['away_team']} @ {trade['odds']:.2f}",
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
            # Check if stop loss is active
            stop_status = stop_loss_manager.get_stop_status()
            if stop_status['is_stopped']:
                logger.info(f"⛔ Trading stopped: {stop_status['stop_reason']}")
                time.sleep(60)  # Check every minute when stopped
                continue
                
            logger.info("⚡ Portfolio optimization cycle starting...")
            result = execute_portfolio_trades()
            
            if result.get("success"):
                trades_made = result.get("trades_made", 0)
                if trades_made > 0:
                    logger.info(f"✅ Portfolio rebalanced: {trades_made} trades, "
                               f"portfolio size: {result.get('portfolio_size', 0)}")
                else:
                    reason = result.get('reason', 'No rebalancing needed')
                    logger.info(f"📊 Portfolio optimization complete: {reason}")
            else:
                logger.warning(f"⚠️ Portfolio optimization failed: {result.get('error', 'Unknown error')}")
                
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
    # Get current session
    session_id = session_manager.get_current_session()
    if session_id:
        # Start stop loss monitoring
        stop_loss_manager.start_monitoring(session_id)
        stop_loss_manager.reset_daily_tracking(session_id)
        logger.info("Started stop loss monitoring")
        
    # Start background threads
    bg_thread = threading.Thread(target=background_updates, daemon=True)
    bg_thread.start()
    
    trading_thread = threading.Thread(target=paper_trading_background_loop, daemon=True)
    trading_thread.start()
    
    # Start server
    logger.info("Starting Ominari Dashboard on http://localhost:8888")
    logger.info("Using direct PostgreSQL connection - no SQLite dependencies")
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)