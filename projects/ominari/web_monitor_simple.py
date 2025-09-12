#!/usr/bin/env python3
"""
Simple Ominari Web Monitor - Clean, stable dashboard
Shows markets, signals, and positions without complex features
"""

import logging
from datetime import datetime, timezone
from flask import Flask, render_template_string, jsonify
from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import func, desc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

SIMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Ominari Trading System</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="30">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Consolas', 'Monaco', monospace;
            background: #0a0a0a;
            color: #00ff00;
            padding: 20px;
        }
        .header {
            background: #111;
            padding: 20px;
            border: 1px solid #333;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .container {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }
        .section {
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 20px;
        }
        .section-title {
            font-size: 1.2em;
            color: #00ffff;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }
        .market-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 10px;
        }
        .market-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .market-teams {
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
            margin-top: 10px;
        }
        .odd-box {
            background: #222;
            border: 1px solid #444;
            padding: 8px;
            text-align: center;
            border-radius: 4px;
        }
        .odd-label {
            font-size: 0.8em;
            color: #888;
        }
        .odd-value {
            font-size: 1.2em;
            color: #00ffff;
            font-weight: bold;
        }
        .signal-box {
            background: #1a2a1a;
            border: 1px solid #2a4a2a;
            padding: 10px;
            margin-top: 10px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        .stat-box {
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 15px;
            text-align: center;
            border-radius: 5px;
        }
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #00ff00;
            margin-bottom: 5px;
        }
        .stat-label {
            color: #888;
            font-size: 0.9em;
        }
        .info-item {
            padding: 10px;
            border-bottom: 1px solid #222;
        }
        .info-label {
            color: #888;
            font-size: 0.9em;
        }
        .info-value {
            color: #fff;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Ominari Trading System - V1 Monitor</h1>
        <div>
            <span style="margin-right: 20px;">⏰ {{ current_time }}</span>
            <span>📊 Markets: {{ total_markets }}</span>
        </div>
    </div>
    
    <div class="container">
        <!-- Markets Section -->
        <div class="section">
            <div class="section-title">⚽ Active Soccer Markets</div>
            <div id="markets-container">
                {% for market in markets %}
                <div class="market-card">
                    <div class="market-header">
                        <div class="market-teams">{{ market.home_team }} vs {{ market.away_team }}</div>
                        <div class="market-time">{{ market.time_until }}</div>
                    </div>
                    
                    <div class="odds-row">
                        <div class="odd-box">
                            <div class="odd-label">Home</div>
                            <div class="odd-value">{{ "%.2f"|format(market.home_odds) if market.home_odds else '-' }}</div>
                        </div>
                        <div class="odd-box">
                            <div class="odd-label">Draw</div>
                            <div class="odd-value">{{ "%.2f"|format(market.draw_odds) if market.draw_odds else '-' }}</div>
                        </div>
                        <div class="odd-box">
                            <div class="odd-label">Away</div>
                            <div class="odd-value">{{ "%.2f"|format(market.away_odds) if market.away_odds else '-' }}</div>
                        </div>
                    </div>
                    
                    {% if market.signal %}
                    <div class="signal-box">
                        📡 {{ market.signal }}
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Stats Section -->
        <div>
            <div class="section" style="margin-bottom: 20px;">
                <div class="section-title">📈 System Stats</div>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value">{{ active_markets }}</div>
                        <div class="stat-label">Active Markets</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{{ soccer_markets }}</div>
                        <div class="stat-label">Soccer Matches</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">✅</div>
                        <div class="stat-label">API Status</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{{ "%.1f"|format(avg_odds) }}</div>
                        <div class="stat-label">Avg Odds</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">ℹ️ System Info</div>
                <div class="info-item">
                    <div class="info-label">Configuration</div>
                    <div class="info-value">V1 - API Data + Blockchain Trades</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Update Frequency</div>
                    <div class="info-value">Every 5 minutes</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Trading Mode</div>
                    <div class="info-value">Paper Trading</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Last Update</div>
                    <div class="info-value">{{ last_update }}</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

def get_market_data():
    """Get market data for display."""
    markets = []
    stats = {
        'total_markets': 0,
        'active_markets': 0,
        'soccer_markets': 0,
        'avg_odds': 0
    }
    
    try:
        with db_manager.get_db_session() as db:
            # Get active soccer markets
            active_markets = db.query(Market).filter(
                Market.sport == 'Soccer',
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).order_by(Market.maturity_date).limit(10).all()
            
            stats['soccer_markets'] = len(active_markets)
            total_odds = []
            
            for market in active_markets:
                # Get latest odds
                odds = db.query(Odd).filter(
                    Odd.source_id == market.source_id
                ).order_by(desc(Odd.updated_at)).limit(3).all()
                
                home_odds = None
                draw_odds = None
                away_odds = None
                
                for odd in odds:
                    if odd.outcome == 'option_1':
                        home_odds = odd.decimal_odds
                    elif odd.outcome == 'option_3':
                        draw_odds = odd.decimal_odds
                    elif odd.outcome == 'option_2':
                        away_odds = odd.decimal_odds
                    
                    if odd.decimal_odds:
                        total_odds.append(odd.decimal_odds)
                
                # Calculate time until
                time_until = "Started"
                if market.maturity_date:
                    delta = market.maturity_date - datetime.now(timezone.utc)
                    if delta.total_seconds() > 0:
                        hours = int(delta.total_seconds() // 3600)
                        mins = int((delta.total_seconds() % 3600) // 60)
                        if hours > 24:
                            time_until = f"{hours // 24}d {hours % 24}h"
                        elif hours > 0:
                            time_until = f"{hours}h {mins}m"
                        else:
                            time_until = f"{mins}m"
                
                # Simple signal
                signal = None
                if home_odds and draw_odds and away_odds:
                    # Check for value
                    implied_total = (1/home_odds + 1/draw_odds + 1/away_odds) if home_odds and draw_odds and away_odds else 0
                    if implied_total > 1.05:
                        margin = (implied_total - 1) * 100
                        signal = f"High margin: {margin:.1f}% - Look for value bets"
                
                markets.append({
                    'home_team': market.home_team,
                    'away_team': market.away_team,
                    'time_until': time_until,
                    'home_odds': home_odds,
                    'draw_odds': draw_odds,
                    'away_odds': away_odds,
                    'signal': signal
                })
            
            # Calculate stats
            stats['active_markets'] = db.query(func.count(Market.id)).filter(
                Market.is_finished == False
            ).scalar() or 0
            
            stats['total_markets'] = db.query(func.count(Market.id)).scalar() or 0
            
            if total_odds:
                stats['avg_odds'] = sum(total_odds) / len(total_odds)
            else:
                stats['avg_odds'] = 0
                
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
    
    return markets, stats

@app.route('/')
def index():
    """Main dashboard."""
    markets, stats = get_market_data()
    
    return render_template_string(SIMPLE_HTML,
        markets=markets,
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_markets=stats['total_markets'],
        active_markets=stats['active_markets'],
        soccer_markets=stats['soccer_markets'],
        avg_odds=stats['avg_odds'],
        last_update=datetime.now().strftime("%H:%M:%S")
    )

@app.route('/api/status')
def api_status():
    """Simple API endpoint."""
    markets, stats = get_market_data()
    return jsonify({
        'status': 'ok',
        'markets': len(markets),
        'stats': stats,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Simple Ominari Monitor Starting...")
    print("📊 Dashboard: http://localhost:8889")
    print("🔄 Auto-refresh every 30 seconds")
    
    app.run(host='0.0.0.0', port=8889, debug=False)