#!/usr/bin/env python3
"""Quick test to verify database and start a simple dashboard"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from flask import Flask, jsonify, request
from database_v2 import db_manager
from models import Market, Odd
from datetime import datetime, timezone
import json

app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ominari Dashboard</title>
        <style>
            body { font-family: monospace; background: #000; color: #0f0; padding: 20px; }
            .market { border: 1px solid #0f0; padding: 10px; margin: 10px 0; }
            .stats { background: #111; padding: 20px; margin-bottom: 20px; }
            h1 { color: #0ff; }
            .filter { margin: 20px 0; }
            select { background: #222; color: #0f0; border: 1px solid #0f0; padding: 5px; }
        </style>
    </head>
    <body>
        <h1>Ominari Trading Dashboard</h1>
        <div class="stats">
            <h2>Database Stats</h2>
            <div id="stats">Loading...</div>
        </div>
        <div class="filter">
            <label>Sport Filter: </label>
            <select id="sport-filter" onchange="loadMarkets()">
                <option value="all">All Sports</option>
                <option value="Soccer">Soccer</option>
                <option value="Tennis">Tennis</option>
                <option value="Baseball">Baseball</option>
                <option value="Hockey">Hockey</option>
                <option value="Fighting">Fighting</option>
            </select>
        </div>
        <h2>Active Markets</h2>
        <div id="markets">Loading...</div>
        <script>
            async function loadStats() {
                const resp = await fetch('/api/stats');
                const data = await resp.json();
                document.getElementById('stats').innerHTML = `
                    <p>Total Markets: ${data.total_markets.toLocaleString()}</p>
                    <p>Active Markets: ${data.active_markets.toLocaleString()}</p>
                    <p>Markets by Sport: ${JSON.stringify(data.sports)}</p>
                `;
            }
            
            async function loadMarkets() {
                const sport = document.getElementById('sport-filter').value;
                const resp = await fetch('/api/markets?sport=' + sport);
                const markets = await resp.json();
                
                const html = markets.map(m => `
                    <div class="market">
                        <h3>${m.home_team} vs ${m.away_team}</h3>
                        <p>Sport: ${m.sport} | Time: ${m.time_until} | Status: ${m.status}</p>
                        <p>Odds: Home=${m.home_odds || '-'} Draw=${m.draw_odds || '-'} Away=${m.away_odds || '-'}</p>
                    </div>
                `).join('');
                
                document.getElementById('markets').innerHTML = html || '<p>No markets found</p>';
            }
            
            loadStats();
            loadMarkets();
            setInterval(loadMarkets, 30000);
        </script>
    </body>
    </html>
    '''

@app.route('/api/stats')
def api_stats():
    """Get database statistics"""
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        active = db.query(Market).filter(Market.is_finished == False).count()
        
        # Count by sport
        sports = {}
        from sqlalchemy import func
        sport_counts = db.query(Market.sport, func.count()).filter(
            Market.is_finished == False
        ).group_by(Market.sport).all()
        
        for sport, count in sport_counts:
            sports[sport] = count
            
        return jsonify({
            'total_markets': total,
            'active_markets': active,
            'sports': sports
        })

@app.route('/api/markets')
def api_markets():
    """Get markets with optional sport filter"""
    sport_filter = request.args.get('sport', 'all')
    markets = []
    
    try:
        with db_manager.get_db_session() as db:
            query = db.query(Market).filter(Market.is_finished == False)
            
            if sport_filter and sport_filter != 'all':
                query = query.filter(Market.sport == sport_filter)
            
            active_markets = query.order_by(Market.maturity_date).limit(100).all()
            
            for market in active_markets:
                # Get odds
                odds = db.query(Odd).filter(
                    Odd.source_id == market.source_id
                ).order_by(Odd.updated_at.desc()).limit(3).all()
                
                home_odds = None
                draw_odds = None
                away_odds = None
                
                for odd in odds:
                    outcome_lower = str(odd.outcome).lower() if odd.outcome else ''
                    if 'home' in outcome_lower:
                        home_odds = odd.decimal_odds
                    elif 'away' in outcome_lower:
                        away_odds = odd.decimal_odds
                    elif 'draw' in outcome_lower:
                        draw_odds = odd.decimal_odds
                
                # Calculate time
                now_utc = datetime.now(timezone.utc)
                if market.maturity_date:
                    if market.maturity_date.tzinfo is None:
                        maturity_aware = market.maturity_date.replace(tzinfo=timezone.utc)
                    else:
                        maturity_aware = market.maturity_date
                    
                    delta = maturity_aware - now_utc
                    if delta.total_seconds() > 0:
                        hours = int(delta.total_seconds() // 3600)
                        mins = int((delta.total_seconds() % 3600) // 60)
                        if hours > 24:
                            time_until = f"{hours // 24}d {hours % 24}h"
                        elif hours > 0:
                            time_until = f"{hours}h {mins}m"
                        else:
                            time_until = f"{mins}m"
                        status = "Upcoming"
                    else:
                        time_until = "Started"
                        status = "Live"
                else:
                    time_until = "Unknown"
                    status = "Unknown"
                
                markets.append({
                    'id': market.source_id,
                    'home_team': market.home_team,
                    'away_team': market.away_team,
                    'sport': market.sport,
                    'time_until': time_until,
                    'status': status,
                    'home_odds': home_odds,
                    'draw_odds': draw_odds,
                    'away_odds': away_odds
                })
                
    except Exception as e:
        print(f"Error: {e}")
        
    return jsonify(markets)

if __name__ == '__main__':
    print("🚀 Starting Ominari Dashboard on http://localhost:8888")
    print("🔍 Sport filtering is available via dropdown")
    app.run(host='0.0.0.0', port=8888, debug=False)