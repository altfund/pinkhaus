#!/usr/bin/env python3
"""
Working Ominari Dashboard with PostgreSQL on port 5999
"""

import os
from flask import Flask, render_template_string, jsonify
from sqlalchemy import create_engine, text

# Set environment for new PostgreSQL port
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999',
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})

app = Flask(__name__)

def get_db_engine():
    """Get database engine"""
    return create_engine(f"postgresql://{os.environ['PG_USER']}:{os.environ['PG_PASSWORD']}@{os.environ['PG_HOST']}:{os.environ['PG_PORT']}/{os.environ['PG_DB']}")

@app.route('/')
def home():
    """Main dashboard page"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 Ominari Trading Dashboard</title>
        <meta charset="utf-8">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'Consolas', 'Monaco', monospace;
                background: #0a0a0a;
                color: #00ff00;
                padding: 20px;
                line-height: 1.6;
            }
            .container { max-width: 1400px; margin: 0 auto; }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 8px;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }
            .stat-card {
                background: #1a1a1a;
                border: 1px solid #333;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-value { font-size: 24px; color: #ffaa00; font-weight: bold; }
            .stat-label { color: #888; margin-top: 5px; }
            .market {
                background: #1a1a1a;
                border: 1px solid #333;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
            }
            .market-header { font-size: 18px; color: #00ff00; margin-bottom: 8px; }
            .market-details { color: #ccc; margin: 5px 0; }
            .odds { color: #ffaa00; font-weight: bold; }
            .source { color: #888; font-size: 12px; }
            .refresh-btn {
                background: #333;
                color: #00ff00;
                border: 1px solid #555;
                padding: 10px 20px;
                cursor: pointer;
                border-radius: 4px;
                margin: 10px 0;
            }
            .refresh-btn:hover { background: #555; }
            .status { color: #00ff00; }
        </style>
        <script>
            function refreshData() {
                location.reload();
            }
            
            // Auto-refresh every 30 seconds
            setInterval(refreshData, 30000);
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Ominari Trading Dashboard</h1>
                <div class="status">✅ Connected to PostgreSQL via Flox (Port 5999)</div>
                <div class="status">🔗 Real Blockchain Data from Optimism & Arbitrum</div>
                <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
            </div>
            
            <div class="stats">
                {{ stats_html|safe }}
            </div>
            
            <div id="markets">
                <h2>📊 Blockchain Markets</h2>
                {{ markets_html|safe }}
            </div>
        </div>
    </body>
    </html>
    """, stats_html=get_stats(), markets_html=get_markets())

@app.route('/api/stats')
def api_stats():
    """API endpoint for stats"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            total_markets = conn.execute(text("SELECT COUNT(*) FROM market")).scalar()
            blockchain_markets = conn.execute(text("SELECT COUNT(*) FROM market WHERE source LIKE 'blockchain_%'")).scalar()
            soccer_markets = conn.execute(text("SELECT COUNT(*) FROM market WHERE source LIKE 'blockchain_%' AND sport = 'Soccer'")).scalar()
            total_odds = conn.execute(text("SELECT COUNT(*) FROM odd WHERE source LIKE 'blockchain_%'")).scalar()
            
            return jsonify({
                'total_markets': total_markets,
                'blockchain_markets': blockchain_markets,
                'soccer_markets': soccer_markets,
                'total_odds': total_odds,
                'database': 'PostgreSQL',
                'port': os.environ['PG_PORT']
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_stats():
    """Get database statistics"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            total_markets = conn.execute(text("SELECT COUNT(*) FROM market")).scalar()
            blockchain_markets = conn.execute(text("SELECT COUNT(*) FROM market WHERE source LIKE 'blockchain_%'")).scalar()
            soccer_markets = conn.execute(text("SELECT COUNT(*) FROM market WHERE source LIKE 'blockchain_%' AND sport = 'Soccer'")).scalar()
            total_odds = conn.execute(text("SELECT COUNT(*) FROM odd WHERE source LIKE 'blockchain_%'")).scalar()
            
            return f"""
            <div class="stat-card">
                <div class="stat-value">{total_markets}</div>
                <div class="stat-label">Total Markets</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{blockchain_markets}</div>
                <div class="stat-label">Blockchain Markets</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{soccer_markets}</div>
                <div class="stat-label">Soccer Markets</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_odds}</div>
                <div class="stat-label">Odds Records</div>
            </div>
            """
            
    except Exception as e:
        return f'<div class="stat-card"><div class="stat-value">❌</div><div class="stat-label">DB Error: {e}</div></div>'

def get_markets():
    """Get market data"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            markets = conn.execute(text("""
                SELECT m.source_id, m.home_team, m.away_team, m.sport, m.source,
                       m.league_name, COUNT(o.source_id) as odds_count,
                       MAX(o.decimal_odds) as max_odds
                FROM market m
                LEFT JOIN odd o ON m.source_id = o.source_id
                WHERE m.source LIKE 'blockchain_%'
                GROUP BY m.source_id, m.home_team, m.away_team, m.sport, m.source, m.league_name
                ORDER BY odds_count DESC
                LIMIT 50
            """)).fetchall()
            
            if not markets:
                return "<div class='market'>⚠️ No blockchain markets found in database</div>"
            
            html = ""
            for market in markets:
                html += f"""
                <div class="market">
                    <div class="market-header">{market[1]} vs {market[2]}</div>
                    <div class="market-details">
                        <span class="odds">Sport: {market[3]}</span> | 
                        <span>League: {market[5] or 'N/A'}</span> | 
                        <span class="odds">{market[6]} odds</span>
                        {f" | Max Odds: {market[7]:.2f}" if market[7] else ""}
                    </div>
                    <div class="source">Source: {market[4]} | ID: {market[0][:30]}...</div>
                </div>
                """
            return html
            
    except Exception as e:
        return f"<div class='market'>❌ Database Error: {e}</div>"

if __name__ == "__main__":
    print("🚀 Starting Ominari Dashboard")
    print("📍 http://localhost:8888")
    print("📊 Connected to PostgreSQL on port 5999")
    print("🔗 Real blockchain data from Optimism & Arbitrum")
    print("Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=8888, debug=False, threaded=True)