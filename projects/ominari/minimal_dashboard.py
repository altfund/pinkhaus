
import os
from flask import Flask, render_template_string
from sqlalchemy import create_engine, text

os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999',
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})

app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ominari Trading Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; background: #0a0a0a; color: #00ff00; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .market { background: #1a1a1a; border: 1px solid #333; padding: 15px; margin: 10px 0; }
            .odds { color: #ffaa00; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Ominari Trading Dashboard</h1>
                <p>✅ Connected to PostgreSQL via Flox</p>
                <p>🔗 Real Blockchain Data from Optimism & Arbitrum</p>
            </div>
            <div id="markets">
                {{ markets_html|safe }}
            </div>
        </div>
    </body>
    </html>
    """, markets_html=get_markets())

def get_markets():
    try:
        engine = create_engine(f"postgresql://{os.environ['PG_USER']}:{os.environ['PG_PASSWORD']}@{os.environ['PG_HOST']}:{os.environ['PG_PORT']}/{os.environ['PG_DB']}")
        
        with engine.connect() as conn:
            markets = conn.execute(text("""
                SELECT m.source_id, m.home_team, m.away_team, m.sport, m.source,
                       COUNT(o.source_id) as odds_count
                FROM market m
                LEFT JOIN odd o ON m.source_id = o.source_id
                WHERE m.source LIKE 'blockchain_%'
                GROUP BY m.source_id, m.home_team, m.away_team, m.sport, m.source
                LIMIT 20
            """)).fetchall()
            
            if not markets:
                return "<div class='market'>⚠️ No blockchain markets found in database</div>"
            
            html = f"<h2>📊 {len(markets)} Blockchain Markets</h2>"
            for market in markets:
                html += f"""
                <div class="market">
                    <strong>{market[1]} vs {market[2]}</strong><br>
                    <span class="odds">Sport: {market[3]} | Source: {market[4]} | Odds: {market[5]}</span>
                </div>
                """
            return html
            
    except Exception as e:
        return f"<div class='market'>❌ Database Error: {e}</div>"

if __name__ == "__main__":
    print("🚀 Starting Minimal Ominari Dashboard")
    print("📍 http://localhost:8888")
    app.run(host='0.0.0.0', port=8888, debug=False)
