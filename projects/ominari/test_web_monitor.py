#!/usr/bin/env python3
"""
Test web_monitor.py startup issues
"""

import os
import sys
import traceback

def test_web_monitor_imports():
    """Test if web_monitor.py imports correctly"""
    print("🧪 Testing web_monitor.py imports...")
    
    # Set environment first
    os.environ.update({
        'PG_HOST': 'localhost',
        'PG_PORT': '5435',
        'PG_USER': 'ominari_user',
        'PG_PASSWORD': 'ominari_2025_secure',
        'PG_DB': 'ominari_production'
    })
    
    try:
        # Test basic imports
        print("  Testing Flask import...")
        from flask import Flask
        print("  ✅ Flask import OK")
        
        print("  Testing database imports...")
        from database_v2 import db_manager
        print("  ✅ Database imports OK")
        
        print("  Testing models...")
        from models import Market, Odd
        print("  ✅ Models import OK")
        
        print("  Testing PostgreSQL connection...")
        with db_manager.get_db_session() as db:
            count = db.query(Market).count()
            print(f"  ✅ Database connection OK: {count} markets")
        
        # Try importing the web monitor
        print("  Testing web_monitor module...")
        import web_monitor
        print("  ✅ web_monitor import OK")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def create_minimal_working_dashboard():
    """Create a minimal dashboard that definitely works"""
    print("🛠️ Creating minimal working dashboard...")
    
    minimal_app = '''
import os
from flask import Flask, render_template_string
from sqlalchemy import create_engine, text

os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5435',
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
'''
    
    with open('minimal_dashboard.py', 'w') as f:
        f.write(minimal_app)
    
    print("✅ Created minimal_dashboard.py")
    return True

def main():
    print("=" * 60)
    print("🔧 Testing Web Monitor Issues")
    print("=" * 60)
    
    # Test imports
    if test_web_monitor_imports():
        print("\n✅ All imports working - issue might be in Flask app configuration")
    else:
        print("\n❌ Import issues detected")
    
    # Create minimal working version
    create_minimal_working_dashboard()
    
    print("\n🚀 Try running: python3 minimal_dashboard.py")
    print("Then visit: http://localhost:8888")

if __name__ == "__main__":
    main()