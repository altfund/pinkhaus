#!/usr/bin/env python3
"""
Soccer-only dashboard for real Overtime markets
"""

import sqlite3
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

DB_PATH = "sport_odds.db"

class SoccerDashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.generate_dashboard().encode())
        elif self.path == '/api/markets':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            markets = self.get_soccer_markets()
            self.wfile.write(json.dumps(markets).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress request logging
        pass
    
    def get_soccer_markets(self):
        """Get only soccer markets from database."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get soccer markets with odds
        cursor.execute('''
            SELECT DISTINCT m.source_id, m.home_team, m.away_team, m.league_name, 
                   m.maturity_date, m.sport
            FROM market m
            WHERE m.source = 'api_live_real'
            AND (m.sport = 'Soccer' 
                 OR m.home_team LIKE '%FC%' 
                 OR m.home_team LIKE '%United%'
                 OR m.home_team LIKE '%City%'
                 OR m.away_team LIKE '%FC%'
                 OR m.away_team LIKE '%United%' 
                 OR m.away_team LIKE '%City%')
            ORDER BY m.maturity_date
            LIMIT 50
        ''')
        
        markets = []
        for source_id, home, away, league, maturity, sport in cursor.fetchall():
            # Get odds for this market
            cursor.execute('''
                SELECT outcome, decimal_odds, american_odds
                FROM odd
                WHERE source_id = ?
                ORDER BY outcome
            ''', (source_id,))
            
            odds = {}
            for outcome, decimal, american in cursor.fetchall():
                odds[outcome] = {
                    'decimal': decimal,
                    'american': int(american)
                }
            
            markets.append({
                'home': home,
                'away': away,
                'league': league,
                'sport': sport,
                'maturity': maturity,
                'odds': odds
            })
        
        conn.close()
        return markets
    
    def generate_dashboard(self):
        """Generate HTML dashboard."""
        markets = self.get_soccer_markets()
        
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>⚽ Soccer Markets - Real Overtime Data</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: #2ecc71;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .stats {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            justify-content: center;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .markets-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }
        .market-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border-left: 4px solid #2ecc71;
        }
        .teams {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        .league {
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        .date {
            color: #95a5a6;
            font-size: 14px;
            margin-bottom: 15px;
        }
        .odds-container {
            display: flex;
            gap: 10px;
        }
        .odd-box {
            flex: 1;
            background: #3498db;
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .odd-outcome {
            font-size: 12px;
            opacity: 0.9;
        }
        .odd-value {
            font-size: 16px;
            font-weight: bold;
        }
        .refresh-btn {
            background: #2ecc71;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚽ Real Soccer Markets Dashboard</h1>
        <p>Live data from Overtime V2 API - Soccer matches only</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <h3>Soccer Markets</h3>
            <h2>''' + str(len(markets)) + '''</h2>
        </div>
        <div class="stat-card">
            <h3>Data Source</h3>
            <h2>Real API</h2>
        </div>
        <div class="stat-card">
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
        </div>
    </div>
    
    <div class="markets-grid">
'''
        
        for market in markets:
            # Format date
            try:
                date_obj = datetime.fromisoformat(market['maturity'].replace('Z', '+00:00'))
                formatted_date = date_obj.strftime('%a %d %b, %H:%M UTC')
            except:
                formatted_date = market['maturity']
            
            html += f'''
        <div class="market-card">
            <div class="teams">{market['home']} vs {market['away']}</div>
            <div class="league">🏆 {market['league'] or 'Soccer League'}</div>
            <div class="date">📅 {formatted_date}</div>
            <div class="odds-container">
'''
            
            # Add odds
            for outcome, odd_data in market['odds'].items():
                decimal = odd_data['decimal']
                american = odd_data['american']
                american_str = f"{american:+d}" if american > 0 else str(american)
                
                html += f'''
                <div class="odd-box">
                    <div class="odd-outcome">{outcome.upper()}</div>
                    <div class="odd-value">{decimal}</div>
                    <div class="odd-outcome">{american_str}</div>
                </div>
'''
            
            html += '''
            </div>
        </div>
'''
        
        html += '''
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
'''
        return html

def main():
    print("⚽ Starting Soccer Dashboard Server")
    print("=" * 50)
    
    # First, show how many soccer markets we have
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT COUNT(*) 
        FROM market 
        WHERE source = 'api_live_real'
        AND (sport = 'Soccer' 
             OR home_team LIKE '%FC%' 
             OR home_team LIKE '%United%'
             OR home_team LIKE '%City%')
    ''')
    
    soccer_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM market WHERE source = 'api_live_real'")
    total_count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"📊 Found {soccer_count} soccer markets out of {total_count} total")
    print(f"🌐 Starting dashboard at http://localhost:8888/")
    print(f"✅ Dashboard will show soccer matches only")
    print(f"🔄 Auto-refreshes every 30 seconds")
    print()
    print("Press Ctrl+C to stop")
    
    # Start server
    server = HTTPServer(('localhost', 8888), SoccerDashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")

if __name__ == "__main__":
    main()