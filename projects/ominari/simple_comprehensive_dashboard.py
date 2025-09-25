#!/usr/bin/env python3
"""
Simple Comprehensive Dashboard - Shows all Overtime data
No external dependencies except standard library
"""

import sqlite3
import json
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

DB_PATH = "sport_odds.db"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Ominari Comprehensive Dashboard</title>
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
            border: 2px solid #00ff00;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .header h1 {
            color: #00ffff;
            margin-bottom: 10px;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .stat-box {
            background: #111;
            border: 1px solid #333;
            padding: 20px;
            text-align: center;
            border-radius: 5px;
        }
        
        .stat-value {
            font-size: 2em;
            color: #00ffff;
            font-weight: bold;
        }
        
        .stat-label {
            color: #888;
            margin-top: 5px;
        }
        
        .section {
            background: #111;
            border: 1px solid #333;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        
        .section h2 {
            color: #00ffff;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }
        
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th {
            background: #222;
            padding: 10px;
            text-align: left;
            color: #00ffff;
            border-bottom: 2px solid #444;
        }
        
        td {
            padding: 8px;
            border-bottom: 1px solid #222;
        }
        
        tr:hover {
            background: #1a1a1a;
        }
        
        .market-card {
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        
        .market-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .market-teams {
            font-weight: bold;
            color: #fff;
        }
        
        .market-sport {
            background: #222;
            padding: 2px 8px;
            border-radius: 3px;
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
            padding: 10px;
            text-align: center;
            border-radius: 3px;
        }
        
        .odd-label {
            font-size: 0.8em;
            color: #888;
        }
        
        .odd-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #00ff00;
        }
        
        .source-api {
            color: #00ccff;
        }
        
        .source-blockchain {
            color: #ff9900;
        }
        
        .sport-bar {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .sport-name {
            width: 150px;
        }
        
        .sport-bar-fill {
            height: 20px;
            background: #00ff00;
            border-radius: 3px;
        }
        
        .sport-count {
            margin-left: 10px;
            color: #888;
        }
        
        .timestamp {
            color: #666;
            font-size: 0.9em;
            text-align: right;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Ominari Comprehensive Trading Dashboard</h1>
        <div>Real-time data from Overtime API and Blockchain</div>
    </div>
    
    <div class="stats">
        <div class="stat-box">
            <div class="stat-value">{total_markets}</div>
            <div class="stat-label">Total Markets</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{api_markets}</div>
            <div class="stat-label">API Markets</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{blockchain_markets}</div>
            <div class="stat-label">Blockchain Markets</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{active_markets}</div>
            <div class="stat-label">Active Markets</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{markets_with_odds}</div>
            <div class="stat-label">Markets w/ Odds</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{unique_sports}</div>
            <div class="stat-label">Sports</div>
        </div>
    </div>
    
    <div class="grid">
        <div class="section">
            <h2>🏆 Sport Distribution</h2>
            {sport_distribution}
        </div>
        
        <div class="section">
            <h2>📊 Data Sources</h2>
            <table>
                <thead>
                    <tr>
                        <th>Source</th>
                        <th>Markets</th>
                        <th>Type</th>
                    </tr>
                </thead>
                <tbody>
                    {data_sources}
                </tbody>
            </table>
        </div>
    </div>
    
    <div class="section">
        <h2>🎲 Live Markets (Next 24 Hours)</h2>
        {live_markets}
    </div>
    
    <div class="grid">
        <div class="section">
            <h2>⛓️ Recent Blockchain Markets</h2>
            {blockchain_markets}
        </div>
        
        <div class="section">
            <h2>📈 Markets with Best Odds</h2>
            {best_odds}
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Sport Analysis</h2>
        <table>
            <thead>
                <tr>
                    <th>Sport</th>
                    <th>Total</th>
                    <th>API</th>
                    <th>Blockchain</th>
                    <th>With Odds</th>
                    <th>Avg Odds</th>
                </tr>
            </thead>
            <tbody>
                {sport_analysis}
            </tbody>
        </table>
    </div>
    
    <div class="timestamp">
        Last updated: {timestamp} UTC | Auto-refresh every 30 seconds
    </div>
</body>
</html>
"""

def get_dashboard_data():
    """Get all data for dashboard"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    data = {}
    
    # Basic stats
    cursor.execute("SELECT COUNT(*) FROM market")
    data['total_markets'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM market WHERE source = 'api_live_real'")
    data['api_markets'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM market WHERE source LIKE '%blockchain%'")
    data['blockchain_markets'] = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) FROM market 
        WHERE is_finished = 0 
        AND (maturity_date > datetime('now') OR maturity_date IS NULL)
    """)
    data['active_markets'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT source_id) FROM odd")
    data['markets_with_odds'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT sport) FROM market")
    data['unique_sports'] = cursor.fetchone()[0]
    
    # Sport distribution
    cursor.execute("""
        SELECT sport, COUNT(*) as count
        FROM market
        GROUP BY sport
        ORDER BY count DESC
        LIMIT 10
    """)
    
    sport_html = ""
    max_count = 0
    sports = cursor.fetchall()
    if sports:
        max_count = sports[0][1]
    
    for sport, count in sports:
        percentage = (count / max_count * 100) if max_count > 0 else 0
        sport_html += f"""
        <div class="sport-bar">
            <div class="sport-name">{sport}</div>
            <div class="sport-bar-fill" style="width: {percentage}%"></div>
            <div class="sport-count">{count:,}</div>
        </div>
        """
    data['sport_distribution'] = sport_html
    
    # Data sources
    cursor.execute("""
        SELECT source, COUNT(*) as count
        FROM market
        GROUP BY source
        ORDER BY count DESC
    """)
    
    sources_html = ""
    for source, count in cursor.fetchall():
        source_type = "Blockchain" if "blockchain" in source else "API"
        source_class = "source-blockchain" if "blockchain" in source else "source-api"
        sources_html += f"""
        <tr>
            <td class="{source_class}">{source}</td>
            <td>{count:,}</td>
            <td>{source_type}</td>
        </tr>
        """
    data['data_sources'] = sources_html
    
    # Live markets
    cursor.execute("""
        SELECT m.home_team, m.away_team, m.sport, m.source,
               o1.decimal_odds as home_odds, o2.decimal_odds as away_odds
        FROM market m
        LEFT JOIN odd o1 ON m.source_id = o1.source_id AND o1.outcome = 'home'
        LEFT JOIN odd o2 ON m.source_id = o2.source_id AND o2.outcome = 'away'
        WHERE m.is_finished = 0
        AND m.maturity_date > datetime('now')
        AND m.maturity_date < datetime('now', '+24 hours')
        ORDER BY m.maturity_date
        LIMIT 10
    """)
    
    markets_html = ""
    for row in cursor.fetchall():
        home_team, away_team, sport, source, home_odds, away_odds = row
        source_class = "source-blockchain" if "blockchain" in source else "source-api"
        
        odds_html = ""
        if home_odds and away_odds:
            odds_html = f"""
            <div class="odds-row">
                <div class="odd-box">
                    <div class="odd-label">HOME</div>
                    <div class="odd-value">{home_odds:.3f}</div>
                </div>
                <div class="odd-box">
                    <div class="odd-label">AWAY</div>
                    <div class="odd-value">{away_odds:.3f}</div>
                </div>
            </div>
            """
        
        markets_html += f"""
        <div class="market-card">
            <div class="market-header">
                <div class="market-teams">{home_team} vs {away_team}</div>
                <div>
                    <span class="market-sport">{sport}</span>
                    <span class="{source_class}" style="margin-left:10px">{source}</span>
                </div>
            </div>
            {odds_html}
        </div>
        """
    data['live_markets'] = markets_html or "<p style='color:#666'>No markets in next 24 hours</p>"
    
    # Blockchain markets
    cursor.execute("""
        SELECT m.home_team, m.away_team, m.sport, m.source
        FROM market m
        WHERE m.source LIKE '%blockchain%'
        ORDER BY m.updated_at DESC
        LIMIT 5
    """)
    
    blockchain_html = ""
    for home, away, sport, source in cursor.fetchall():
        blockchain_html += f"""
        <div class="market-card">
            <div class="market-teams">{home} vs {away}</div>
            <div style="color:#666; margin-top:5px">
                {sport} • {source}
            </div>
        </div>
        """
    data['blockchain_markets'] = blockchain_html
    
    # Best odds
    cursor.execute("""
        SELECT m.home_team, m.away_team, 
               MAX(o.decimal_odds) as best_odds, o.outcome
        FROM market m
        JOIN odd o ON m.source_id = o.source_id
        WHERE o.decimal_odds > 2.5
        GROUP BY m.source_id
        ORDER BY best_odds DESC
        LIMIT 5
    """)
    
    best_odds_html = ""
    for home, away, odds, outcome in cursor.fetchall():
        best_odds_html += f"""
        <div class="market-card">
            <div class="market-teams">{home} vs {away}</div>
            <div style="margin-top:5px">
                <span style="color:#888">{outcome.upper()}:</span>
                <span style="color:#00ff00; font-weight:bold; font-size:1.2em">{odds:.3f}</span>
            </div>
        </div>
        """
    data['best_odds'] = best_odds_html
    
    # Sport analysis
    cursor.execute("""
        SELECT 
            m.sport,
            COUNT(DISTINCT m.source_id) as total,
            COUNT(DISTINCT CASE WHEN m.source = 'api_live_real' THEN m.source_id END) as api,
            COUNT(DISTINCT CASE WHEN m.source LIKE '%blockchain%' THEN m.source_id END) as blockchain,
            COUNT(DISTINCT o.source_id) as with_odds,
            ROUND(AVG(o.decimal_odds), 3) as avg_odds
        FROM market m
        LEFT JOIN odd o ON m.source_id = o.source_id
        GROUP BY m.sport
        ORDER BY total DESC
        LIMIT 10
    """)
    
    analysis_html = ""
    for row in cursor.fetchall():
        analysis_html += f"""
        <tr>
            <td>{row[0]}</td>
            <td>{row[1]:,}</td>
            <td>{row[2]:,}</td>
            <td>{row[3]:,}</td>
            <td>{row[4]:,}</td>
            <td>{row[5] or 'N/A'}</td>
        </tr>
        """
    data['sport_analysis'] = analysis_html
    
    data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    conn.close()
    return data

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            data = get_dashboard_data()
            # Use safe string substitution to avoid issues with CSS braces
            html = HTML_TEMPLATE
            for key, value in data.items():
                html = html.replace('{' + key + '}', str(value))
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress request logs
        pass

def main():
    print("🎯 Starting Ominari Comprehensive Dashboard")
    print("📊 Dashboard available at: http://localhost:8889")
    print("🔄 Auto-refreshes every 30 seconds")
    print("Press Ctrl+C to stop")
    
    server = HTTPServer(('0.0.0.0', 8889), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n✋ Shutting down dashboard...")
        server.shutdown()

if __name__ == '__main__':
    main()