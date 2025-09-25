#!/usr/bin/env python3
"""
Simple dashboard to show the real Overtime markets we just synced
"""

import sqlite3
from datetime import datetime, timezone
import json

DB_PATH = "sport_odds.db"

def get_real_markets():
    """Get the real markets we just synced."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get real API markets with odds
    cursor.execute('''
        SELECT m.home_team, m.away_team, m.sport, m.league_name, m.maturity_date,
               o.outcome, o.decimal_odds, o.american_odds
        FROM market m
        LEFT JOIN odd o ON m.source_id = o.source_id
        WHERE m.source = 'api_live_real'
        ORDER BY m.maturity_date, m.home_team, o.outcome
    ''')
    
    results = cursor.fetchall()
    conn.close()
    
    # Group by market
    markets = {}
    for row in results:
        home, away, sport, league, maturity, outcome, decimal, american = row
        
        market_key = f"{home} vs {away}"
        if market_key not in markets:
            markets[market_key] = {
                'home_team': home,
                'away_team': away,
                'sport': sport,
                'league': league,
                'maturity_date': maturity,
                'odds': {}
            }
        
        if outcome and decimal:
            markets[market_key]['odds'][outcome] = {
                'decimal': decimal,
                'american': american
            }
    
    return list(markets.values())

def generate_html_dashboard(markets):
    """Generate HTML dashboard."""
    css = """
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .stats { display: flex; gap: 20px; margin-bottom: 20px; }
        .stat-card { background: white; padding: 15px; border-radius: 8px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .market-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; }
        .market-card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .market-header { font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }
        .market-info { color: #7f8c8d; margin-bottom: 15px; }
        .odds-row { display: flex; gap: 10px; margin-bottom: 8px; }
        .odd-button { background: #3498db; color: white; padding: 8px 12px; border-radius: 5px; text-align: center; flex: 1; }
        .sport-soccer { border-left: 4px solid #27ae60; }
        .sport-hockey { border-left: 4px solid #e74c3c; }
        .sport-baseball { border-left: 4px solid #f39c12; }
        .sport-unknown { border-left: 4px solid #95a5a6; }
        .live-indicator { background: #e74c3c; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px; }
    """
    
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🎯 Real Overtime Markets Dashboard</title>
    <style>
        {css}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Real Overtime Markets Dashboard</h1>
            <p>Live markets from Overtime API with real team names and odds</p>
            <div class="live-indicator">● LIVE DATA FROM OVERTIME V2 API</div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Total Markets</h3>
                <h2>{{total_markets}}</h2>
            </div>
            <div class="stat-card">
                <h3>Sports</h3>
                <h2>{{unique_sports}}</h2>
            </div>
            <div class="stat-card">
                <h3>Data Source</h3>
                <h2>Real API</h2>
            </div>
        </div>
        
        <div class="market-grid">
            {{market_cards}}
        </div>
    </div>
</body>
</html>
    """
    
    # Generate market cards
    market_cards = ""
    sports_count = {}
    
    for market in markets:
        sport = market['sport']
        sports_count[sport] = sports_count.get(sport, 0) + 1
        
        sport_class = f"sport-{sport.lower()}" if sport != 'Unknown' else 'sport-unknown'
        
        # Format date
        try:
            date_obj = datetime.fromisoformat(market['maturity_date'].replace('Z', '+00:00'))
            formatted_date = date_obj.strftime('%Y-%m-%d %H:%M UTC')
        except:
            formatted_date = market['maturity_date']
        
        # Generate odds buttons
        odds_html = ""
        for outcome, odd_data in market['odds'].items():
            decimal = odd_data['decimal']
            american = odd_data['american']
            odds_html += f"""
                <div class="odd-button">
                    {outcome.title()}: {decimal} ({int(american):+d})
                </div>
            """
        
        if not odds_html:
            odds_html = '<div class="odd-button">Odds Loading...</div>'
            
        market_cards += f"""
            <div class="market-card {sport_class}">
                <div class="market-header">{market['home_team']} vs {market['away_team']}</div>
                <div class="market-info">
                    🏆 {market['league']} | 🎮 {market['sport']} | 📅 {formatted_date}
                </div>
                <div class="odds-row">
                    {odds_html}
                </div>
            </div>
        """
    
    return html_template.format(
        total_markets=len(markets),
        unique_sports=len(sports_count),
        market_cards=market_cards
    )

def main():
    print("🚀 Generating Real Overtime Markets Dashboard...")
    
    # Get real markets
    markets = get_real_markets()
    print(f"📊 Found {len(markets)} real markets")
    
    # Show sample data
    print("🎯 Sample markets:")
    for i, market in enumerate(markets[:5]):
        print(f"  {i+1}. {market['home_team']} vs {market['away_team']} ({market['sport']})")
    
    # Generate HTML
    html = generate_html_dashboard(markets)
    
    # Save to file
    with open('real_dashboard.html', 'w') as f:
        f.write(html)
    
    print("✅ Dashboard generated: real_dashboard.html")
    print("🌐 Open real_dashboard.html in your browser to view real markets!")
    print(f"📈 Showing {len(markets)} real games from Overtime API")
    
    # Show breakdown by sport
    sports = {}
    for market in markets:
        sport = market['sport']
        sports[sport] = sports.get(sport, 0) + 1
    
    print("🏆 Markets by sport:")
    for sport, count in sports.items():
        print(f"  {sport}: {count} markets")

if __name__ == "__main__":
    main()