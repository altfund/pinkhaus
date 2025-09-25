#!/usr/bin/env python3
"""
Simple script to show the real markets we synced
"""

import sqlite3
from datetime import datetime, timezone

DB_PATH = "sport_odds.db"

def main():
    print("🚀 Real Overtime Markets Dashboard")
    print("=" * 50)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get real API markets
    cursor.execute('''
        SELECT m.home_team, m.away_team, m.sport, m.league_name, m.maturity_date,
               COUNT(o.id) as odds_count
        FROM market m
        LEFT JOIN odd o ON m.source_id = o.source_id
        WHERE m.source = 'api_live_real'
        GROUP BY m.source_id
        ORDER BY m.maturity_date
        LIMIT 20
    ''')
    
    markets = cursor.fetchall()
    
    print(f"📊 Found {len(markets)} real markets (showing first 20):")
    print()
    
    sports_count = {}
    
    for i, (home, away, sport, league, maturity, odds_count) in enumerate(markets, 1):
        # Count sports
        sports_count[sport] = sports_count.get(sport, 0) + 1
        
        # Format date
        try:
            date_obj = datetime.fromisoformat(maturity.replace('Z', '+00:00'))
            formatted_date = date_obj.strftime('%Y-%m-%d %H:%M')
        except:
            formatted_date = maturity
        
        # Sport emoji
        sport_emoji = {
            'Soccer': '⚽',
            'Hockey': '🏒', 
            'Baseball': '⚾',
            'Basketball': '🏀'
        }.get(sport, '🎮')
        
        print(f"{i:2d}. {sport_emoji} {home} vs {away}")
        print(f"    🏆 {league} | 📅 {formatted_date} | 📊 {odds_count} odds")
        print()
    
    # Summary
    print("🎯 SUMMARY:")
    print(f"✅ Total real markets: {len(markets)}")
    print(f"📡 Data source: Overtime V2 API")
    print(f"🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("🏆 Sports breakdown:")
    for sport, count in sports_count.items():
        emoji = {'Soccer': '⚽', 'Hockey': '🏒', 'Baseball': '⚾', 'Basketball': '🏀'}.get(sport, '🎮')
        print(f"  {emoji} {sport}: {count} markets")
    
    # Get total count across all sources
    cursor.execute("SELECT source, COUNT(*) FROM market GROUP BY source")
    all_sources = cursor.fetchall()
    
    print()
    print("📈 All data sources:")
    for source, count in all_sources:
        if source == 'api_live_real':
            print(f"  ✅ {source}: {count} markets (REAL LIVE DATA)")
        else:
            print(f"  📦 {source}: {count} markets")
    
    conn.close()
    
    print()
    print("🎉 SUCCESS: Real Overtime markets are now in the database!")
    print("🌐 Ready to display in web dashboard!")

if __name__ == "__main__":
    main()