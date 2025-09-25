#!/usr/bin/env python3
"""
Classify remaining unknown markets based on patterns
"""

import sqlite3

DB_PATH = "sport_odds.db"

def main():
    print("🎯 Classifying remaining Unknown markets...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Pattern-based updates for individual sports
    updates = [
        # Table Tennis - individual player names
        ("UPDATE market SET sport = 'Table Tennis' WHERE source = 'api_live_real' AND sport = 'Unknown' AND (home_team LIKE '% %' AND away_team LIKE '% %' AND LENGTH(home_team) < 30 AND LENGTH(away_team) < 30 AND home_team NOT LIKE '% vs %' AND away_team NOT LIKE '% vs %' AND home_team NOT LIKE '%College%' AND away_team NOT LIKE '%College%' AND home_team NOT LIKE '%University%' AND league_name IN ('N/A', ''))", "Table Tennis (individual matches)"),
        
        # Special/Future bets
        ("UPDATE market SET sport = 'Futures' WHERE source = 'api_live_real' AND sport = 'Unknown' AND (home_team LIKE '%Super Bowl%' OR home_team LIKE '%World Series%' OR home_team LIKE '%Winner%' OR away_team LIKE '%Winner%' OR home_team LIKE '%MVP%' OR away_team LIKE '%MVP%')", "Futures/Special bets"),
        
        # College sports with specific keywords
        ("UPDATE market SET sport = 'Football' WHERE source = 'api_live_real' AND sport = 'Unknown' AND (home_team LIKE '%College%' OR away_team LIKE '%College%' OR home_team LIKE '%University%' OR away_team LIKE '%University%')", "College Football"),
        
        # International soccer teams
        ("UPDATE market SET sport = 'Soccer' WHERE source = 'api_live_real' AND sport = 'Unknown' AND (home_team LIKE '%FK%' OR away_team LIKE '%FK%' OR home_team LIKE '%SC%' OR away_team LIKE '%SC%' OR home_team LIKE '%CF%' OR away_team LIKE '%CF%' OR home_team LIKE '%IL%' OR away_team LIKE '%IL%' OR home_team LIKE '%IF%' OR away_team LIKE '%IF%' OR home_team LIKE '%BV%' OR away_team LIKE '%BV%' OR home_team LIKE '% SA' OR away_team LIKE '% SA')", "International Soccer"),
        
        # More esports patterns
        ("UPDATE market SET sport = 'eSports' WHERE source = 'api_live_real' AND sport = 'Unknown' AND (league_name IN ('ESEA', 'Asia-Pacific League', 'Europe MENA League', 'North America League', 'ESL Impact League', 'ESL Challenger League') OR home_team LIKE '%XI %' OR away_team LIKE '%XI %' OR home_team LIKE '%LFO%' OR away_team LIKE '%LFO%')", "eSports leagues"),
        
        # International hockey
        ("UPDATE market SET sport = 'Hockey' WHERE source = 'api_live_real' AND sport = 'Unknown' AND (home_team LIKE '%Traktor%' OR home_team LIKE '%Avtomobilist%' OR home_team LIKE '%Ilves%' OR home_team LIKE '%Kiekko%' OR home_team LIKE '%Sport' OR home_team LIKE '%Medvedi%')", "International Hockey"),
        
        # Handball teams
        ("UPDATE market SET sport = 'Handball' WHERE source = 'api_live_real' AND sport = 'Unknown' AND (home_team LIKE '%Wisła%' OR home_team LIKE '%Płock%' OR league_name LIKE '%Handball%')", "Handball"),
    ]
    
    total_updated = 0
    
    for query, description in updates:
        cursor.execute(query)
        count = cursor.rowcount
        if count > 0:
            print(f"  ✅ {description}: {count} markets")
            total_updated += count
    
    conn.commit()
    
    # Show final distribution
    print(f"\n📊 Final sport distribution:")
    cursor.execute('''
        SELECT sport, COUNT(*) as count 
        FROM market 
        WHERE source = 'api_live_real' 
        GROUP BY sport 
        ORDER BY count DESC
    ''')
    
    for sport, count in cursor.fetchall():
        print(f"  {sport}: {count} markets")
    
    conn.close()
    print(f"\n✅ Total markets updated: {total_updated}")

if __name__ == "__main__":
    main()