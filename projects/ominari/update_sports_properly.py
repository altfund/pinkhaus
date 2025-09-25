#!/usr/bin/env python3
"""
Update sports classification properly for all real markets
"""

import sqlite3

DB_PATH = "sport_odds.db"

def main():
    print("🔧 Updating sport classifications with proper logic...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Define the sport mapping based on the patterns from old scripts
    sport_updates = [
        # Soccer teams with FC, City, United, etc
        ("UPDATE market SET sport = 'Soccer' WHERE source = 'api_live_real' AND (home_team LIKE '%FC%' OR away_team LIKE '%FC%' OR home_team LIKE '%City%' OR away_team LIKE '%City%' OR home_team LIKE '%United%' OR away_team LIKE '%United%' OR home_team LIKE '%Real %' OR away_team LIKE '%Real %' OR home_team LIKE '%Club %' OR away_team LIKE '%Club %')", "Soccer by team name patterns"),
        
        # Baseball teams
        ("UPDATE market SET sport = 'Baseball' WHERE source = 'api_live_real' AND (home_team LIKE '%Yankees%' OR home_team LIKE '%Dodgers%' OR home_team LIKE '%Giants%' OR home_team LIKE '%Cubs%' OR home_team LIKE '%Red Sox%' OR home_team LIKE '%Astros%' OR home_team LIKE '%Rangers%' OR home_team LIKE '%Rays%' OR home_team LIKE '%Blue Jays%' OR home_team LIKE '%Orioles%' OR home_team LIKE '%Twins%' OR home_team LIKE '%Mariners%' OR home_team LIKE '%Royals%' OR home_team LIKE '%Athletics%' OR home_team LIKE '%Phillies%' OR home_team LIKE '%Braves%' OR home_team LIKE '%Marlins%' OR home_team LIKE '%Mets%' OR home_team LIKE '%Padres%' OR home_team LIKE '%Rockies%' OR home_team LIKE '%Diamondbacks%' OR home_team LIKE '%Brewers%' OR home_team LIKE '%Pirates%' OR home_team LIKE '%Reds%' OR home_team LIKE '%Nationals%' OR home_team LIKE '%Cardinals%')", "Baseball teams"),
        
        # Fix Kansas City Royals specifically
        ("UPDATE market SET sport = 'Baseball' WHERE source = 'api_live_real' AND home_team = 'Kansas City Royals'", "Kansas City Royals fix"),
        
        # Hockey teams
        ("UPDATE market SET sport = 'Hockey' WHERE source = 'api_live_real' AND (home_team LIKE '%Oilers%' OR home_team LIKE '%Panthers%' OR home_team LIKE '%Lightning%' OR home_team LIKE '%Avalanche%' OR home_team LIKE '%Bruins%' OR home_team LIKE '%Canadiens%' OR home_team LIKE '%HC %' OR home_team LIKE '% HK%' OR league_name LIKE '%Hockey%' OR league_name LIKE '%NHL%')", "Hockey teams"),
        
        # Football teams  
        ("UPDATE market SET sport = 'Football' WHERE source = 'api_live_real' AND (home_team LIKE '%Raiders%' OR home_team LIKE '%Chargers%' OR home_team LIKE '%Chiefs%' OR home_team LIKE '%Broncos%' OR home_team LIKE '%Cowboys%' OR home_team LIKE '%Eagles%' OR home_team LIKE '%49ers%' OR home_team LIKE '%Seahawks%' OR home_team LIKE '%Cardinals%' AND home_team NOT LIKE '%St. Louis Cardinals%' AND home_team NOT LIKE '%Arizona Cardinals%')", "NFL teams"),
        
        # Australian Football (AFL)
        ("UPDATE market SET sport = 'AFL' WHERE source = 'api_live_real' AND (home_team LIKE '%Bulldogs%' OR home_team LIKE '%SUNS%' OR home_team LIKE '%Swans%' OR home_team LIKE '%Lions%' OR home_team LIKE '%Cats%' OR home_team LIKE '%Eagles%' OR home_team LIKE '%Crows%' OR home_team LIKE '%Hawthorn%' OR home_team LIKE '%Melbourne%' OR home_team LIKE '%Richmond%' OR home_team LIKE '%Essendon%' OR home_team LIKE '%Collingwood%' OR home_team LIKE '%Geelong%' OR home_team LIKE '%Port Adelaide%' OR home_team LIKE '%Fremantle%' OR home_team LIKE '%Carlton%' OR home_team LIKE '%St Kilda%')", "AFL teams"),
        
        # Basketball
        ("UPDATE market SET sport = 'Basketball' WHERE source = 'api_live_real' AND (home_team LIKE '%Lakers%' OR home_team LIKE '%Celtics%' OR home_team LIKE '%Warriors%' OR home_team LIKE '%Bulls%' OR home_team LIKE '%Heat%' OR home_team LIKE '%Spurs%' OR league_name LIKE '%NBA%' OR league_name LIKE '%Basketball%')", "Basketball teams"),
        
        # Esports
        ("UPDATE market SET sport = 'Esports' WHERE source = 'api_live_real' AND (home_team LIKE '%Gaming%' OR home_team LIKE '%Esports%' OR home_team LIKE '%G2 %' OR home_team LIKE '%Fnatic%' OR home_team LIKE '%Secret%' OR home_team LIKE '%Virtus%' OR home_team LIKE '%Academy%' OR league_name LIKE '%League%' AND sport = 'Unknown')", "Esports teams"),
        
        # MMA/Fighting
        ("UPDATE market SET sport = 'MMA' WHERE source = 'api_live_real' AND league_name LIKE '%UFC%' OR league_name LIKE '%PFL%' OR league_name LIKE '%Fight Night%'", "MMA/UFC events"),
        
        # Racing
        ("UPDATE market SET sport = 'Racing' WHERE source = 'api_live_real' AND (home_team LIKE '%Grand Prix%' OR home_team LIKE '%NASCAR%' OR league_name LIKE '%Formula%')", "Racing events"),
        
        # Golf
        ("UPDATE market SET sport = 'Golf' WHERE source = 'api_live_real' AND (home_team LIKE '%Open 2025%' OR home_team LIKE '%Masters%' OR home_team LIKE '%PGA%' OR home_team LIKE '%Round%Leader%')", "Golf tournaments"),
    ]
    
    total_updated = 0
    
    for query, description in sport_updates:
        cursor.execute(query)
        updated = cursor.rowcount
        if updated > 0:
            print(f"✅ {description}: {updated} markets updated")
            total_updated += updated
    
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
    
    # Show some soccer examples to verify
    print(f"\n⚽ Sample soccer markets:")
    cursor.execute('''
        SELECT home_team, away_team 
        FROM market 
        WHERE source = 'api_live_real' AND sport = 'Soccer' 
        LIMIT 10
    ''')
    
    for home, away in cursor.fetchall():
        print(f"  • {home} vs {away}")
    
    conn.close()
    print(f"\n✅ Updated {total_updated} markets total!")

if __name__ == "__main__":
    main()