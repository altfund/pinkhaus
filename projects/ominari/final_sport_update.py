#!/usr/bin/env python3
"""
Final comprehensive sport update using all available data
"""

import sqlite3
import requests

DB_PATH = "sport_odds.db"

def main():
    print("🎯 Final sport classification update...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all Unknown markets to analyze
    cursor.execute("""
        SELECT source_id, home_team, away_team, league_name 
        FROM market 
        WHERE source = 'api_live_real' AND sport = 'Unknown'
    """)
    
    unknown_markets = cursor.fetchall()
    print(f"📊 Found {len(unknown_markets)} Unknown markets to classify")
    
    # Comprehensive team-to-sport mapping
    team_sport_map = {
        # Baseball teams
        'Yankees': 'Baseball', 'Dodgers': 'Baseball', 'Giants': 'Baseball', 'Cubs': 'Baseball',
        'Red Sox': 'Baseball', 'Astros': 'Baseball', 'Rangers': 'Baseball', 'Rays': 'Baseball',
        'Blue Jays': 'Baseball', 'Orioles': 'Baseball', 'Twins': 'Baseball', 'Mariners': 'Baseball',
        'Royals': 'Baseball', 'Athletics': 'Baseball', 'Phillies': 'Baseball', 'Braves': 'Baseball',
        'Marlins': 'Baseball', 'Mets': 'Baseball', 'Padres': 'Baseball', 'Rockies': 'Baseball',
        'Diamondbacks': 'Baseball', 'Brewers': 'Baseball', 'Pirates': 'Baseball', 'Reds': 'Baseball',
        'Nationals': 'Baseball', 'Cardinals': 'Baseball', 'White Sox': 'Baseball', 'Angels': 'Baseball',
        'Tigers': 'Baseball', 'Guardians': 'Baseball',
        # Japanese baseball
        'Fighters': 'Baseball', 'Marines': 'Baseball', 'Lions': 'Baseball', 'Hawks': 'Baseball',
        'Swallows': 'Baseball', 'Carp': 'Baseball', 'Dragons': 'Baseball', 'BayStars': 'Baseball',
        'Buffaloes': 'Baseball', 'Eagles': 'Baseball',
        # Korean baseball
        'Bears': 'Baseball', 'Wyverns': 'Baseball', 'Dinos': 'Baseball', 'Landers': 'Baseball',
        # Hockey teams
        'Oilers': 'Hockey', 'Panthers': 'Hockey', 'Lightning': 'Hockey', 'Avalanche': 'Hockey',
        'Bruins': 'Hockey', 'Canadiens': 'Hockey', 'Islanders': 'Hockey', 'Devils': 'Hockey',
        'Flyers': 'Hockey', 'Penguins': 'Hockey', 'Capitals': 'Hockey', 'Hurricanes': 'Hockey',
        'Blue Jackets': 'Hockey', 'Red Wings': 'Hockey', 'Blackhawks': 'Hockey', 'Blues': 'Hockey',
        'Predators': 'Hockey', 'Wild': 'Hockey', 'Stars': 'Hockey', 'Coyotes': 'Hockey',
        'Golden Knights': 'Hockey', 'Sharks': 'Hockey', 'Ducks': 'Hockey', 'Canucks': 'Hockey',
        'Flames': 'Hockey', 'Senators': 'Hockey', 'Sabres': 'Hockey', 'Maple Leafs': 'Hockey',
        'Kraken': 'Hockey',
        # Basketball teams
        'Lakers': 'Basketball', 'Celtics': 'Basketball', 'Warriors': 'Basketball', 'Bulls': 'Basketball',
        'Heat': 'Basketball', 'Spurs': 'Basketball', 'Nets': 'Basketball', 'Knicks': 'Basketball',
        'Clippers': 'Basketball', 'Suns': 'Basketball', 'Mavericks': 'Basketball', 'Rockets': 'Basketball',
        'Thunder': 'Basketball', 'Trail Blazers': 'Basketball', 'Pelicans': 'Basketball',
        'Grizzlies': 'Basketball', 'Hornets': 'Basketball', 'Magic': 'Basketball', 'Wizards': 'Basketball',
        'Pacers': 'Basketball', 'Cavaliers': 'Basketball', 'Pistons': 'Basketball', 'Raptors': 'Basketball',
        'Bucks': 'Basketball', 'Timberwolves': 'Basketball', 'Nuggets': 'Basketball', '76ers': 'Basketball',
        # Football teams
        'Raiders': 'Football', 'Chargers': 'Football', 'Chiefs': 'Football', 'Broncos': 'Football',
        'Cowboys': 'Football', '49ers': 'Football', 'Seahawks': 'Football', 'Rams': 'Football',
        'Packers': 'Football', 'Vikings': 'Football', 'Buccaneers': 'Football', 'Saints': 'Football',
        'Falcons': 'Football', 'Patriots': 'Football', 'Bills': 'Football', 'Dolphins': 'Football',
        'Steelers': 'Football', 'Ravens': 'Football', 'Browns': 'Football', 'Bengals': 'Football',
        'Titans': 'Football', 'Jaguars': 'Football', 'Colts': 'Football', 'Texans': 'Football',
        'Commanders': 'Football',
        # AFL teams
        'Bulldogs': 'AFL', 'Swans': 'AFL', 'Lions': 'AFL', 'Cats': 'AFL', 'Crows': 'AFL',
        'Hawthorn': 'AFL', 'Richmond': 'AFL', 'Essendon': 'AFL', 'Collingwood': 'AFL',
        'Geelong': 'AFL', 'Port Adelaide': 'AFL', 'Fremantle': 'AFL', 'Carlton': 'AFL',
        'St Kilda': 'AFL', 'North Melbourne': 'AFL', 'West Coast': 'AFL', 'Adelaide': 'AFL'
    }
    
    # Pattern-based classification for remaining unknowns
    updates_by_sport = {}
    
    for source_id, home_team, away_team, league_name in unknown_markets:
        sport = None
        
        # First try exact team matching
        for team_name, team_sport in team_sport_map.items():
            if team_name in home_team or team_name in away_team:
                sport = team_sport
                break
        
        # If no match, try patterns
        if not sport:
            teams_text = f"{home_team} {away_team} {league_name or ''}"
            
            # Soccer patterns
            if any(pattern in teams_text for pattern in ['FC ', ' FC', 'United', 'City', 'Real ', 'Club ', 'Athletic', 'Sporting']):
                sport = 'Soccer'
            # Hockey patterns (HC = Hockey Club)
            elif any(pattern in teams_text for pattern in ['HC ', ' HC', ' HK', 'Dynamo', 'SKA', 'CSKA']):
                sport = 'Hockey'
            # Esports patterns
            elif any(pattern in teams_text.lower() for pattern in ['esports', 'gaming', 'academy', 'g2 ', 'fnatic', 'cloud9', 'tsm', 'liquid']):
                sport = 'eSports'
            # College patterns
            elif any(pattern in teams_text for pattern in ['University', 'College', 'State']):
                if 'Baseball' in teams_text:
                    sport = 'Baseball'
                elif 'Basketball' in teams_text:
                    sport = 'Basketball'
                elif 'Football' in teams_text:
                    sport = 'Football'
            # Japanese teams
            elif any(pattern in teams_text for pattern in ['Nippon', 'Tohoku', 'Chiba', 'Saitama', 'Hiroshima', 'Hanshin', 'Yokohama']):
                sport = 'Baseball'
        
        if sport:
            cursor.execute(
                "UPDATE market SET sport = ? WHERE source_id = ?",
                (sport, source_id)
            )
            updates_by_sport[sport] = updates_by_sport.get(sport, 0) + 1
    
    conn.commit()
    
    # Show updates
    print("\n✅ Updates by sport:")
    total_updates = 0
    for sport, count in sorted(updates_by_sport.items()):
        print(f"  {sport}: {count} markets")
        total_updates += count
    
    # Final cleanup - merge duplicate sport names
    cursor.execute("UPDATE market SET sport = 'eSports' WHERE sport = 'Esports'")
    cursor.execute("UPDATE market SET sport = 'Fighting' WHERE sport = 'MMA'")
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
    
    unknown_count = 0
    for sport, count in cursor.fetchall():
        print(f"  {sport}: {count} markets")
        if sport == 'Unknown':
            unknown_count = count
    
    conn.close()
    
    print(f"\n✅ Total markets updated: {total_updates}")
    print(f"📊 Remaining Unknown markets: {unknown_count}")
    
    if unknown_count > 0:
        print("\n💡 Most Unknown markets are likely:")
        print("  - Future/special bets (e.g., 'World Series Winner')")
        print("  - New teams not in our mapping")
        print("  - International leagues")

if __name__ == "__main__":
    main()