#!/usr/bin/env python3
"""
Update sports in SQLite using Overtime API sport definitions
"""

import sqlite3
import requests
import json

DB_PATH = "sport_odds.db"

def get_sport_mappings():
    """Get sport definitions from Overtime API"""
    try:
        response = requests.get('https://api.overtime.io/overtime-v2/sports', timeout=10)
        return response.json()
    except:
        return {}

def main():
    print("🎯 Updating sports using Overtime API definitions...")
    
    # Get sport mappings
    sport_mappings = get_sport_mappings()
    print(f"📊 Found {len(sport_mappings)} sport definitions from API")
    
    # Build comprehensive sport detection patterns
    sport_patterns = {}
    
    # Extract patterns from API
    for sport_id, info in sport_mappings.items():
        sport_name = info.get('sport', 'Unknown')
        label = info.get('label', '')
        
        if sport_name not in sport_patterns:
            sport_patterns[sport_name] = []
        
        # Add league patterns
        if label:
            sport_patterns[sport_name].append(label)
            # Add variations
            if 'NCAA' in label:
                sport_patterns[sport_name].append('College')
            if sport_name == 'Fighting' and 'UFC' in label:
                sport_patterns['Fighting'].append('MMA')
    
    print("\n📋 Sport categories from API:")
    for sport, patterns in sorted(sport_patterns.items()):
        print(f"  {sport}: {', '.join(patterns[:5])}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Define update queries based on API patterns
    updates = [
        # Soccer
        ("UPDATE market SET sport = 'Soccer' WHERE source = 'api_live_real' AND sport != 'Soccer' AND (league_name LIKE '%MLS%' OR league_name LIKE '%Premier%' OR league_name LIKE '%Liga%' OR league_name LIKE '%Bundesliga%' OR league_name LIKE '%Serie A%' OR league_name LIKE '%Ligue%' OR league_name LIKE '%UEFA%' OR league_name LIKE '%Champions League%' OR league_name LIKE '%Europa%' OR league_name LIKE '%Copa%' OR league_name LIKE '%Eredivisie%' OR home_team LIKE '%FC%' OR away_team LIKE '%FC%' OR home_team LIKE '%United%' OR away_team LIKE '%United%' OR home_team LIKE '%City%' OR away_team LIKE '%City%')", "Soccer"),
        
        # Baseball
        ("UPDATE market SET sport = 'Baseball' WHERE source = 'api_live_real' AND sport != 'Baseball' AND (league_name LIKE '%MLB%' OR league_name LIKE '%NPB%' OR league_name LIKE '%KBO%' OR league_name LIKE '%Baseball%' OR home_team IN ('Yankees', 'Dodgers', 'Giants', 'Cubs', 'Red Sox', 'Astros', 'Rangers', 'Rays', 'Blue Jays', 'Orioles', 'Twins', 'Mariners', 'Royals', 'Athletics', 'Phillies', 'Braves', 'Marlins', 'Mets', 'Padres', 'Rockies', 'Diamondbacks', 'Brewers', 'Pirates', 'Reds', 'Nationals', 'Cardinals', 'White Sox', 'Angels', 'Tigers', 'Guardians'))", "Baseball"),
        
        # Basketball
        ("UPDATE market SET sport = 'Basketball' WHERE source = 'api_live_real' AND sport != 'Basketball' AND (league_name LIKE '%NBA%' OR league_name LIKE '%WNBA%' OR league_name LIKE '%NCAA%Basketball%' OR league_name LIKE '%Euroleague%' OR home_team IN ('Lakers', 'Celtics', 'Warriors', 'Bulls', 'Heat', 'Spurs', 'Nets', 'Knicks', 'Clippers', 'Suns', 'Mavericks', 'Rockets', 'Thunder', 'Jazz', 'Trail Blazers', 'Kings', 'Pelicans', 'Grizzlies', 'Hornets', 'Hawks', 'Magic', 'Wizards', 'Pacers', 'Cavaliers', 'Pistons', 'Raptors', 'Bucks', 'Timberwolves', 'Nuggets', '76ers'))", "Basketball"),
        
        # Hockey
        ("UPDATE market SET sport = 'Hockey' WHERE source = 'api_live_real' AND sport != 'Hockey' AND (league_name LIKE '%NHL%' OR league_name LIKE '%KHL%' OR league_name LIKE '%SHL%' OR league_name LIKE '%IIHF%' OR league_name LIKE '%Hockey%' OR league_name LIKE '%Ice Hockey%' OR home_team IN ('Oilers', 'Panthers', 'Lightning', 'Avalanche', 'Bruins', 'Canadiens', 'Rangers', 'Islanders', 'Devils', 'Flyers', 'Penguins', 'Capitals', 'Hurricanes', 'Blue Jackets', 'Red Wings', 'Blackhawks', 'Blues', 'Predators', 'Jets', 'Wild', 'Stars', 'Coyotes', 'Golden Knights', 'Sharks', 'Kings', 'Ducks', 'Canucks', 'Flames', 'Senators', 'Sabres', 'Maple Leafs', 'Kraken'))", "Hockey"),
        
        # Football
        ("UPDATE market SET sport = 'Football' WHERE source = 'api_live_real' AND sport != 'Football' AND (league_name LIKE '%NFL%' OR league_name LIKE '%NCAA%Football%' OR league_name LIKE '%CFB%' OR home_team IN ('Raiders', 'Chargers', 'Chiefs', 'Broncos', 'Cowboys', 'Eagles', '49ers', 'Seahawks', 'Cardinals', 'Rams', 'Packers', 'Bears', 'Lions', 'Vikings', 'Buccaneers', 'Saints', 'Falcons', 'Panthers', 'Patriots', 'Bills', 'Dolphins', 'Jets', 'Steelers', 'Ravens', 'Browns', 'Bengals', 'Titans', 'Jaguars', 'Colts', 'Texans', 'Commanders', 'Giants'))", "Football"),
        
        # Fighting (MMA/Boxing)
        ("UPDATE market SET sport = 'Fighting' WHERE source = 'api_live_real' AND sport != 'Fighting' AND (league_name LIKE '%UFC%' OR league_name LIKE '%PFL%' OR league_name LIKE '%Bellator%' OR league_name LIKE '%ONE%' OR league_name LIKE '%Fight%' OR league_name LIKE '%Boxing%' OR league_name LIKE '%MMA%')", "Fighting"),
        
        # Tennis
        ("UPDATE market SET sport = 'Tennis' WHERE source = 'api_live_real' AND sport != 'Tennis' AND (league_name LIKE '%ATP%' OR league_name LIKE '%WTA%' OR league_name LIKE '%Tennis%' OR league_name LIKE '%Open%' OR league_name LIKE '%Grand Slam%' OR league_name LIKE '%Wimbledon%')", "Tennis"),
        
        # Golf
        ("UPDATE market SET sport = 'Golf' WHERE source = 'api_live_real' AND sport != 'Golf' AND (league_name LIKE '%PGA%' OR league_name LIKE '%Golf%' OR league_name LIKE '%Masters%' OR home_team LIKE '%Round%Leader%' OR away_team LIKE '%Tournament%Winner%')", "Golf"),
        
        # eSports
        ("UPDATE market SET sport = 'eSports' WHERE source = 'api_live_real' AND sport != 'eSports' AND (home_team LIKE '%Esports%' OR away_team LIKE '%Esports%' OR home_team LIKE '%Gaming%' OR away_team LIKE '%Gaming%' OR league_name LIKE '%CS:%' OR league_name LIKE '%League of Legends%' OR league_name LIKE '%Dota%' OR league_name LIKE '%Valorant%')", "eSports"),
        
        # Cricket
        ("UPDATE market SET sport = 'Cricket' WHERE source = 'api_live_real' AND sport != 'Cricket' AND (league_name LIKE '%IPL%' OR league_name LIKE '%T20%' OR league_name LIKE '%Cricket%' OR league_name LIKE '%BBL%')", "Cricket"),
        
        # Handball
        ("UPDATE market SET sport = 'Handball' WHERE source = 'api_live_real' AND sport != 'Handball' AND (league_name LIKE '%Handball%' OR league_name LIKE '%HB%' OR league_name LIKE '%EHF%')", "Handball"),
    ]
    
    total_updated = 0
    
    print("\n🔄 Applying updates...")
    for query, sport_name in updates:
        cursor.execute(query)
        count = cursor.rowcount
        if count > 0:
            print(f"  ✅ {sport_name}: {count} markets updated")
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