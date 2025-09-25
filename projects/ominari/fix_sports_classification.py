#!/usr/bin/env python3
"""
Fix sport classification for the markets we synced
Since API doesn't provide sport field, we need better detection
"""

import sqlite3
import re

DB_PATH = "sport_odds.db"

def detect_sport(home_team, away_team, league_name):
    """Better sport detection based on team names and leagues."""
    combined = f"{home_team} {away_team} {league_name}".lower()
    
    # Soccer patterns
    if any(pattern in combined for pattern in [
        'fc', 'united', 'city', 'real', 'barcelona', 'juventus', 
        'arsenal', 'chelsea', 'liverpool', 'manchester', 'madrid',
        'milan', 'inter', 'roma', 'atletico', 'bayern', 'dortmund',
        'psg', 'marseille', 'ajax', 'benfica', 'porto', 'sporting',
        'club ', ' sc', 'afc ', ' cf ', 'football club', 'fútbol',
        'calcio', 'groupe', 'group stage', 'premier league', 'la liga',
        'serie a', 'bundesliga', 'ligue', 'eredivisie', 'primeira'
    ]):
        return 'Soccer'
    
    # Baseball patterns  
    if any(pattern in combined for pattern in [
        'yankees', 'red sox', 'dodgers', 'giants', 'cubs', 'cardinals',
        'astros', 'rangers', 'rays', 'blue jays', 'orioles', 'twins',
        'white sox', 'mariners', 'angels', 'athletics', 'phillies',
        'braves', 'marlins', 'mets', 'padres', 'rockies', 'diamondbacks',
        'brewers', 'pirates', 'reds', 'nationals', 'guardians', 'tigers',
        'royals', 'mlb', 'american league', 'national league', 'world series'
    ]):
        return 'Baseball'
    
    # Hockey patterns
    if any(pattern in combined for pattern in [
        'oilers', 'panthers', 'lightning', 'avalanche', 'maple leafs',
        'canadiens', 'bruins', 'rangers', 'penguins', 'capitals',
        'hurricanes', 'islanders', 'devils', 'flyers', 'senators',
        'sabres', 'red wings', 'blue jackets', 'wild', 'blues',
        'blackhawks', 'predators', 'stars', 'jets', 'flames',
        'canucks', 'sharks', 'ducks', 'kings', 'coyotes', 'kraken',
        'nhl', 'hockey', ' hc ', 'hk ', 'ice hockey'
    ]):
        return 'Hockey'
    
    # Basketball patterns
    if any(pattern in combined for pattern in [
        'lakers', 'celtics', 'warriors', 'bulls', 'heat', 'spurs',
        'mavericks', 'suns', 'nuggets', 'clippers', 'kings', 'blazers',
        'jazz', 'grizzlies', 'pelicans', 'timberwolves', 'thunder',
        'rockets', 'pistons', 'pacers', 'bucks', 'cavaliers', 'raptors',
        'knicks', 'nets', '76ers', 'wizards', 'hornets', 'hawks',
        'magic', 'nba', 'basketball'
    ]):
        return 'Basketball'
    
    # Football patterns
    if any(pattern in combined for pattern in [
        'raiders', 'chargers', 'chiefs', 'broncos', 'cowboys', 'eagles',
        'giants', 'washington', 'packers', 'bears', 'lions', 'vikings',
        'buccaneers', 'saints', 'falcons', 'panthers', '49ers', 'seahawks',
        'rams', 'cardinals', 'patriots', 'bills', 'dolphins', 'jets',
        'steelers', 'ravens', 'browns', 'bengals', 'titans', 'colts',
        'jaguars', 'texans', 'nfl', 'super bowl', 'playoff'
    ]):
        return 'Football'
    
    # MMA/Fighting patterns
    if any(pattern in combined for pattern in [
        'ufc', 'fight night', 'pfl', 'bellator', 'one championship',
        ' vs ', 'boxing', 'mma'
    ]) and len(home_team.split()) <= 3 and len(away_team.split()) <= 3:
        return 'MMA'
    
    # Golf patterns
    if any(pattern in combined for pattern in [
        'pga', 'open 2025', 'grand prix', 'masters', 'golf', 'round leader',
        'podium', 'belgian grand prix', 'dutch grand prix', 'italian grand prix'
    ]):
        return 'Golf'
    
    # Esports patterns
    if any(pattern in combined for pattern in [
        'gaming', 'esports', 'esport', 'g2 ', 'fnatic', 'vitality',
        'secret', 'virtus', 'pro', 'academy', 'lfo', 'dplus', 'scarz',
        'weibo', 'psg talon', 'mir gaming', 'wolves esports'
    ]):
        return 'Esports'
        
    # Australian Football
    if any(pattern in combined for pattern in [
        'bulldogs', 'gws', 'giants', 'crows', 'hawthorn', 'melbourne',
        'eagles', 'suns', 'richmond', 'swans', 'essendon', 'collingwood',
        'brisbane lions', 'geelong', 'port adelaide', 'fremantle', 'carlton',
        'st kilda', 'afl'
    ]):
        return 'AFL'
    
    return 'Other'

def main():
    print("🔧 Fixing sport classifications...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all markets that need sport updates
    cursor.execute('''
        SELECT source_id, home_team, away_team, league_name, sport
        FROM market
        WHERE source = 'api_live_real'
    ''')
    
    markets = cursor.fetchall()
    print(f"📊 Processing {len(markets)} markets...")
    
    # Count updates by sport
    sport_counts = {}
    updates = 0
    
    for source_id, home, away, league, current_sport in markets:
        detected_sport = detect_sport(home, away, league or '')
        
        if detected_sport != current_sport:
            cursor.execute('''
                UPDATE market 
                SET sport = ? 
                WHERE source_id = ?
            ''', (detected_sport, source_id))
            updates += 1
            
        sport_counts[detected_sport] = sport_counts.get(detected_sport, 0) + 1
    
    conn.commit()
    
    print(f"✅ Updated {updates} markets with correct sports")
    print("\n📊 Final sport distribution:")
    for sport, count in sorted(sport_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sport}: {count} markets")
    
    # Show some examples of each sport
    print("\n🎯 Sample markets by sport:")
    for sport in ['Soccer', 'Baseball', 'Hockey', 'Basketball', 'Football']:
        cursor.execute('''
            SELECT home_team, away_team 
            FROM market 
            WHERE source = 'api_live_real' AND sport = ?
            LIMIT 3
        ''', (sport,))
        
        examples = cursor.fetchall()
        if examples:
            print(f"\n{sport}:")
            for home, away in examples:
                print(f"  • {home} vs {away}")
    
    conn.close()
    print("\n✅ Sport classification fixed!")

if __name__ == "__main__":
    main()