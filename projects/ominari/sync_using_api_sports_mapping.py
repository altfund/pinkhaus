#!/usr/bin/env python3
"""
Sync markets using the actual sport definitions from Overtime API
Maps tournament names and team patterns to the official sport categories
"""

import sqlite3
import requests
import json
from datetime import datetime

DB_PATH = "sport_odds.db"

def get_api_sport_mappings():
    """Get official sport definitions from Overtime API"""
    try:
        response = requests.get('https://api.overtime.io/overtime-v2/sports', timeout=10)
        return response.json()
    except:
        return {}

def build_sport_detector_from_api(sport_mappings):
    """Build sport detection using actual API sport definitions"""
    
    # Create mappings from API data
    sport_detector = {
        'by_league': {},      # Tournament/league name -> sport
        'by_keyword': {},     # Keywords -> sport
        'by_id': {}          # Sport ID -> sport info
    }
    
    for sport_id, info in sport_mappings.items():
        sport_name = info.get('sport', 'Unknown')
        label = info.get('label', '')
        optic_name = info.get('opticOddsName', '')
        
        # Store by ID
        sport_detector['by_id'][sport_id] = {
            'sport': sport_name,
            'label': label
        }
        
        # Map league names (case insensitive)
        if label:
            sport_detector['by_league'][label.upper()] = sport_name
            # Add variations
            if 'NCAA' in label:
                sport_detector['by_league']['NCAA'] = sport_name
                sport_detector['by_league']['COLLEGE'] = sport_name
            if 'NFL' in label:
                sport_detector['by_league']['NFL'] = sport_name
            if 'NBA' in label:
                sport_detector['by_league']['NBA'] = sport_name
            if 'MLB' in label:
                sport_detector['by_league']['MLB'] = sport_name
            if 'NHL' in label:
                sport_detector['by_league']['NHL'] = sport_name
            if 'UFC' in label or 'PFL' in label:
                sport_detector['by_league']['MMA'] = sport_name
                sport_detector['by_league']['FIGHTING'] = sport_name
        
        if optic_name:
            sport_detector['by_league'][optic_name.upper()] = sport_name
            # Extract country/league from format "Country - League"
            if ' - ' in optic_name:
                parts = optic_name.split(' - ')
                if len(parts) == 2:
                    sport_detector['by_league'][parts[1].upper()] = sport_name
    
    # Add keywords based on sport types from API
    sport_keywords = {
        'Soccer': ['FC', 'CITY', 'UNITED', 'REAL', 'CLUB', 'CF', 'SC', 'FOOTBALL CLUB'],
        'Baseball': ['YANKEES', 'DODGERS', 'GIANTS', 'CUBS', 'SOX', 'MARINERS', 'RANGERS'],
        'Basketball': ['LAKERS', 'CELTICS', 'WARRIORS', 'HEAT', 'SPURS', 'NETS', 'KNICKS'],
        'Hockey': ['OILERS', 'PANTHERS', 'LIGHTNING', 'BRUINS', 'RANGERS', 'PENGUINS'],
        'Football': ['RAIDERS', 'CHIEFS', 'COWBOYS', 'EAGLES', '49ERS', 'PACKERS'],
        'Fighting': ['VS', 'FIGHT', 'BOUT'],
        'Tennis': ['OPEN', 'MASTERS', 'WIMBLEDON', 'GRAND SLAM'],
        'Golf': ['ROUND LEADER', 'TOURNAMENT WINNER', 'MASTERS'],
        'eSports': ['ESPORTS', 'GAMING', 'ACADEMY', 'TEAM']
    }
    
    for sport, keywords in sport_keywords.items():
        for keyword in keywords:
            sport_detector['by_keyword'][keyword] = sport
    
    return sport_detector

def detect_sport_from_api_data(home_team, away_team, tournament, sport_detector):
    """Detect sport using official API mappings"""
    
    # Combine text for searching
    home_upper = home_team.upper()
    away_upper = away_team.upper()
    tournament_upper = tournament.upper() if tournament else ''
    full_text = f"{home_upper} {away_upper} {tournament_upper}"
    
    # 1. Try exact tournament match
    if tournament_upper in sport_detector['by_league']:
        return sport_detector['by_league'][tournament_upper]
    
    # 2. Try partial tournament match
    if tournament:
        for league_key, sport in sport_detector['by_league'].items():
            if league_key in tournament_upper or tournament_upper in league_key:
                return sport
    
    # 3. Check for sport keywords in tournament
    tournament_words = tournament_upper.split()
    for word in tournament_words:
        if word in sport_detector['by_league']:
            return sport_detector['by_league'][word]
    
    # 4. Special cases based on API sport categories
    if 'WORLD SERIES' in full_text or 'SUPER BOWL' in full_text:
        return 'Futures'
    
    # 5. Individual player matches (Table Tennis pattern)
    if not tournament or tournament in ['N/A', '']:
        # Check if looks like individual names
        if ' ' in home_team and ' ' in away_team:
            if len(home_team) < 30 and len(away_team) < 30:
                if not any(word in full_text for word in ['COLLEGE', 'UNIVERSITY', 'FC', 'UNITED']):
                    if home_team.count(' ') <= 2 and away_team.count(' ') <= 2:
                        return 'TableTennis'
    
    # 6. Check team keywords
    for keyword, sport in sport_detector['by_keyword'].items():
        if keyword in home_upper or keyword in away_upper:
            return sport
    
    # 7. Pattern-based detection for remaining
    if any(pattern in full_text for pattern in ['FC ', ' FC', 'UNITED', 'CITY', 'REAL ']):
        return 'Soccer'
    elif any(pattern in full_text for pattern in ['HC ', ' HC', ' HK', 'DYNAMO']):
        return 'Hockey'
    elif 'COLLEGE' in full_text or 'UNIVERSITY' in full_text:
        return 'Football'  # Default college sport
    elif any(pattern in full_text for pattern in ['ESPORTS', 'GAMING', 'ACADEMY']):
        return 'eSports'
    
    return 'Unknown'

def sync_with_api_sports():
    """Sync markets using official API sport definitions"""
    print("🎯 Syncing markets using official Overtime API sport definitions...")
    
    try:
        # Get official sport mappings
        sport_mappings = get_api_sport_mappings()
        if not sport_mappings:
            print("❌ Failed to get sport mappings from API")
            return False
        
        print(f"✅ Loaded {len(sport_mappings)} official sport definitions from API")
        
        # Build detector
        sport_detector = build_sport_detector_from_api(sport_mappings)
        print(f"📊 Created {len(sport_detector['by_league'])} league mappings")
        
        # Get games
        response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=10)
        games_data = response.json()
        
        print(f"📊 Found {len(games_data)} total games from API")
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Update all markets with better sport classification
        cursor.execute("""
            SELECT source_id, home_team, away_team, league_name 
            FROM market 
            WHERE source = 'api_live_real'
        """)
        
        all_markets = cursor.fetchall()
        print(f"\n🔄 Updating {len(all_markets)} markets with API sport definitions...")
        
        sport_distribution = {}
        updates_made = 0
        
        for source_id, home_team, away_team, league_name in all_markets:
            sport = detect_sport_from_api_data(home_team, away_team, league_name or '', sport_detector)
            
            # Update market
            cursor.execute(
                "UPDATE market SET sport = ?, updated_at = ? WHERE source_id = ?",
                (sport, datetime.now().isoformat(), source_id)
            )
            
            sport_distribution[sport] = sport_distribution.get(sport, 0) + 1
            updates_made += 1
            
            if updates_made % 500 == 0:
                print(f"  Progress: {updates_made}/{len(all_markets)} markets updated...")
        
        conn.commit()
        
        print(f"\n✅ Update complete! {updates_made} markets processed")
        
        print(f"\n📊 Sport distribution (based on API definitions):")
        for sport, count in sorted(sport_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sport}: {count} markets")
        
        # Show final database state
        cursor.execute("""
            SELECT sport, COUNT(*) 
            FROM market 
            WHERE source = 'api_live_real' 
            GROUP BY sport 
            ORDER BY COUNT(*) DESC
        """)
        
        print(f"\n📊 Final database totals:")
        unknown_count = 0
        for sport, count in cursor.fetchall():
            print(f"  {sport}: {count} markets")
            if sport == 'Unknown':
                unknown_count = count
        
        conn.close()
        
        print(f"\n✨ Success! Using official Overtime API sport definitions")
        print(f"📊 {unknown_count} markets remain Unknown (likely futures, props, or new leagues)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    sync_with_api_sports()