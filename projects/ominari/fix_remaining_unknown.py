#!/usr/bin/env python3
"""Fix remaining Unknown markets based on deeper analysis"""

from database_v2 import db_manager
from models import Market
from sqlalchemy import func
import re

def determine_sport_comprehensive(home_team, away_team):
    """More comprehensive sport determination"""
    full_text = f"{home_team} vs {away_team}".lower()
    home_lower = home_team.lower()
    away_lower = away_team.lower()
    
    # eSports patterns - expanded
    esports_patterns = [
        # Team names
        'fnatic', 'g2', 'team liquid', 'navi', 'vitality', 'faze', 'astralis',
        'og ', 'eg ', 'tsm', 'cloud9', 'sentinels', 'dignitas', 't1 ', 'dwg',
        'mad lions', '9ine', 'parivision', 'eyeballers', 'minlate', 'chaos',
        'gaming', 'esports', ' ec ', 'kru ', 'xset', 'optic', 'nemiga',
        'betboom', 'geek fam', 'onic', 'xperion', 'procamp', 'enter force',
        'back2thegame', 'xi esport', 'reveal', 'regnum4games', 'hindsight',
        '888aura', 'vp.prodigy', 'pipsqueak', 'ukrainian boys', 'imperial',
        'b8', '6gpa', 'fury', 'invaders', 'i love you', 'stronghold',
        'cold metal', 'the gatos guapos', 'newgens', 'kalmychata', 'red canids',
        'bestia', 'preasy mix', 'hyperion', 'wopa esport', 'qmistry', 'reason',
        'skinrave', 'marsborne', 'phantom', 'flame sharks', 'khan', 'mouz nxt',
        'wylde', 'alliance', 'falcons esport'
    ]
    if any(pattern in full_text for pattern in esports_patterns):
        return 'eSports'
    
    # Soccer/Football - international teams and clubs
    if any(pattern in full_text for pattern in [
        'fc ', ' fc', 'united', 'city', 'real ', 'atletico', 'juventus', 'chelsea',
        'arsenal', 'liverpool', 'barcelona', 'madrid', 'munich', 'milan', 'inter ',
        'tottenham', 'everton', 'leicester', 'west ham', 'crystal palace', 'fulham',
        'bournemouth', 'brighton', 'burnley', 'newcastle', 'southampton', 'watford',
        'norwich', 'aston villa', 'leeds', 'wolverhampton', 'sheffield',
        # International teams
        'spain', 'england', 'france', 'germany', 'italy', 'portugal', 'belgium',
        'netherlands', 'croatia', 'argentina', 'brazil', 'uruguay', 'colombia',
        'mexico', 'poland', 'switzerland', 'denmark', 'sweden', 'austria',
        'czech', 'romania', 'hungary', 'serbia', 'greece', 'turkey', 'turkiye',
        'bulgaria', 'slovakia', 'norway', 'finland', 'iceland', 'ireland',
        'scotland', 'wales', 'northern ireland', 'albania', 'montenegro',
        'macedonia', 'kosovo', 'lithuania', 'latvia', 'estonia', 'belarus',
        'ukraine', 'russia', 'georgia', 'armenia', 'azerbaijan', 'kazakhstan',
        'israel', 'cyprus', 'malta', 'luxembourg', 'liechtenstein', 'andorra',
        'gibraltar', 'moldova', 'bosnia', 'herzegovina', 'slovenia', 'croatia',
        # Brazilian teams
        'palmeiras', 'flamengo', 'corinthians', 'santos', 'sao paulo', 'gremio',
        'internacional', 'fluminense', 'vasco', 'botafogo', 'cruzeiro', 'atletico mg',
        'river plate', 'boca juniors'
    ]):
        # Avoid false positives
        if not any(word in full_text for word in ['athletics', 'united states', 'championship winner']):
            return 'Soccer'
    
    # Handball patterns
    if any(pattern in full_text for pattern in [
        'handball', ' hb', 'handewitt', 'kiel', 'magdeburg', 'flensburg',
        'burgdorf', 'hannover', 'montpellier handball', 'chambery', 'nantes hb',
        'fenix toulouse', 'dunkerque hb', 'dinamo bucuresti', 'kolstad',
        'telekom veszprem', 'pler-budapest', 'paris saint germain hb',
        'barcelone hb', 'hbc nantes'
    ]):
        return 'Handball'
    
    # Cricket patterns
    if any(pattern in full_text for pattern in [
        'glamorgan', 'surrey', 'yorkshire', 'lancashire', 'warwickshire',
        'nottinghamshire', 'essex', 'kent', 'sussex', 'middlesex', 'gloucestershire',
        'worcestershire', 'northamptonshire', 'derbyshire', 'leicestershire',
        'durham', 'hampshire', 'somerset'
    ]):
        return 'Cricket'
    
    # Ice Hockey - Finnish and Swedish leagues
    if any(pattern in full_text for pattern in [
        'kalpa', 'kiekko-espoo', 'hv71', 'brynas', 'jyp jyvaskyla', 'hifk',
        'ilves', 'tps turku', 'lukko', 'saipa', 'jukurit', 'sport vaasa',
        'karpat', 'assat', 'kookoo', 'ftc-telekom', 'kac klagenfurt'
    ]):
        return 'Hockey'
    
    # Volleyball patterns
    if any(pattern in full_text for pattern in [
        'volley', 'volleyball', 'kyky-betset', 'savo volley'
    ]):
        return 'Volleyball'
    
    # Water Polo patterns
    if 'water polo' in full_text:
        return 'Water Polo'
    
    # Boxing/Combat sports
    if any(pattern in full_text for pattern in ['boxing', 'bout', 'vs round']):
        return 'Boxing'
    
    # Athletics
    if 'athletics' in full_text and 'oakland' not in full_text:
        return 'Athletics'
    
    # Cycling
    if any(pattern in full_text for pattern in ['cycling', 'tour de', 'giro', 'vuelta']):
        return 'Cycling'
    
    # Winter Sports
    if any(pattern in full_text for pattern in ['ski', 'snowboard', 'slalom', 'biathlon']):
        return 'Winter Sports'
    
    # Rugby
    if any(pattern in full_text for pattern in ['rugby', 'crusaders', 'hurricanes', 'chiefs']):
        return 'Rugby'
    
    # Tennis players - check for common tennis player name patterns
    if ' vs ' in full_text:
        # Single name vs single name often indicates individual sports
        home_words = home_team.split()
        away_words = away_team.split()
        if len(home_words) <= 2 and len(away_words) <= 2:
            # Check if it looks like person names (First Last format)
            if all(word[0].isupper() for word in home_words + away_words if word):
                # Could be tennis, boxing, or other individual sport
                if 'round' in full_text or 'bout' in full_text:
                    return 'Boxing'
                elif any(pattern in full_text for pattern in ['ace', 'set', 'game', 'deuce']):
                    return 'Tennis'
    
    # MVP/Award markets
    if any(pattern in full_text for pattern in ['mvp', 'heisman trophy', 'award winner']):
        if 'mlb' in full_text:
            return 'Baseball'
        elif 'nba' in full_text:
            return 'Basketball'
        elif 'nfl' in full_text:
            return 'American Football'
        elif 'nhl' in full_text:
            return 'Hockey'
    
    # Country vs Country (international matches)
    countries = [
        'spain', 'england', 'france', 'germany', 'italy', 'portugal', 'belgium',
        'netherlands', 'croatia', 'argentina', 'brazil', 'uruguay', 'colombia',
        'poland', 'switzerland', 'denmark', 'sweden', 'austria', 'norway',
        'finland', 'iceland', 'ireland', 'scotland', 'wales', 'albania',
        'montenegro', 'macedonia', 'lithuania', 'latvia', 'estonia', 'belarus',
        'ukraine', 'russia', 'georgia', 'armenia', 'azerbaijan', 'israel',
        'cyprus', 'malta', 'luxembourg', 'liechtenstein', 'gibraltar', 'moldova',
        'bosnia', 'herzegovina', 'slovenia', 'croatia', 'serbia', 'greece',
        'turkey', 'turkiye', 'bulgaria', 'slovakia', 'romania', 'hungary'
    ]
    
    home_is_country = any(country in home_lower for country in countries)
    away_is_country = any(country in away_lower for country in countries)
    
    if home_is_country and away_is_country:
        return 'Soccer'  # Most international matches are soccer
    
    return None

def main():
    with db_manager.get_db_session() as db:
        print("Fixing remaining Unknown sport designations...")
        
        # Get all Unknown markets
        unknown_markets = db.query(Market).filter(
            Market.sport == 'Unknown'
        ).all()
        
        print(f"Found {len(unknown_markets)} Unknown markets")
        
        fixed_count = 0
        sport_fixes = {}
        still_unknown = []
        
        for market in unknown_markets:
            new_sport = determine_sport_comprehensive(market.home_team, market.away_team)
            
            if new_sport:
                print(f"Fixing: {market.home_team} vs {market.away_team} -> {new_sport}")
                market.sport = new_sport
                fixed_count += 1
                sport_fixes[new_sport] = sport_fixes.get(new_sport, 0) + 1
            else:
                still_unknown.append(f"{market.home_team} vs {market.away_team}")
        
        db.commit()
        
        print(f"\n✅ Fixed {fixed_count} Unknown markets")
        print("\nBreakdown of fixes:")
        for sport, count in sorted(sport_fixes.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sport:20} {count:6d}")
        
        # Show what's still unknown
        print(f"\nStill Unknown: {len(still_unknown)} markets")
        if len(still_unknown) > 0:
            print("\nFirst 20 still unknown markets:")
            for market in still_unknown[:20]:
                print(f"  - {market}")
        
        # Show final counts
        print("\nFinal sport distribution:")
        sport_counts = db.query(
            Market.sport,
            func.count().label('count')
        ).group_by(Market.sport).order_by(func.count().desc()).all()
        
        for sport, count in sport_counts:
            print(f"  {sport:20} {count:6d}")

if __name__ == "__main__":
    main()