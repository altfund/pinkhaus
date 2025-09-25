#!/usr/bin/env python3
"""Fix markets with 'Unknown' sport designation"""

from database_v2 import db_manager
from models import Market
from sqlalchemy import func
import re

def determine_sport(home_team, away_team):
    """Determine sport based on team names"""
    teams_text = f"{home_team} {away_team}".lower()
    
    # Soccer patterns
    if any(pattern in teams_text for pattern in [
        'fc', 'united', 'city', 'real ', 'atletico', 'juventus', 'chelsea', 'arsenal',
        'liverpool', 'barcelona', 'madrid', 'munich', 'milan', 'inter ', 'tottenham'
    ]):
        # But not if it's a championship/cup winner market
        if 'championship winner' not in teams_text and 'cup winner' not in teams_text:
            return 'Soccer'
    
    # Baseball
    if any(pattern in teams_text for pattern in [
        'yankees', 'dodgers', 'astros', 'braves', 'mets', 'cubs', 'sox', 'nationals',
        'orioles', 'blue jays', 'rays', 'marlins', 'phillies', 'reds', 'brewers'
    ]):
        return 'Baseball'
    
    # Basketball (NBA/WNBA)
    if any(pattern in teams_text for pattern in [
        'lakers', 'warriors', 'celtics', 'heat', 'bulls', 'spurs', 'rockets',
        'clippers', 'nets', 'knicks', 'sixers', 'bucks', 'suns', 'kings',
        'liberty', 'lynx', 'mercury', 'storm', 'aces', 'sparks', 'fever'
    ]):
        return 'Basketball'
    
    # Hockey
    if any(pattern in teams_text for pattern in [
        'avalanche', 'bruins', 'sabres', 'flames', 'hurricanes', 'blackhawks',
        'blue jackets', 'stars', 'red wings', 'oilers', 'panthers', 'kings',
        'wild', 'canadiens', 'predators', 'devils', 'islanders', 'rangers',
        'senators', 'flyers', 'penguins', 'sharks', 'kraken', 'blues',
        'lightning', 'maple leafs', 'utah hc', 'canucks', 'golden knights',
        'capitals', 'jets'
    ]):
        return 'Hockey'
    
    # eSports patterns
    if any(pattern in teams_text for pattern in [
        'fnatic', 'g2', 'team liquid', 'navi', 'vitality', 'faze', 'astralis',
        'og ', 'eg ', 'tsm', 'cloud9', 'sentinels', 'dignitas', 't1 ', 'dwg',
        'mad lions', '9ine', 'parivision', 'eyeballers', 'minlate', 'chaos',
        'gaming', 'esports', ' ec ', 'kru ', 'xset', 'optic'
    ]):
        return 'eSports'
    
    # American Football
    if any(pattern in teams_text for pattern in [
        'packers', 'bears', 'vikings', 'lions', 'eagles', 'cowboys', 'giants',
        'commanders', 'buccaneers', 'saints', 'falcons', 'panthers', '49ers',
        'seahawks', 'rams', 'cardinals', 'patriots', 'bills', 'dolphins', 'jets',
        'steelers', 'browns', 'bengals', 'ravens', 'texans', 'colts', 'jaguars',
        'titans', 'broncos', 'chiefs', 'raiders', 'chargers'
    ]):
        return 'American Football'
    
    # Australian Football
    if any(pattern in teams_text for pattern in [
        'adelaide', 'brisbane lions', 'carlton', 'collingwood', 'essendon',
        'fremantle', 'geelong', 'gold coast', 'gws giants', 'hawthorn',
        'melbourne fc', 'north melbourne', 'port adelaide', 'richmond',
        'st kilda', 'sydney swans', 'west coast', 'western bulldogs',
        'cairns taipans', 'phoenix', 'sydney kings'
    ]):
        return 'Australian Football'
    
    # Table Tennis
    if any(pattern in teams_text for pattern in [
        'table tennis', 'tt ', 'ping pong'
    ]) or re.match(r'^[a-z]+ [a-z]+ vs [a-z]+ [a-z]+$', teams_text):
        # Simple name pattern often indicates table tennis
        if all(len(word) < 10 for word in teams_text.split()):
            return 'TableTennis'
    
    # Racing/F1
    if any(pattern in teams_text for pattern in [
        'grand prix', 'gp ', ' f1', 'formula', 'racing', 'nascar', 'indycar',
        'motogp', 'rally', 'le mans'
    ]):
        return 'Racing'
    
    # Tennis
    if 'vs' in teams_text and any(pattern in teams_text for pattern in [
        'atp', 'wta', 'grand slam', 'wimbledon', 'roland garros'
    ]):
        return 'Tennis'
    
    # Fighting/MMA
    if any(pattern in teams_text for pattern in [
        'ufc', 'bellator', 'one championship', 'pfl', 'boxing', 'mma'
    ]):
        return 'Fighting'
    
    # College Sports
    if any(pattern in teams_text for pattern in [
        'alabama', 'georgia', 'ohio state', 'michigan', 'clemson', 'lsu',
        'notre dame', 'texas', 'oklahoma', 'florida', 'auburn', 'penn state',
        'oregon', 'washington', 'usc', 'ucla', 'stanford', 'cal ', 'duke',
        'north carolina', 'virginia', 'miami', 'florida state', 'nc state'
    ]):
        return 'College Sports'
    
    # If still unknown, check for patterns
    if 'championship winner' in teams_text or 'cup winner' in teams_text:
        if 'nba' in teams_text:
            return 'Basketball'
        elif 'nhl' in teams_text:
            return 'Hockey'
        elif 'uefa' in teams_text or 'premier league' in teams_text:
            return 'Soccer'
    
    return None  # Keep as Unknown if can't determine

def main():
    with db_manager.get_db_session() as db:
        print("Fixing Unknown sport designations...")
        
        # Get all Unknown markets
        unknown_markets = db.query(Market).filter(
            Market.sport == 'Unknown'
        ).all()
        
        print(f"Found {len(unknown_markets)} Unknown markets")
        
        fixed_count = 0
        sport_fixes = {}
        
        for market in unknown_markets:
            new_sport = determine_sport(market.home_team, market.away_team)
            
            if new_sport:
                print(f"Fixing: {market.home_team} vs {market.away_team} -> {new_sport}")
                market.sport = new_sport
                fixed_count += 1
                sport_fixes[new_sport] = sport_fixes.get(new_sport, 0) + 1
        
        db.commit()
        
        print(f"\n✅ Fixed {fixed_count} Unknown markets")
        print("\nBreakdown of fixes:")
        for sport, count in sorted(sport_fixes.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sport:20} {count:6d}")
        
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