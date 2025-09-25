#!/usr/bin/env python3
"""Final comprehensive fix for Unknown markets"""

from database_v2 import db_manager
from models import Market
from sqlalchemy import func
import re

def is_person_name(text):
    """Check if text looks like a person's name"""
    words = text.strip().split()
    if len(words) >= 2 and len(words) <= 4:
        # Check if words start with capital letters
        if all(word[0].isupper() for word in words if word):
            # Check if it doesn't contain common team indicators
            team_indicators = ['fc', 'sc', 'ac', 'united', 'city', 'real', 'atletico', 
                             'sporting', 'club', 'team', 'esport', 'gaming']
            text_lower = text.lower()
            if not any(indicator in text_lower for indicator in team_indicators):
                return True
    return False

def determine_sport_final(home_team, away_team):
    """Final comprehensive sport determination"""
    full_text = f"{home_team} vs {away_team}"
    full_lower = full_text.lower()
    home_lower = home_team.lower()
    away_lower = away_team.lower()
    
    # Prop bets and special markets - check first
    if any(word in away_lower for word in ['over', 'under', '+', '-', 'yes', 'no']):
        # It's a prop bet - try to determine the sport from context
        if 'tigers' in home_lower or 'guardians' in home_lower:
            return 'Baseball'
        elif any(word in home_lower for word in ['lakers', 'celtics', 'warriors']):
            return 'Basketball'
        else:
            # Default prop bets to the sport they're most likely from
            return 'Soccer'  # Most common
    
    # eSports - expanded patterns
    esports_indicators = [
        'dplus', 'fearx', 'elevate', 'daystar', 'soul', 'arrival', 'cag osaka',
        'gentle mates', 'passion ua', 'shinigami', 'peekaboo', 'legacy lotus',
        'black mold', 'overpeek', 'nemesis impact', 'ghost', 'shimmer',
        'little bocks', 'flyquest', 'thekillaz', 'capivaras', 'curralzinho',
        'dusty roses', 'mibr fe', 'level up', 'espoiled', 'gaming', 'esport',
        'legacy', 'impact', 'phantom', 'ghost'
    ]
    if any(indicator in full_lower for indicator in esports_indicators):
        return 'eSports'
    
    # Individual person names - likely combat sports or tennis
    if is_person_name(home_team) and is_person_name(away_team):
        # Check for tennis player patterns
        if any(word in full_lower for word in ['ace', 'set', 'game', 'serve']):
            return 'Tennis'
        # Default individual matchups to MMA/Fighting
        return 'Fighting'
    
    # Soccer/Football - European and Latin American clubs
    soccer_indicators = [
        # Club prefixes/suffixes
        ' fc', 'fc ', ' sc', 'sc ', ' ac', 'ac ', ' sv', 'sv ', ' vf',
        ' il', 'il ', ' if', 'if ', ' sk', 'sk ', ' jk', 'jk ', ' rk',
        ' ca', 'ca ', ' ad', 'ad ', ' ld', 'ld ', ' cd', 'cd ',
        # Common club names
        'united', 'city', 'real', 'atletico', 'sporting', 'deportivo',
        'asociación', 'club', 'sociedad', 'universidad', 'instituto',
        # Specific teams
        'forli', 'pesaro', 'rimini', 'feyenoord', 'rotterdam', 'fortuna',
        'fenerbahçe', 'alanyaspor', 'alajuelense', 'san carlos', 'herediano',
        'saprissa', 'ferencváros', 'vålerenga', 'oud-heverlee', 'leuven',
        'vorskla', 'poltava', 'kardzhali', 'cska', 'waldhof', 'mannheim',
        'stuttgart', 'osnabrück', 'essen', 'rot-weiss', 'wehen', 'wiesbaden',
        'regensburg', 'oss', 'venlo', 'waalwijk', 'vitesse', 'slavia',
        'praha', 'bodø', 'glimt', 'tammeka', 'tartu', 'parnu', 'vaprus',
        'narva', 'trans', 'nomme', 'kalju', 'tallinna', 'kalev', 'paide',
        # Norwegian teams
        'melhus', 'aasane', 'krageroe', 'bodo', 'nordstrand', 'bergsoey',
        'haukar', 'reykjavik', 'hafnarfjordur', 'ibv',
        # Egyptian teams
        'al-masry', 'ghazl', 'mahalla', 'wadi', 'degla', 'tala\'ea', 'gaish'
    ]
    if any(indicator in full_lower for indicator in soccer_indicators):
        return 'Soccer'
    
    # Baseball - MLB teams and KBO
    baseball_teams = [
        'tigers', 'guardians', 'rangers', 'pirates', 'twins', 'orioles',
        'yankees', 'red sox', 'white sox', 'blue jays', 'rays', 'marlins',
        'nationals', 'braves', 'phillies', 'mets', 'dodgers', 'padres',
        'giants', 'athletics', 'mariners', 'angels', 'astros', 'rangers',
        'diamondbacks', 'rockies', 'cubs', 'cardinals', 'brewers', 'reds',
        'pirates', 'royals', 'twins', 'kt wiz', 'lg twins', 'nc dinos',
        'doosan bears', 'samsung lions', 'ssg landers', 'kia tigers',
        'lotte giants', 'hanwha eagles', 'kiwoom heroes'
    ]
    if any(team in full_lower for team in baseball_teams):
        return 'Baseball'
    
    # Ice Hockey - Swedish and other leagues
    hockey_indicators = [
        'skellefteå', 'färjestad', 'malmö', 'linköping', 'luleå', 'växjö',
        'örebro', 'djurgården', 'frölunda', 'rögle', 'leksand', 'mora',
        'brynäs', 'timrå', 'oskarshamn', 'hv71', 'modo'
    ]
    if any(indicator in full_lower for indicator in hockey_indicators):
        return 'Hockey'
    
    # College teams - US colleges
    college_indicators = [
        'lafayette', 'columbia', 'monmouth', 'villanova', 'ohio', 'gardner-webb',
        'morgan state', 'central state', 'missouri southern', 'northwest missouri',
        'southeast missouri', 'southern illinois', 'mississippi state',
        'northern illinois', 'winona state', 'minot state', 'carson newman',
        'lenoir rhyne', 'state', 'university', 'college'
    ]
    if any(indicator in full_lower for indicator in college_indicators):
        return 'College Sports'
    
    # Women's teams
    if any(word in full_lower for word in [' fe', 'women', 'ladies', 'feminino']):
        # Try to determine the sport
        if any(word in full_lower for word in ['mibr', 'dusty', 'flyquest']):
            return 'eSports'
        else:
            return 'Soccer'  # Default women's teams to soccer
    
    # Default unknown patterns
    # If it has typical sports team structure but we can't identify
    if ' vs ' in full_text and not is_person_name(home_team):
        # Default to Soccer as it's the most common
        return 'Soccer'
    
    return None

def main():
    with db_manager.get_db_session() as db:
        print("Final fix for Unknown sport designations...")
        
        # Get all Unknown markets
        unknown_markets = db.query(Market).filter(
            Market.sport == 'Unknown'
        ).all()
        
        print(f"Found {len(unknown_markets)} Unknown markets")
        
        fixed_count = 0
        sport_fixes = {}
        still_unknown = []
        
        for market in unknown_markets:
            new_sport = determine_sport_final(market.home_team, market.away_team)
            
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
        
        print(f"\nStill Unknown: {len(still_unknown)} markets")
        if len(still_unknown) > 0 and len(still_unknown) < 50:
            print("\nRemaining unknown markets:")
            for market in still_unknown:
                print(f"  - {market}")
        
        # Show final counts
        print("\nFinal sport distribution:")
        sport_counts = db.query(
            Market.sport,
            func.count().label('count')
        ).group_by(Market.sport).order_by(func.count().desc()).all()
        
        total = 0
        for sport, count in sport_counts:
            print(f"  {sport:20} {count:6d}")
            total += count
        print(f"  {'TOTAL':20} {total:6d}")

if __name__ == "__main__":
    main()