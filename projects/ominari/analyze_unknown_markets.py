#!/usr/bin/env python3
"""Analyze all Unknown markets to find patterns"""

from database_v2 import db_manager
from models import Market
from collections import defaultdict
import re

def main():
    with db_manager.get_db_session() as db:
        # Get all Unknown markets
        unknown_markets = db.query(Market).filter(
            Market.sport == 'Unknown'
        ).all()
        
        print(f"Analyzing {len(unknown_markets)} Unknown markets...")
        print("=" * 80)
        
        # Categorize by patterns
        patterns = defaultdict(list)
        
        # Pattern analysis
        for market in unknown_markets:
            home = market.home_team
            away = market.away_team
            full_text = f"{home} vs {away}"
            
            # Check for specific patterns
            if ' vs ' not in full_text:
                patterns['no_vs_separator'].append(full_text)
            elif re.match(r'^[A-Za-z\s]+ vs (Winner|Championship|Cup|Leader)', full_text):
                patterns['tournament_winner'].append(full_text)
            elif 'over' in away.lower() or 'under' in away.lower():
                patterns['over_under'].append(full_text)
            elif 'yes' in away.lower() or 'no' in away.lower():
                patterns['yes_no_prop'].append(full_text)
            elif 'draw' in away.lower():
                patterns['draw_markets'].append(full_text)
            elif re.match(r'^[A-Z]{2,4} ', home):
                patterns['abbreviations'].append(full_text)
            elif any(char.isdigit() for char in full_text):
                patterns['contains_numbers'].append(full_text)
            elif len(home.split()) == 1 and len(away.split()) == 1:
                patterns['single_word_teams'].append(full_text)
            elif '/' in full_text:
                patterns['contains_slash'].append(full_text)
            elif '-' in full_text and 'vs' in full_text:
                patterns['contains_dash'].append(full_text)
            else:
                patterns['other'].append(full_text)
        
        # Show pattern summary
        print("Pattern Summary:")
        for pattern, markets in sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\n{pattern}: {len(markets)} markets")
            # Show first 10 examples
            for i, market in enumerate(markets[:10]):
                print(f"  - {market}")
            if len(markets) > 10:
                print(f"  ... and {len(markets) - 10} more")
        
        # Analyze team name frequencies
        print("\n" + "=" * 80)
        print("Most common team names in Unknown markets:")
        team_counts = defaultdict(int)
        
        for market in unknown_markets:
            team_counts[market.home_team] += 1
            team_counts[market.away_team] += 1
        
        # Show top 30 most common teams
        for team, count in sorted(team_counts.items(), key=lambda x: x[1], reverse=True)[:30]:
            print(f"  {team:40} appears {count:4} times")
        
        # Check for specific sport indicators we might have missed
        print("\n" + "=" * 80)
        print("Checking for missed sport indicators...")
        
        sport_indicators = {
            'Tennis': ['ace', 'set', 'game', 'deuce', 'serve', 'tennis'],
            'Cricket': ['cricket', 'innings', 'wicket', 'boundary', 'sixes', 'runs'],
            'Rugby': ['rugby', 'try', 'scrum', 'lineout'],
            'Volleyball': ['volleyball', 'spike', 'serve', 'block'],
            'Handball': ['handball'],
            'Darts': ['darts', '180s', 'checkout'],
            'Snooker': ['snooker', 'frame', 'century'],
            'Boxing': ['boxing', 'round', 'knockout', 'ko'],
            'MMA': ['mma', 'ufc', 'submission', 'knockout'],
            'Cycling': ['cycling', 'tour de', 'giro', 'vuelta'],
            'Athletics': ['athletics', 'meter', 'marathon', 'sprint'],
            'Swimming': ['swimming', 'freestyle', 'butterfly', 'backstroke'],
            'Winter Sports': ['ski', 'snowboard', 'slalom', 'biathlon', 'luge'],
            'Badminton': ['badminton'],
            'Water Polo': ['water polo']
        }
        
        sport_matches = defaultdict(list)
        
        for market in unknown_markets:
            text = f"{market.home_team} {market.away_team}".lower()
            for sport, keywords in sport_indicators.items():
                if any(keyword in text for keyword in keywords):
                    sport_matches[sport].append(f"{market.home_team} vs {market.away_team}")
                    break
        
        print("\nPotential sport matches found:")
        for sport, matches in sorted(sport_matches.items()):
            print(f"\n{sport}: {len(matches)} potential matches")
            for match in matches[:5]:
                print(f"  - {match}")
            if len(matches) > 5:
                print(f"  ... and {len(matches) - 5} more")

if __name__ == "__main__":
    main()