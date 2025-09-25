#!/usr/bin/env python3
"""
Update sport classifications using comprehensive pattern matching
"""

import sqlite3
import re

DB_PATH = "sport_odds.db"

def main():
    print("🔧 Updating sport classifications with comprehensive pattern matching...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all api_live_real markets
    cursor.execute("""
        SELECT source_id, home_team, away_team, league_name, sport 
        FROM market 
        WHERE source = 'api_live_real'
        ORDER BY home_team
    """)
    
    markets = cursor.fetchall()
    print(f"📊 Found {len(markets)} markets to process")
    
    sport_updates = {
        'Soccer': 0,
        'Baseball': 0,
        'Basketball': 0,
        'Hockey': 0,
        'Football': 0,
        'AFL': 0,
        'MMA': 0,
        'Tennis': 0,
        'Golf': 0,
        'Racing': 0,
        'Esports': 0,
        'Unknown': 0
    }
    
    # Sport detection patterns
    patterns = {
        'Soccer': [
            # Teams
            r'\b(FC|City|United|Real|Club|Atletico|Barcelona|Madrid|Chelsea|Arsenal|Liverpool)\b',
            r'\b(Milan|Inter|Juventus|Roma|Dortmund|Bayern|PSG|Lyon|Marseille|Monaco)\b',
            r'\b(Galaxy|LAFC|Sounders|Timbers|Whitecaps|Impact|Wanderers|Victory|Phoenix)\b',
            # Leagues
            r'(Premier League|La Liga|Serie A|Bundesliga|Ligue 1|MLS|UEFA|Champions League)',
            # Common patterns
            r'(FC\s+\w+|\w+\s+FC|\w+\s+United|\w+\s+City)',
        ],
        'Baseball': [
            r'\b(Yankees|Dodgers|Giants|Cubs|Red Sox|Astros|Rangers|Rays|Blue Jays)\b',
            r'\b(Orioles|Twins|Mariners|Royals|Athletics|Phillies|Braves|Marlins|Mets)\b',
            r'\b(Padres|Rockies|Diamondbacks|Brewers|Pirates|Reds|Nationals|Cardinals)\b',
            r'\b(White Sox|Angels|Tigers|Guardians)\b',
            r'(MLB|AL|NL|American League|National League)',
        ],
        'Basketball': [
            r'\b(Lakers|Celtics|Warriors|Bulls|Heat|Spurs|Nets|Knicks|Clippers|Suns)\b',
            r'\b(Mavericks|Rockets|Thunder|Jazz|Trail Blazers|Kings|Pelicans|Grizzlies)\b',
            r'\b(Hornets|Hawks|Magic|Wizards|Pacers|Cavaliers|Pistons|Raptors|Bucks)\b',
            r'\b(Timberwolves|Nuggets|76ers|Sixers)\b',
            r'\b(Wildcats|Tar Heels|Blue Devils|Cardinals|Bruins|Maccabi|CSKA|Olimpia)\b',
            r'(NBA|NCAA|Euroleague|Basketball)',
        ],
        'Hockey': [
            r'\b(Oilers|Panthers|Lightning|Avalanche|Bruins|Canadiens|Rangers|Islanders)\b',
            r'\b(Devils|Flyers|Penguins|Capitals|Hurricanes|Blue Jackets|Red Wings)\b',
            r'\b(Blackhawks|Blues|Predators|Jets|Wild|Stars|Coyotes|Golden Knights)\b',
            r'\b(Sharks|Kings|Ducks|Canucks|Flames|Senators|Sabres|Maple Leafs|Kraken)\b',
            r'(HC\s+\w+|\w+\s+HK|Dynamo|SKA|CSKA)',
            r'(NHL|KHL|SHL|Liiga|Hockey)',
        ],
        'Football': [
            r'\b(Raiders|Chargers|Chiefs|Broncos|Cowboys|Eagles|49ers|Seahawks)\b',
            r'\b(Cardinals|Rams|Packers|Bears|Lions|Vikings|Buccaneers|Saints)\b',
            r'\b(Falcons|Panthers|Patriots|Bills|Dolphins|Jets|Steelers|Ravens)\b',
            r'\b(Browns|Bengals|Titans|Jaguars|Colts|Texans|Commanders|Giants)\b',
            r'\b(Crimson Tide|Buckeyes|Wolverines|Fighting Irish|Trojans|Longhorns)\b',
            r'(NFL|NCAA Football|CFB|College Football)',
        ],
        'AFL': [
            r'\b(Bulldogs|SUNS|Swans|Lions|Cats|Eagles|Crows|Hawthorn|Melbourne)\b',
            r'\b(Richmond|Essendon|Collingwood|Geelong|Port Adelaide|Fremantle)\b',
            r'\b(Carlton|St Kilda|North Melbourne|West Coast|Brisbane|Adelaide|GWS)\b',
            r'(AFL|AFLW|Australian Football)',
        ],
        'MMA': [
            r'(UFC|PFL|Bellator|ONE Championship|Fight Night)',
            r'(Lightweight|Welterweight|Middleweight|Heavyweight|Bantamweight)',
            r'(\w+\s+vs\s+\w+.*Fight)',
        ],
        'Tennis': [
            r'(Open|Masters|Grand Slam|Wimbledon|US Open|French Open|Australian Open)',
            r'(ATP|WTA|Davis Cup)',
            r'\b(Djokovic|Nadal|Federer|Murray|Sinner|Alcaraz|Medvedev|Zverev)\b',
        ],
        'Golf': [
            r'(Open Championship|Masters|PGA|US Open|British Open|Ryder Cup)',
            r'(FedEx Cup|Round Leader|Tournament Winner)',
        ],
        'Racing': [
            r'(Grand Prix|NASCAR|Formula|F1|IndyCar|Speedway|Circuit|Rally|Le Mans)',
        ],
        'Esports': [
            r'\b(Gaming|Esports|G2|Fnatic|Secret|Virtus|Academy|Cloud9|TSM|FaZe)\b',
            r'\b(Liquid|NaVi|Astralis)\b',
            r'(League of Legends|CS:GO|CSGO|Dota|Valorant|LoL)',
        ],
    }
    
    # Process each market
    for source_id, home_team, away_team, league_name, current_sport in markets:
        # Combine text for searching
        search_text = f"{home_team} {away_team} {league_name or ''}".upper()
        
        # Score each sport
        sport_scores = {}
        
        for sport, sport_patterns in patterns.items():
            score = 0
            for pattern in sport_patterns:
                if re.search(pattern, search_text, re.IGNORECASE):
                    score += 1
                    # League names are worth more
                    if any(league in pattern for league in ['NFL', 'NBA', 'MLB', 'NHL', 'AFL', 'UFC', 'ATP', 'PGA']):
                        score += 2
            
            if score > 0:
                sport_scores[sport] = score
        
        # Determine sport
        if sport_scores:
            detected_sport = max(sport_scores, key=sport_scores.get)
        else:
            detected_sport = 'Unknown'
        
        # Update if different from current
        if detected_sport != current_sport:
            cursor.execute(
                "UPDATE market SET sport = ? WHERE source_id = ?",
                (detected_sport, source_id)
            )
            sport_updates[detected_sport] += 1
            
            if sport_updates[detected_sport] <= 3:  # Show first 3 examples
                print(f"  {detected_sport}: {home_team} vs {away_team}")
    
    conn.commit()
    
    # Show results
    print(f"\n✅ Updates by sport:")
    for sport, count in sport_updates.items():
        if count > 0:
            print(f"  {sport}: {count} markets updated")
    
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

if __name__ == "__main__":
    main()