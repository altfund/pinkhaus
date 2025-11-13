#!/usr/bin/env python3
"""
Populate nation and governing body data based on leagues and teams
"""
import os

# Set environment for PostgreSQL on port 5999
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

import psycopg2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# League to Nation/Governing Body mappings
LEAGUE_MAPPINGS = {
    # England
    'Premier League': {'nation': 'England', 'governing_body': 'FA (The Football Association)'},
    'Championship': {'nation': 'England', 'governing_body': 'FA (The Football Association)'},
    'League One': {'nation': 'England', 'governing_body': 'FA (The Football Association)'},
    'League Two': {'nation': 'England', 'governing_body': 'FA (The Football Association)'},
    'FA Cup': {'nation': 'England', 'governing_body': 'FA (The Football Association)'},
    'Carabao Cup': {'nation': 'England', 'governing_body': 'EFL'},
    
    # Spain  
    'La Liga': {'nation': 'Spain', 'governing_body': 'RFEF (Royal Spanish Football Federation)'},
    'La Liga 2': {'nation': 'Spain', 'governing_body': 'RFEF (Royal Spanish Football Federation)'},
    'Copa del Rey': {'nation': 'Spain', 'governing_body': 'RFEF (Royal Spanish Football Federation)'},
    
    # Italy
    'Serie A': {'nation': 'Italy', 'governing_body': 'FIGC (Italian Football Federation)'},
    'Serie B': {'nation': 'Italy', 'governing_body': 'FIGC (Italian Football Federation)'},
    'Coppa Italia': {'nation': 'Italy', 'governing_body': 'FIGC (Italian Football Federation)'},
    
    # Germany
    'Bundesliga': {'nation': 'Germany', 'governing_body': 'DFB (German Football Association)'},
    '2. Bundesliga': {'nation': 'Germany', 'governing_body': 'DFB (German Football Association)'},
    'DFB-Pokal': {'nation': 'Germany', 'governing_body': 'DFB (German Football Association)'},
    
    # France
    'Ligue 1': {'nation': 'France', 'governing_body': 'FFF (French Football Federation)'},
    'Ligue 2': {'nation': 'France', 'governing_body': 'FFF (French Football Federation)'},
    'Coupe de France': {'nation': 'France', 'governing_body': 'FFF (French Football Federation)'},
    
    # Netherlands
    'Eredivisie': {'nation': 'Netherlands', 'governing_body': 'KNVB (Royal Dutch Football Association)'},
    'Eerste Divisie': {'nation': 'Netherlands', 'governing_body': 'KNVB (Royal Dutch Football Association)'},
    
    # Portugal
    'Primeira Liga': {'nation': 'Portugal', 'governing_body': 'FPF (Portuguese Football Federation)'},
    'Liga Portugal 2': {'nation': 'Portugal', 'governing_body': 'FPF (Portuguese Football Federation)'},
    
    # International/Continental
    'UEFA Champions League': {'nation': 'Europe', 'governing_body': 'UEFA'},
    'Europa League': {'nation': 'Europe', 'governing_body': 'UEFA'},
    'UEFA Conference League': {'nation': 'Europe', 'governing_body': 'UEFA'},
    'European Championship': {'nation': 'Europe', 'governing_body': 'UEFA'},
    'European Football': {'nation': 'Europe', 'governing_body': 'UEFA'},
    
    'World Cup': {'nation': 'International', 'governing_body': 'FIFA'},
    'International Football': {'nation': 'International', 'governing_body': 'FIFA'},
    'FIFA Club World Cup': {'nation': 'International', 'governing_body': 'FIFA'},
    
    # South America
    'Copa America': {'nation': 'South America', 'governing_body': 'CONMEBOL'},
    'Copa Libertadores': {'nation': 'South America', 'governing_body': 'CONMEBOL'},
    'Copa Sudamericana': {'nation': 'South America', 'governing_body': 'CONMEBOL'},
    
    # USA
    'MLS': {'nation': 'USA', 'governing_body': 'US Soccer Federation'},
    'NBA': {'nation': 'USA', 'governing_body': 'NBA'},
    'WNBA': {'nation': 'USA', 'governing_body': 'WNBA'},
    'NFL': {'nation': 'USA', 'governing_body': 'NFL'},
    'MLB': {'nation': 'USA', 'governing_body': 'MLB'},
    'NHL': {'nation': 'USA/Canada', 'governing_body': 'NHL'},
    
    # Other sports
    'Handball Bundesliga': {'nation': 'Germany', 'governing_body': 'DHB (German Handball Federation)'},
    'KBO': {'nation': 'South Korea', 'governing_body': 'KBO (Korea Baseball Organization)'},
    
    # Esports (global)
    'CCT Europe': {'nation': 'Europe', 'governing_body': 'CCT'},
    'ESEA League': {'nation': 'Global', 'governing_body': 'ESEA'},
    'ESL Pro League': {'nation': 'Global', 'governing_body': 'ESL'},
    'BLAST Premier': {'nation': 'Global', 'governing_body': 'BLAST'},
}

# Team-based nation detection patterns
TEAM_NATION_PATTERNS = {
    # England
    r'Manchester|Liverpool|Chelsea|Arsenal|Tottenham|Leicester|Everton|Leeds|Newcastle': 'England',
    # Spain
    r'Real Madrid|Barcelona|Atletico|Sevilla|Valencia|Villarreal|Real Sociedad|Betis': 'Spain',
    # Italy
    r'Juventus|Milan|Inter|Roma|Napoli|Lazio|Fiorentina|Atalanta': 'Italy',
    # Germany
    r'Bayern|Dortmund|Leipzig|Leverkusen|Frankfurt|Wolfsburg|Gladbach|Schalke': 'Germany',
    # France
    r'PSG|Paris Saint|Lyon|Marseille|Monaco|Lille|Nice|Rennes': 'France',
    # Netherlands
    r'Ajax|PSV|Feyenoord|AZ Alkmaar|Utrecht|Vitesse': 'Netherlands',
    # Portugal
    r'Benfica|Porto|Sporting|Braga': 'Portugal',
}

def populate_nation_data():
    conn = psycopg2.connect(
        host=os.environ['PG_HOST'],
        port=os.environ['PG_PORT'],
        user=os.environ['PG_USER'],
        password=os.environ['PG_PASSWORD'],
        database=os.environ['PG_DB']
    )
    cur = conn.cursor()
    
    total_updated = 0
    
    # Update based on league mappings
    logger.info("=== Updating nation/governing body from league mappings ===")
    
    for league, data in LEAGUE_MAPPINGS.items():
        cur.execute("""
            UPDATE market 
            SET nation = %s, governing_body = %s
            WHERE league_name = %s 
            AND (nation IS NULL OR nation = '')
        """, (data['nation'], data['governing_body'], league))
        
        updated = cur.rowcount
        if updated > 0:
            logger.info(f"  {league}: {updated} markets → {data['nation']} / {data['governing_body']}")
            total_updated += updated
    
    # Update based on team patterns for remaining markets
    logger.info("\n=== Updating nation from team patterns ===")
    
    for pattern, nation in TEAM_NATION_PATTERNS.items():
        cur.execute("""
            UPDATE market 
            SET nation = %s
            WHERE (nation IS NULL OR nation = '')
            AND sport = 'Soccer'
            AND (home_team ~* %s OR away_team ~* %s)
        """, (nation, pattern, pattern))
        
        updated = cur.rowcount
        if updated > 0:
            logger.info(f"  {nation}: {updated} markets updated")
            total_updated += updated
    
    # Set default values for remaining markets
    logger.info("\n=== Setting defaults for remaining markets ===")
    
    # Soccer defaults
    cur.execute("""
        UPDATE market 
        SET nation = 'International',
            governing_body = 'FIFA'
        WHERE sport = 'Soccer' 
        AND (nation IS NULL OR nation = '')
    """)
    logger.info(f"  Soccer defaults: {cur.rowcount} markets")
    total_updated += cur.rowcount
    
    # Other sports defaults
    sport_defaults = {
        'Basketball': {'nation': 'USA', 'governing_body': 'FIBA'},
        'Baseball': {'nation': 'USA', 'governing_body': 'MLB'},
        'American Football': {'nation': 'USA', 'governing_body': 'NFL'},
        'Hockey': {'nation': 'USA/Canada', 'governing_body': 'NHL/IIHF'},
        'Cricket': {'nation': 'International', 'governing_body': 'ICC'},
        'Tennis': {'nation': 'International', 'governing_body': 'ITF/ATP/WTA'},
        'Golf': {'nation': 'International', 'governing_body': 'PGA/European Tour'},
        'MMA': {'nation': 'International', 'governing_body': 'UFC'},
        'Handball': {'nation': 'Europe', 'governing_body': 'IHF'},
        'Esports': {'nation': 'Global', 'governing_body': 'Various'},
    }
    
    for sport, data in sport_defaults.items():
        cur.execute("""
            UPDATE market 
            SET nation = %s, governing_body = %s
            WHERE sport = %s 
            AND (nation IS NULL OR nation = '')
        """, (data['nation'], data['governing_body'], sport))
        
        updated = cur.rowcount
        if updated > 0:
            logger.info(f"  {sport}: {updated} markets → {data['nation']} / {data['governing_body']}")
            total_updated += updated
    
    conn.commit()
    
    logger.info(f"\n✅ Total markets updated: {total_updated}")
    
    # Show distribution
    logger.info("\n=== Nation Distribution ===")
    cur.execute("""
        SELECT nation, COUNT(*) as count 
        FROM market 
        WHERE nation IS NOT NULL 
        GROUP BY nation 
        ORDER BY count DESC
        LIMIT 20
    """)
    
    for nation, count in cur.fetchall():
        logger.info(f"  {nation}: {count} markets")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    populate_nation_data()