#!/usr/bin/env python3
"""
Sync nation data from API using intelligent inference from opticOddsName
This replaces manual nation mappings with API-derived data
"""
import os

# Set environment for PostgreSQL on port 5999
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

import requests
import logging
import re
from database_v2 import db_manager
from models import Market
from sqlalchemy import func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def infer_nation_from_optic_name(optic_name, label):
    """
    Intelligently infer nation from opticOddsName
    """
    if not optic_name:
        return None
    
    # Method 1: Split on " - " for "Country - League" format
    if ' - ' in optic_name:
        parts = optic_name.split(' - ', 1)
        nation = parts[0].strip()
        # Clean up common prefixes
        if nation.startswith('USA '):
            nation = 'USA'
        return nation
    
    # Method 2: Remove label from optic_name
    if label and label in optic_name:
        # Remove label and clean up
        nation = optic_name.replace(label, '').strip(' -')
        if nation and nation != optic_name:
            return nation
    
    # Method 3: Special cases for known leagues
    optic_upper = optic_name.upper()
    
    # US Sports
    if any(league in optic_upper for league in ['NFL', 'NBA', 'MLB', 'NHL', 'MLS', 'WNBA']):
        return 'USA'
    
    # European competitions
    if 'UEFA' in optic_upper:
        return 'Europe'
    
    if 'EURO ' in optic_upper and 'LEAGUE' in optic_upper:
        return 'Europe'
    
    # International
    if any(word in optic_upper for word in ['WORLD CUP', 'FIFA', 'INTERNATIONAL']):
        return 'International'
    
    # South America
    if 'COPA AMERICA' in optic_upper or 'CONMEBOL' in optic_upper:
        return 'South America'
    
    # Try to extract country name patterns
    # Common country names that might appear without " - "
    countries = [
        'England', 'Spain', 'Italy', 'Germany', 'France', 'Portugal', 'Netherlands',
        'Belgium', 'Scotland', 'Turkey', 'Greece', 'Russia', 'Ukraine', 'Poland',
        'Brazil', 'Argentina', 'Mexico', 'Japan', 'China', 'Korea', 'Australia',
        'USA', 'Canada'
    ]
    
    for country in countries:
        if country.upper() in optic_upper:
            return country
    
    return None

def get_api_nation_mappings():
    """Get nation mappings from API with intelligent inference"""
    try:
        # Get sports data
        response = requests.get('https://api.overtime.io/overtime-v2/sports', timeout=10)
        if response.status_code != 200:
            logger.error(f"Sports API returned {response.status_code}")
            return {}, {}
        
        sports_data = response.json()
        sport_id_to_nation = {}
        league_to_nation = {}
        
        # Process each sport entry
        for sport_id, info in sports_data.items():
            sport = info.get('sport', 'Unknown')
            label = info.get('label', '')
            optic_name = info.get('opticOddsName', '')
            
            # Infer nation
            nation = infer_nation_from_optic_name(optic_name, label)
            
            if nation:
                sport_id_to_nation[sport_id] = {
                    'nation': nation,
                    'league': label or optic_name,
                    'sport': sport,
                    'optic_name': optic_name
                }
                
                # Also create league mapping
                if label:
                    league_to_nation[label] = nation
                
                # Map the optic_name too if different from label
                if optic_name and optic_name != label:
                    league_to_nation[optic_name] = nation
        
        return sport_id_to_nation, league_to_nation
        
    except Exception as e:
        logger.error(f"Error getting API mappings: {e}")
        return {}, {}

def sync_nations_with_inference():
    """Update database with API-inferred nation data"""
    
    # Get mappings
    logger.info("=== Getting nation mappings from API with inference ===")
    sport_id_to_nation, league_to_nation = get_api_nation_mappings()
    
    logger.info(f"Created {len(sport_id_to_nation)} sport ID mappings")
    logger.info(f"Created {len(league_to_nation)} league mappings")
    
    # Show samples
    logger.info("\n=== Sample Nation Inferences ===")
    for sport_id, data in list(sport_id_to_nation.items())[:15]:
        logger.info(f"  {data['optic_name']} → {data['nation']}")
    
    with db_manager.get_db_session() as db:
        # Get games to map sportId to markets
        logger.info("\n=== Fetching games from API ===")
        try:
            response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
            if response.status_code != 200:
                logger.error(f"Games API returned {response.status_code}")
                return
            
            games = response.json()
            logger.info(f"Found {len(games)} games")
            
            # Build game_id to nation mapping
            game_to_nation = {}
            missing_sport_ids = set()
            
            for game_id, info in games.items():
                sport_id = info.get('sportId')
                if sport_id in sport_id_to_nation:
                    game_to_nation[game_id] = sport_id_to_nation[sport_id]['nation']
                else:
                    missing_sport_ids.add(sport_id)
            
            logger.info(f"Mapped {len(game_to_nation)} games to nations")
            if missing_sport_ids:
                logger.warning(f"Missing sport IDs: {missing_sport_ids}")
            
            # Update markets using game mappings
            logger.info("\n=== Updating markets with inferred nations ===")
            
            # Method 1: Update using game_id mapping
            batch_updated = 0
            for game_id, nation in game_to_nation.items():
                result = db.query(Market).filter(
                    Market.source_id == game_id
                ).update({'nation': nation})
                
                if result > 0:
                    batch_updated += result
                
                if batch_updated % 100 == 0 and batch_updated > 0:
                    db.commit()
                    logger.info(f"  Updated {batch_updated} markets...")
            
            db.commit()
            logger.info(f"✅ Updated {batch_updated} markets using game ID mappings")
            
            # Method 2: Update remaining using league mappings
            logger.info("\n=== Updating remaining markets by league ===")
            league_updated = 0
            
            for league, nation in league_to_nation.items():
                result = db.query(Market).filter(
                    Market.league_name == league,
                    (Market.nation == None) | (Market.nation == '')
                ).update({'nation': nation})
                
                if result > 0:
                    logger.info(f"  {league} → {nation}: {result} markets")
                    league_updated += result
                    db.commit()
            
            logger.info(f"✅ Updated {league_updated} additional markets using league mappings")
            
            # Set defaults for any remaining
            logger.info("\n=== Setting defaults for unmapped markets ===")
            
            # Default nations by sport
            sport_defaults = {
                'Soccer': 'International',
                'Basketball': 'USA',
                'Baseball': 'USA',
                'Hockey': 'North America',
                'American Football': 'USA',
                'Tennis': 'International',
                'Golf': 'International',
                'Cricket': 'International',
                'Esports': 'Global',
                'MMA': 'International',
                'Handball': 'Europe'
            }
            
            default_updated = 0
            for sport, default_nation in sport_defaults.items():
                result = db.query(Market).filter(
                    Market.sport == sport,
                    (Market.nation == None) | (Market.nation == '')
                ).update({'nation': default_nation})
                
                if result > 0:
                    logger.info(f"  {sport} → {default_nation}: {result} markets")
                    default_updated += result
            
            db.commit()
            logger.info(f"✅ Set defaults for {default_updated} markets")
            
            # Final statistics
            logger.info("\n=== Final Nation Distribution ===")
            nation_stats = db.query(
                Market.nation,
                func.count(Market.source_id)
            ).group_by(
                Market.nation
            ).order_by(
                func.count(Market.source_id).desc()
            ).limit(25).all()
            
            total = sum(count for _, count in nation_stats)
            for nation, count in nation_stats:
                pct = count / total * 100
                logger.info(f"  {nation or 'Unknown'}: {count} markets ({pct:.1f}%)")
            
        except Exception as e:
            logger.error(f"Error syncing: {e}")
            db.rollback()

if __name__ == "__main__":
    sync_nations_with_inference()
    logger.info("\n✅ Nation sync complete! Data is now sourced from API with intelligent inference.")