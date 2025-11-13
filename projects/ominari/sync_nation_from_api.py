#!/usr/bin/env python3
"""
Sync nation/country data from Overtime API sports endpoint
This is the PROPER way to get nation data - from the API itself!
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
from database_v2 import db_manager
from models import Market
from sqlalchemy import func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_nation_mappings_from_api():
    """Get nation mappings from Overtime API sports endpoint"""
    try:
        response = requests.get('https://api.overtime.io/overtime-v2/sports', timeout=10)
        if response.status_code != 200:
            logger.error(f"API returned status {response.status_code}")
            return {}
        
        sports_data = response.json()
        
        # Create mappings: sportId -> nation/country
        sport_id_to_nation = {}
        league_to_nation = {}
        
        for sport_id, info in sports_data.items():
            optic_name = info.get('opticOddsName', '')
            label = info.get('label', '')
            
            # Extract country from opticOddsName (format: "Country - League")
            if ' - ' in optic_name:
                parts = optic_name.split(' - ', 1)  # Split only on first occurrence
                if len(parts) == 2:
                    country = parts[0].strip()
                    league = parts[1].strip()
                    
                    sport_id_to_nation[sport_id] = {
                        'nation': country,
                        'league': league,
                        'sport': info.get('sport', 'Unknown')
                    }
                    
                    # Also map by league name
                    if label:
                        league_to_nation[label] = country
                    if league:
                        league_to_nation[league] = country
            
            # Handle special cases without country prefix
            elif optic_name:
                # Default mappings for common leagues
                if any(x in optic_name.upper() for x in ['NFL', 'NBA', 'MLB', 'NHL', 'MLS']):
                    sport_id_to_nation[sport_id] = {
                        'nation': 'USA',
                        'league': optic_name,
                        'sport': info.get('sport', 'Unknown')
                    }
                elif 'UEFA' in optic_name or 'Champions League' in optic_name:
                    sport_id_to_nation[sport_id] = {
                        'nation': 'Europe',
                        'league': optic_name,
                        'sport': info.get('sport', 'Unknown')
                    }
                
                if label:
                    league_to_nation[label] = sport_id_to_nation.get(sport_id, {}).get('nation', 'International')
        
        return sport_id_to_nation, league_to_nation
    
    except Exception as e:
        logger.error(f"Error fetching sports data: {e}")
        return {}, {}

def sync_nations_from_api():
    """Sync nation data using API sportId mappings"""
    
    # Get nation mappings from API
    logger.info("=== Fetching nation mappings from Overtime API ===")
    sport_id_to_nation, league_to_nation = get_nation_mappings_from_api()
    logger.info(f"Loaded {len(sport_id_to_nation)} sport ID mappings")
    logger.info(f"Loaded {len(league_to_nation)} league mappings")
    
    # Show sample mappings
    logger.info("\n=== Sample Sport ID to Nation Mappings ===")
    for sport_id, data in list(sport_id_to_nation.items())[:10]:
        logger.info(f"  {sport_id}: {data['nation']} - {data['league']} ({data['sport']})")
    
    with db_manager.get_db_session() as db:
        # First, get games from API to build game_id -> sportId mapping
        logger.info("\n=== Fetching games to get sportId mappings ===")
        try:
            response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
            if response.status_code != 200:
                logger.error(f"Games API returned status {response.status_code}")
                return
            
            games = response.json()
            logger.info(f"Found {len(games)} games from API")
            
            # Build mapping of game_id to sportId and nation
            game_to_nation = {}
            nation_stats = {}
            
            for game_id, info in games.items():
                sport_id = info.get('sportId')
                if sport_id and sport_id in sport_id_to_nation:
                    nation_data = sport_id_to_nation[sport_id]
                    game_to_nation[game_id] = {
                        'nation': nation_data['nation'],
                        'league': nation_data['league'],
                        'sport': nation_data['sport']
                    }
                    
                    # Track stats
                    nation = nation_data['nation']
                    if nation not in nation_stats:
                        nation_stats[nation] = 0
                    nation_stats[nation] += 1
            
            logger.info(f"\n=== Nation distribution from API ===")
            for nation, count in sorted(nation_stats.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {nation}: {count} games")
            
            # Update markets with API-based nation data
            logger.info(f"\n=== Updating markets with API nation data ===")
            
            total_updated = 0
            batch_size = 100
            
            # Process markets in batches
            offset = 0
            while True:
                markets = db.query(Market).filter(
                    Market.source_id.in_(game_to_nation.keys())
                ).offset(offset).limit(batch_size).all()
                
                if not markets:
                    break
                
                for market in markets:
                    if market.source_id in game_to_nation:
                        nation_data = game_to_nation[market.source_id]
                        market.nation = nation_data['nation']
                        # Don't override league_name if it's already good
                        if not market.league_name or market.league_name in ['N/A', 'Regular Season']:
                            market.league_name = nation_data['league']
                        total_updated += 1
                
                db.commit()
                logger.info(f"  Updated batch {offset//batch_size + 1}: {len(markets)} markets")
                offset += batch_size
            
            logger.info(f"\n✅ Updated {total_updated} markets with API nation data")
            
            # Update remaining markets using league mappings
            logger.info("\n=== Updating remaining markets using league mappings ===")
            
            remaining_updated = 0
            for league, nation in league_to_nation.items():
                result = db.query(Market).filter(
                    Market.league_name == league,
                    (Market.nation == None) | (Market.nation == '')
                ).update({'nation': nation})
                
                if result > 0:
                    logger.info(f"  {league} → {nation}: {result} markets")
                    remaining_updated += result
            
            db.commit()
            logger.info(f"\n✅ Updated {remaining_updated} additional markets using league mappings")
            
            # Show final nation distribution
            logger.info("\n=== Final Nation Distribution ===")
            nation_counts = db.query(
                Market.nation,
                func.count(Market.source_id)
            ).filter(
                Market.sport == 'Soccer'
            ).group_by(
                Market.nation
            ).order_by(
                func.count(Market.source_id).desc()
            ).limit(20).all()
            
            for nation, count in nation_counts:
                logger.info(f"  {nation or 'Unknown'}: {count} markets")
            
        except Exception as e:
            logger.error(f"Error syncing nations: {e}")
            db.rollback()

if __name__ == "__main__":
    sync_nations_from_api()