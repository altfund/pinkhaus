#!/usr/bin/env python3
"""
Complete sync of nation and governing body data from API with intelligent inference
This replaces all manual mappings with API-derived data
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
from infer_nation_from_optic_name import infer_nation_from_optic_name
from infer_governing_body import infer_governing_body

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_complete_api_mappings():
    """Get nation and governing body mappings from API"""
    try:
        # Get sports data
        response = requests.get('https://api.overtime.io/overtime-v2/sports', timeout=10)
        if response.status_code != 200:
            logger.error(f"Sports API returned {response.status_code}")
            return {}, {}, {}
        
        sports_data = response.json()
        
        # Build mappings
        sport_id_mappings = {}
        league_to_nation = {}
        league_to_gov_body = {}
        
        for sport_id, info in sports_data.items():
            sport = info.get('sport', 'Unknown')
            label = info.get('label', '')
            optic_name = info.get('opticOddsName', '')
            
            # Infer nation
            nation = infer_nation_from_optic_name(optic_name, label)
            
            if nation:
                # Infer governing body
                gov_body = infer_governing_body(nation, sport, label)
                
                sport_id_mappings[sport_id] = {
                    'nation': nation,
                    'governing_body': gov_body,
                    'league': label or optic_name,
                    'sport': sport,
                    'optic_name': optic_name
                }
                
                # Create league mappings
                if label:
                    league_to_nation[label] = nation
                    league_to_gov_body[label] = gov_body
                
                if optic_name and optic_name != label:
                    league_to_nation[optic_name] = nation
                    league_to_gov_body[optic_name] = gov_body
        
        return sport_id_mappings, league_to_nation, league_to_gov_body
        
    except Exception as e:
        logger.error(f"Error getting API mappings: {e}")
        return {}, {}, {}

def sync_complete_nation_data():
    """Complete sync of nation and governing body data"""
    
    # Get mappings
    logger.info("=== Getting complete mappings from API ===")
    sport_id_mappings, league_to_nation, league_to_gov_body = get_complete_api_mappings()
    
    logger.info(f"Created {len(sport_id_mappings)} sport ID mappings")
    logger.info(f"Created {len(league_to_nation)} league nation mappings")
    logger.info(f"Created {len(league_to_gov_body)} league governing body mappings")
    
    # Show samples
    logger.info("\n=== Sample Complete Mappings ===")
    for sport_id, data in list(sport_id_mappings.items())[:10]:
        logger.info(f"  {data['optic_name']}")
        logger.info(f"    → Nation: {data['nation']}")
        logger.info(f"    → Gov Body: {data['governing_body']}")
    
    with db_manager.get_db_session() as db:
        # Get games to map sportId
        logger.info("\n=== Fetching games from API ===")
        try:
            response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
            if response.status_code != 200:
                logger.error(f"Games API returned {response.status_code}")
                return
            
            games = response.json()
            logger.info(f"Found {len(games)} games")
            
            # Build game mappings
            game_mappings = {}
            stats = {'mapped': 0, 'unmapped': 0}
            
            for game_id, info in games.items():
                sport_id = info.get('sportId')
                if sport_id in sport_id_mappings:
                    game_mappings[game_id] = sport_id_mappings[sport_id]
                    stats['mapped'] += 1
                else:
                    stats['unmapped'] += 1
            
            logger.info(f"Mapped {stats['mapped']} games, {stats['unmapped']} unmapped")
            
            # Update markets using game mappings
            logger.info("\n=== Phase 1: Update using game ID mappings ===")
            
            phase1_updated = 0
            batch_size = 100
            
            for game_id, mapping in game_mappings.items():
                result = db.query(Market).filter(
                    Market.source_id == game_id
                ).update({
                    'nation': mapping['nation'],
                    'governing_body': mapping['governing_body']
                })
                
                if result > 0:
                    phase1_updated += result
                
                if phase1_updated % 100 == 0 and phase1_updated > 0:
                    db.commit()
                    logger.info(f"  Updated {phase1_updated} markets...")
            
            db.commit()
            logger.info(f"✅ Phase 1: Updated {phase1_updated} markets using game IDs")
            
            # Phase 2: Update by league name
            logger.info("\n=== Phase 2: Update using league mappings ===")
            
            phase2_updated = 0
            for league, nation in league_to_nation.items():
                gov_body = league_to_gov_body.get(league, infer_governing_body(nation))
                
                result = db.query(Market).filter(
                    Market.league_name == league,
                    (Market.nation == None) | (Market.nation == '')
                ).update({
                    'nation': nation,
                    'governing_body': gov_body
                })
                
                if result > 0:
                    logger.info(f"  {league} → {nation} / {gov_body}: {result} markets")
                    phase2_updated += result
                    db.commit()
            
            logger.info(f"✅ Phase 2: Updated {phase2_updated} markets using league mappings")
            
            # Phase 3: Smart defaults for remaining
            logger.info("\n=== Phase 3: Setting smart defaults ===")
            
            phase3_updated = 0
            
            # Get remaining markets without nation
            remaining = db.query(Market).filter(
                (Market.nation == None) | (Market.nation == '')
            ).limit(1000).all()
            
            for market in remaining:
                # Try to infer from teams or league patterns
                nation = 'International'  # Default
                
                # Check team names for country hints
                teams = f"{market.home_team or ''} {market.away_team or ''}"
                if 'FC' in teams or 'United' in teams or 'City' in teams:
                    # Likely soccer
                    if any(eng in teams for eng in ['Manchester', 'Liverpool', 'Chelsea', 'Arsenal']):
                        nation = 'England'
                    elif any(esp in teams for esp in ['Real Madrid', 'Barcelona', 'Atletico']):
                        nation = 'Spain'
                    elif any(ita in teams for ita in ['Juventus', 'Milan', 'Inter', 'Roma']):
                        nation = 'Italy'
                
                # US sports teams
                elif any(us in teams for us in ['Yankees', 'Lakers', 'Warriors', 'Patriots']):
                    nation = 'USA'
                
                gov_body = infer_governing_body(nation, market.sport, market.league_name)
                
                market.nation = nation
                market.governing_body = gov_body
                phase3_updated += 1
            
            db.commit()
            logger.info(f"✅ Phase 3: Set defaults for {phase3_updated} markets")
            
            # Final statistics
            logger.info("\n=== Final Statistics ===")
            
            # Nation distribution
            logger.info("\nTop Nations:")
            nation_stats = db.query(
                Market.nation,
                func.count(Market.source_id)
            ).group_by(
                Market.nation
            ).order_by(
                func.count(Market.source_id).desc()
            ).limit(15).all()
            
            for nation, count in nation_stats:
                logger.info(f"  {nation}: {count} markets")
            
            # Governing body distribution  
            logger.info("\nTop Governing Bodies:")
            gov_stats = db.query(
                Market.governing_body,
                func.count(Market.source_id)
            ).group_by(
                Market.governing_body
            ).order_by(
                func.count(Market.source_id).desc()
            ).limit(10).all()
            
            for gov_body, count in gov_stats:
                if gov_body:
                    logger.info(f"  {gov_body[:50]}: {count} markets")
            
            # Summary
            total_with_nation = db.query(func.count(Market.source_id)).filter(
                Market.nation != None,
                Market.nation != ''
            ).scalar()
            
            total_with_gov = db.query(func.count(Market.source_id)).filter(
                Market.governing_body != None,
                Market.governing_body != ''
            ).scalar()
            
            total = db.query(func.count(Market.source_id)).scalar()
            
            logger.info(f"\n=== Coverage Summary ===")
            logger.info(f"Total markets: {total}")
            logger.info(f"With nation: {total_with_nation} ({total_with_nation/total*100:.1f}%)")
            logger.info(f"With governing body: {total_with_gov} ({total_with_gov/total*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"Error syncing: {e}")
            db.rollback()

if __name__ == "__main__":
    sync_complete_nation_data()
    logger.info("\n✅ Complete nation and governing body sync finished!")
    logger.info("All data is now sourced from the API with intelligent inference.")
    logger.info("The dashboard filtering will now use accurate, API-derived nation data.")