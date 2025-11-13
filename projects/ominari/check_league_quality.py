#!/usr/bin/env python3
"""
Check league data quality and identify issues
"""
import os

# Set environment for PostgreSQL on port 5999
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market
from sqlalchemy import func, distinct
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_league_quality():
    with db_manager.get_db_session() as db:
        # Get total markets
        total_markets = db.query(Market).count()
        logger.info(f"Total markets in database: {total_markets}")
        
        # Check for problematic league names
        logger.info("\n=== Problematic League Names ===")
        
        # Generic leagues
        generic_leagues = ['N/A', 'Regular Season', 'Playoffs', 'Regular', 'Unknown', 
                          'Soccer League', 'Basketball League', 'Baseball League']
        
        for league in generic_leagues:
            count = db.query(Market).filter(Market.league_name == league).count()
            if count > 0:
                logger.info(f"  '{league}': {count} markets")
                # Show sample teams
                samples = db.query(Market).filter(Market.league_name == league).limit(3).all()
                for s in samples:
                    logger.info(f"    -> {s.sport}: {s.home_team} vs {s.away_team}")
        
        # Check NULL or empty leagues
        null_count = db.query(Market).filter(Market.league_name == None).count()
        empty_count = db.query(Market).filter(Market.league_name == '').count()
        logger.info(f"\n  NULL leagues: {null_count}")
        logger.info(f"  Empty string leagues: {empty_count}")
        
        # Get league distribution by sport
        logger.info("\n=== League Distribution by Sport ===")
        sports = db.query(distinct(Market.sport)).all()
        
        for sport_row in sports:
            sport = sport_row[0]
            if sport:
                logger.info(f"\n{sport}:")
                
                # Get top leagues for this sport
                leagues = db.query(
                    Market.league_name, 
                    func.count(Market.source_id).label('count')
                ).filter(
                    Market.sport == sport
                ).group_by(Market.league_name).order_by(func.count(Market.source_id).desc()).limit(10).all()
                
                for league, count in leagues:
                    league_display = league if league else "[NULL]"
                    logger.info(f"  - {league_display}: {count} markets")
        
        # Identify leagues that need mapping
        logger.info("\n=== Leagues Needing Better Names ===")
        
        # Get all leagues with generic names
        generic_filter = Market.league_name.in_(generic_leagues)
        needs_fixing = db.query(
            Market.sport,
            Market.league_name,
            func.count(Market.source_id).label('count')
        ).filter(generic_filter).group_by(
            Market.sport, Market.league_name
        ).order_by(func.count(Market.source_id).desc()).all()
        
        total_to_fix = 0
        for sport, league, count in needs_fixing:
            logger.info(f"  {sport} - '{league}': {count} markets")
            total_to_fix += count
        
        logger.info(f"\nTotal markets needing league fixes: {total_to_fix}")
        
        # Check for esports leagues labeled as soccer
        logger.info("\n=== Potential Misclassified Leagues ===")
        esports_leagues = ['ESEA', 'CCT', 'VCT', 'MPL', 'PGL', 'ESL', 'DACH']
        
        for league_pattern in esports_leagues:
            misclassified = db.query(Market).filter(
                Market.league_name.ilike(f'%{league_pattern}%'),
                Market.sport != 'Esports'
            ).count()
            if misclassified > 0:
                logger.info(f"  {league_pattern} pattern found in non-Esports: {misclassified} markets")

if __name__ == "__main__":
    check_league_quality()