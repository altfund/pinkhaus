#!/usr/bin/env python3
"""
Show soccer league distribution after fixes
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
from sqlalchemy import func
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def show_soccer_leagues():
    with db_manager.get_db_session() as db:
        # Get soccer league distribution
        leagues = db.query(
            Market.league_name, 
            func.count(Market.source_id).label('count')
        ).filter(
            Market.sport == 'Soccer'
        ).group_by(Market.league_name).order_by(func.count(Market.source_id).desc()).all()
        
        logger.info("=== Soccer League Distribution ===")
        logger.info(f"Total leagues: {len(leagues)}")
        logger.info("\nTop 30 Soccer Leagues:")
        
        for i, (league, count) in enumerate(leagues[:30]):
            league_display = league if league else "[Empty]"
            logger.info(f"{i+1:2d}. {league_display:40s} - {count:4d} markets")
        
        # Show specific major leagues
        major_leagues = [
            'Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1',
            'UEFA Champions League', 'Europa League', 'World Cup', 
            'European Championship', 'Copa America'
        ]
        
        logger.info("\n=== Major League Status ===")
        for league in major_leagues:
            count = db.query(Market).filter(
                Market.sport == 'Soccer',
                Market.league_name == league
            ).count()
            if count > 0:
                logger.info(f"✅ {league}: {count} markets")
            else:
                # Try case-insensitive
                count = db.query(Market).filter(
                    Market.sport == 'Soccer',
                    Market.league_name.ilike(f'%{league}%')
                ).count()
                if count > 0:
                    logger.info(f"⚠️  {league}: {count} markets (case variation)")
                else:
                    logger.info(f"❌ {league}: Not found")

if __name__ == "__main__":
    show_soccer_leagues()