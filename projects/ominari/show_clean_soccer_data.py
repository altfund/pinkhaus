#!/usr/bin/env python3
"""
Show clean soccer data after all fixes
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
from models import Market, Odd
from sqlalchemy import and_, or_, not_, func
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def show_clean_data():
    """Show clean soccer data"""
    
    with db_manager.get_db_session() as db:
        # Excluding International Football league
        query = db.query(Market, Odd).join(Odd, Market.source_id == Odd.source_id).filter(
            # Exclude default odds
            not_(and_(
                Odd.decimal_odds.in_([2.5, 2.8, 3.0])
            )),
            Market.sport == 'Soccer',
            Market.league_name != 'International Football',  # Exclude mixed sport league
            Market.nation.in_(['England', 'International', 'Europe'])
        ).order_by(Market.maturity_date.desc()).limit(30)
        
        results = query.all()
        
        logger.info("\n=== Clean Soccer Data (Excluding International Football) ===")
        logger.info(f"Found {len(results)} matches\n")
        
        # Group by league
        leagues = {}
        nations = {}
        
        for market, odd in results:
            league = market.league_name or 'Unknown'
            nation = market.nation or 'Unknown'
            
            if league not in leagues:
                leagues[league] = []
            leagues[league].append(f"{market.home_team} vs {market.away_team}")
            
            if nation not in nations:
                nations[nation] = 0
            nations[nation] += 1
        
        logger.info("=== By League ===")
        for league, matches in sorted(leagues.items()):
            logger.info(f"\n{league} ({len(matches)} matches):")
            for match in matches[:5]:  # Show first 5
                logger.info(f"  - {match}")
            if len(matches) > 5:
                logger.info(f"  ... and {len(matches) - 5} more")
        
        logger.info("\n=== By Nation ===")
        for nation, count in sorted(nations.items()):
            logger.info(f"  {nation}: {count} matches")
        
        logger.info("\n=== Dashboard URL ===")
        logger.info("http://localhost:8888")
        logger.info("\nThe dashboard now shows:")
        logger.info("- Only Soccer matches (no American Football)")
        logger.info("- Filtered by England, International, Europe nations")
        logger.info("- Excluding 'International Football' league (contains mixed sports)")
        logger.info("- With clickable links to Overtime and blockchain explorers")

if __name__ == "__main__":
    show_clean_data()