#!/usr/bin/env python3
"""
Show what the tabular dashboard displays
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

def show_preview():
    """Show preview of tabular format"""
    
    with db_manager.get_db_session() as db:
        query = db.query(Market, Odd).join(Odd, Market.source_id == Odd.source_id).filter(
            not_(and_(
                Odd.decimal_odds.in_([2.5, 2.8, 3.0])
            )),
            Market.sport == 'Soccer',
            Market.league_name != 'International Football',
            Market.nation.in_(['England', 'International', 'Europe'])
        ).order_by(Market.maturity_date.desc()).limit(30)
        
        results = query.all()
        
        # Group by match
        matches = {}
        for market, odd in results:
            match_key = f"{market.home_team} vs {market.away_team}"
            if match_key not in matches:
                matches[match_key] = {
                    'league': market.league_name,
                    'nation': market.nation,
                    'home': None,
                    'draw': None,
                    'away': None
                }
            matches[match_key][odd.outcome.lower()] = odd.decimal_odds
        
        logger.info("\n=== TABULAR DASHBOARD PREVIEW ===")
        logger.info("\nThe dashboard now shows markets in a clean table format:")
        logger.info("\n%-40s %-25s %8s %8s %8s" % ("MATCH", "LEAGUE/NATION", "HOME", "DRAW", "AWAY"))
        logger.info("=" * 100)
        
        for match, data in list(matches.items())[:10]:
            league_info = f"{data['league']} ({data['nation']})"
            home_odds = f"{data['home']:.3f}" if data['home'] else "-"
            draw_odds = f"{data['draw']:.3f}" if data['draw'] else "-"
            away_odds = f"{data['away']:.3f}" if data['away'] else "-"
            
            logger.info("%-40s %-25s %8s %8s %8s" % (
                match[:40], 
                league_info[:25],
                home_odds,
                draw_odds, 
                away_odds
            ))
        
        logger.info("\n=== FEATURES ===")
        logger.info("- Clean tabular layout (not cards)")
        logger.info("- All odds for a match on one row")
        logger.info("- Implied probabilities shown below odds")
        logger.info("- Sticky header for easy scrolling")
        logger.info("- Hover effects for better readability")
        logger.info("- Edge column ready for signal integration")
        logger.info("\nURL: http://localhost:8888")

if __name__ == "__main__":
    show_preview()