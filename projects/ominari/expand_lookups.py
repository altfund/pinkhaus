#!/usr/bin/env python3
"""Quickly expand lookup tables by sampling unique values."""

import logging
from sqlalchemy import create_engine, text
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def expand_lookups_fast():
    """Expand lookup tables using sampling for speed."""
    engine = create_engine("sqlite:///sport_odds.db")
    
    with engine.begin() as conn:
        logger.info("Finding unique values using sampling...")
        
        # Get a sample of unique bookmakers from recent records
        logger.info("Expanding bookmakers...")
        result = conn.execute(text("""
            INSERT OR IGNORE INTO lu_bookmakers (name)
            SELECT DISTINCT bookmaker 
            FROM (
                SELECT bookmaker FROM odd 
                WHERE bookmaker IS NOT NULL 
                ORDER BY id DESC 
                LIMIT 1000000
            )
            WHERE bookmaker NOT IN (SELECT name FROM lu_bookmakers)
        """))
        logger.info(f"Added {result.rowcount} new bookmakers")
        
        # Get unique sources (should be small)
        logger.info("Expanding sources...")
        result = conn.execute(text("""
            INSERT OR IGNORE INTO lu_sources (name)
            SELECT DISTINCT source 
            FROM odd 
            WHERE source IS NOT NULL 
            AND source NOT IN (SELECT name FROM lu_sources)
        """))
        logger.info(f"Added {result.rowcount} new sources")
        
        # Get unique market types (should be small)
        logger.info("Expanding market types...")
        result = conn.execute(text("""
            INSERT OR IGNORE INTO lu_market_types (name)
            SELECT DISTINCT market_type 
            FROM odd 
            WHERE market_type IS NOT NULL 
            AND market_type NOT IN (SELECT name FROM lu_market_types)
        """))
        logger.info(f"Added {result.rowcount} new market types")
        
        # Show final counts
        result = conn.execute(text("SELECT COUNT(*) FROM lu_bookmakers"))
        logger.info(f"Total bookmakers: {result.scalar()}")
        
        result = conn.execute(text("SELECT COUNT(*) FROM lu_sources"))
        logger.info(f"Total sources: {result.scalar()}")
        
        result = conn.execute(text("SELECT COUNT(*) FROM lu_market_types"))
        logger.info(f"Total market types: {result.scalar()}")

if __name__ == "__main__":
    expand_lookups_fast()