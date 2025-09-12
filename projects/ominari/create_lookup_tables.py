#!/usr/bin/env python3
"""Create lookup tables for odds normalization."""

import logging
from sqlalchemy import text
from database_v2 import db_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_lookup_tables():
    """Create lookup tables for normalization."""
    logger.info("Creating lookup tables...")
    
    with db_manager.get_db_session() as db:
        # Create lookup tables
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS lu_bookmakers (
                id TINYINT PRIMARY KEY,
                name VARCHAR(50) UNIQUE NOT NULL
            )
        """))
        
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS lu_sources (
                id TINYINT PRIMARY KEY,
                name VARCHAR(50) UNIQUE NOT NULL
            )
        """))
        
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS lu_market_types (
                id TINYINT PRIMARY KEY,
                name VARCHAR(50) UNIQUE NOT NULL
            )
        """))
        
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS lu_outcomes (
                id TINYINT PRIMARY KEY,
                name VARCHAR(10) UNIQUE NOT NULL
            )
        """))
        
        # Insert fixed outcomes
        db.execute(text("DELETE FROM lu_outcomes"))
        db.execute(text("""
            INSERT INTO lu_outcomes (id, name) VALUES 
            (0, 'option_1'),
            (1, 'option_2'), 
            (2, 'option_3')
        """))
        
        db.commit()
        logger.info("Lookup tables created")


def populate_lookup_tables():
    """Populate lookup tables from existing data."""
    logger.info("Populating lookup tables from existing data...")
    
    with db_manager.get_db_session() as db:
        # Get unique bookmakers
        logger.info("Loading bookmakers...")
        bookmakers = db.execute(text("""
            SELECT DISTINCT bookmaker 
            FROM odd 
            WHERE bookmaker IS NOT NULL
            ORDER BY bookmaker
            LIMIT 255
        """)).fetchall()
        
        logger.info(f"Found {len(bookmakers)} unique bookmakers")
        
        # Insert bookmakers
        for i, (name,) in enumerate(bookmakers):
            try:
                db.execute(text("""
                    INSERT OR IGNORE INTO lu_bookmakers (id, name) 
                    VALUES (:id, :name)
                """), {"id": i, "name": name})
            except Exception as e:
                logger.error(f"Error inserting bookmaker {name}: {e}")
        
        # Get unique sources
        logger.info("Loading sources...")
        sources = db.execute(text("""
            SELECT DISTINCT source 
            FROM odd 
            WHERE source IS NOT NULL
            ORDER BY source
            LIMIT 255
        """)).fetchall()
        
        logger.info(f"Found {len(sources)} unique sources")
        
        # Insert sources
        for i, (name,) in enumerate(sources):
            try:
                db.execute(text("""
                    INSERT OR IGNORE INTO lu_sources (id, name) 
                    VALUES (:id, :name)
                """), {"id": i, "name": name})
            except Exception as e:
                logger.error(f"Error inserting source {name}: {e}")
        
        # Get unique market types
        logger.info("Loading market types...")
        market_types = db.execute(text("""
            SELECT DISTINCT market_type 
            FROM odd 
            WHERE market_type IS NOT NULL
            ORDER BY market_type
            LIMIT 255
        """)).fetchall()
        
        logger.info(f"Found {len(market_types)} unique market types")
        
        # Insert market types
        for i, (name,) in enumerate(market_types):
            try:
                db.execute(text("""
                    INSERT OR IGNORE INTO lu_market_types (id, name) 
                    VALUES (:id, :name)
                """), {"id": i, "name": name})
            except Exception as e:
                logger.error(f"Error inserting market type {name}: {e}")
        
        db.commit()
        logger.info("Lookup tables populated")
        
        # Show summary
        logger.info("\nLookup table summary:")
        
        count = db.execute(text("SELECT COUNT(*) FROM lu_bookmakers")).scalar()
        logger.info(f"  Bookmakers: {count}")
        
        count = db.execute(text("SELECT COUNT(*) FROM lu_sources")).scalar()
        logger.info(f"  Sources: {count}")
        
        count = db.execute(text("SELECT COUNT(*) FROM lu_market_types")).scalar()
        logger.info(f"  Market types: {count}")
        
        count = db.execute(text("SELECT COUNT(*) FROM lu_outcomes")).scalar()
        logger.info(f"  Outcomes: {count}")


def verify_lookups():
    """Verify lookup tables are correct."""
    logger.info("\nVerifying lookup tables...")
    
    with db_manager.get_db_session() as db:
        # Sample some mappings
        samples = db.execute(text("""
            SELECT b.id, b.name, COUNT(*) as odds_count
            FROM lu_bookmakers b
            JOIN odd o ON o.bookmaker = b.name
            GROUP BY b.id, b.name
            ORDER BY odds_count DESC
            LIMIT 5
        """)).fetchall()
        
        logger.info("\nTop 5 bookmakers by odds count:")
        for id, name, count in samples:
            logger.info(f"  {id}: {name} ({count:,} odds)")


if __name__ == "__main__":
    create_lookup_tables()
    populate_lookup_tables()
    verify_lookups()