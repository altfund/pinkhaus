#!/usr/bin/env python3
"""Safe lookup table creation with timeouts and small queries."""

import logging
import sqlite3
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def get_safe_db_connection():
    """Get a safe database connection with optimizations."""
    conn = sqlite3.connect('sport_odds.db', timeout=30.0)
    conn.row_factory = sqlite3.Row
    
    # Apply read optimizations
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL") 
    conn.execute("PRAGMA cache_size=50000")  # 200MB cache
    conn.execute("PRAGMA temp_store=MEMORY")
    
    try:
        yield conn
    finally:
        conn.close()


def create_lookup_schema():
    """Create lookup table schemas."""
    logger.info("Creating lookup table schemas...")
    
    with get_safe_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create lookup tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lu_bookmakers (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lu_sources (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lu_market_types (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lu_outcomes (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            )
        """)
        
        # Insert fixed outcomes
        cursor.execute("DELETE FROM lu_outcomes")
        cursor.execute("""
            INSERT INTO lu_outcomes (id, name) VALUES 
            (0, 'option_1'),
            (1, 'option_2'), 
            (2, 'option_3')
        """)
        
        # Create normalized odds table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odds_normalized (
                market_id TEXT NOT NULL,
                bookmaker_id INTEGER NOT NULL,
                outcome_id INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                source_id INTEGER NOT NULL,
                market_type_id INTEGER NOT NULL,
                position INTEGER DEFAULT 0,
                line_x100 INTEGER,
                decimal_odds_x1000 INTEGER NOT NULL,
                american_odds INTEGER,
                implied_x10000 INTEGER,
                PRIMARY KEY (market_id, bookmaker_id, outcome_id, updated_at)
            ) WITHOUT ROWID
        """)
        
        # Create indexes for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_odds_norm_market 
            ON odds_normalized(market_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_odds_norm_time 
            ON odds_normalized(updated_at)
        """)
        
        conn.commit()
        logger.info("Lookup schemas created successfully")


def populate_bookmakers():
    """Populate bookmakers using safe sampling."""
    logger.info("Populating bookmakers...")
    
    with get_safe_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get unique bookmakers using safe sampling approach
        cursor.execute("""
            SELECT DISTINCT bookmaker 
            FROM odd 
            WHERE bookmaker IS NOT NULL 
            AND rowid <= 500000
            ORDER BY bookmaker
        """)
        
        bookmakers = cursor.fetchall()
        logger.info(f"Found {len(bookmakers)} unique bookmakers (from sample)")
        
        # Clear existing
        cursor.execute("DELETE FROM lu_bookmakers")
        
        # Insert bookmakers
        for i, row in enumerate(bookmakers):
            cursor.execute("""
                INSERT INTO lu_bookmakers (id, name) VALUES (?, ?)
            """, (i, row['bookmaker']))
        
        conn.commit()
        logger.info(f"Inserted {len(bookmakers)} bookmakers")
        

def populate_sources():
    """Populate sources using safe sampling.""" 
    logger.info("Populating sources...")
    
    with get_safe_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get unique sources using safe sampling
        cursor.execute("""
            SELECT DISTINCT source 
            FROM odd 
            WHERE source IS NOT NULL
            AND rowid <= 500000
            ORDER BY source
        """)
        
        sources = cursor.fetchall()
        logger.info(f"Found {len(sources)} unique sources (from sample)")
        
        # Clear existing
        cursor.execute("DELETE FROM lu_sources")
        
        # Insert sources
        for i, row in enumerate(sources):
            cursor.execute("""
                INSERT INTO lu_sources (id, name) VALUES (?, ?)
            """, (i, row['source']))
        
        conn.commit()
        logger.info(f"Inserted {len(sources)} sources")


def populate_market_types():
    """Populate market types using safe sampling."""
    logger.info("Populating market types...")
    
    with get_safe_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get unique market types using safe sampling
        cursor.execute("""
            SELECT DISTINCT market_type 
            FROM odd 
            WHERE market_type IS NOT NULL
            AND rowid <= 500000
            ORDER BY market_type
        """)
        
        market_types = cursor.fetchall()
        logger.info(f"Found {len(market_types)} unique market types (from sample)")
        
        # Clear existing
        cursor.execute("DELETE FROM lu_market_types")
        
        # Insert market types
        for i, row in enumerate(market_types):
            cursor.execute("""
                INSERT INTO lu_market_types (id, name) VALUES (?, ?)
            """, (i, row['market_type']))
        
        conn.commit()
        logger.info(f"Inserted {len(market_types)} market types")


def verify_lookups():
    """Verify lookup tables."""
    logger.info("\nVerifying lookup tables...")
    
    with get_safe_db_connection() as conn:
        cursor = conn.cursor()
        
        # Count each lookup table
        cursor.execute("SELECT COUNT(*) FROM lu_bookmakers")
        bookmaker_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM lu_sources")  
        source_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM lu_market_types")
        market_type_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM lu_outcomes")
        outcome_count = cursor.fetchone()[0]
        
        logger.info(f"\nLookup table summary:")
        logger.info(f"  Bookmakers: {bookmaker_count}")
        logger.info(f"  Sources: {source_count}")
        logger.info(f"  Market types: {market_type_count}")
        logger.info(f"  Outcomes: {outcome_count}")
        
        # Show some sample bookmakers
        cursor.execute("SELECT id, name FROM lu_bookmakers ORDER BY id LIMIT 5")
        samples = cursor.fetchall()
        
        logger.info(f"\nSample bookmakers:")
        for row in samples:
            logger.info(f"  {row['id']}: {row['name']}")


def main():
    """Main execution."""
    try:
        logger.info("🏗️  Starting safe lookup table creation...")
        
        create_lookup_schema()
        populate_bookmakers()
        populate_sources() 
        populate_market_types()
        verify_lookups()
        
        logger.info("✅ Lookup tables created successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error creating lookup tables: {e}")
        raise


if __name__ == "__main__":
    main()