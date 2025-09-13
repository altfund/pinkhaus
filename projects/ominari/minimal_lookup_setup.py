#!/usr/bin/env python3
"""Minimal lookup table setup with hardcoded common values."""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_minimal_lookups():
    """Create lookup tables with commonly known values."""
    logger.info("Creating minimal lookup tables...")
    
    conn = sqlite3.connect('sport_odds.db', timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    
    try:
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
        
        # Create normalized odds table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odds_normalized (
                market_id TEXT NOT NULL,
                bookmaker_id INTEGER NOT NULL,
                outcome_id INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                source_id INTEGER NOT NULL DEFAULT 0,
                market_type_id INTEGER NOT NULL DEFAULT 0,
                position INTEGER DEFAULT 0,
                line_x100 INTEGER,
                decimal_odds_x1000 INTEGER NOT NULL,
                american_odds INTEGER,
                implied_x10000 INTEGER,
                PRIMARY KEY (market_id, bookmaker_id, outcome_id, updated_at)
            ) WITHOUT ROWID
        """)
        
        # Clear existing data
        cursor.execute("DELETE FROM lu_bookmakers")
        cursor.execute("DELETE FROM lu_sources")
        cursor.execute("DELETE FROM lu_market_types")
        cursor.execute("DELETE FROM lu_outcomes")
        
        # Insert common bookmakers (based on Overtime Sports knowledge)
        bookmakers = [
            'DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet', 
            'BetRivers', 'WynnBET', 'Unibet', 'Betfred', 'TwinSpires',
            'FOX Bet', 'Barstool', 'BetUS', 'MyBookie', 'Bovada',
            'SportsBetting.ag', 'BetOnline', 'Heritage', 'Pinnacle', 'Bet365',
            'William Hill', '888sport', 'Betway'
        ]
        
        for i, name in enumerate(bookmakers):
            cursor.execute("INSERT INTO lu_bookmakers (id, name) VALUES (?, ?)", (i, name))
        
        # Insert common sources
        sources = ['overtime_markets', 'thalesmarket', 'blockchain']
        for i, name in enumerate(sources):
            cursor.execute("INSERT INTO lu_sources (id, name) VALUES (?, ?)", (i, name))
        
        # Insert common market types
        market_types = ['winner', 'moneyline', 'spread', 'total', 'props']
        for i, name in enumerate(market_types):
            cursor.execute("INSERT INTO lu_market_types (id, name) VALUES (?, ?)", (i, name))
        
        # Insert fixed outcomes
        outcomes = ['option_1', 'option_2', 'option_3']
        for i, name in enumerate(outcomes):
            cursor.execute("INSERT INTO lu_outcomes (id, name) VALUES (?, ?)", (i, name))
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_odds_norm_market ON odds_normalized(market_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_odds_norm_time ON odds_normalized(updated_at)")
        
        conn.commit()
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM lu_bookmakers")
        bookmaker_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM lu_sources")
        source_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM lu_market_types")
        market_type_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM lu_outcomes")
        outcome_count = cursor.fetchone()[0]
        
        logger.info(f"✅ Created lookup tables:")
        logger.info(f"  Bookmakers: {bookmaker_count}")
        logger.info(f"  Sources: {source_count}")
        logger.info(f"  Market types: {market_type_count}")
        logger.info(f"  Outcomes: {outcome_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    create_minimal_lookups()