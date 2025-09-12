#!/usr/bin/env python3
"""
Create Migration Prerequisites

Creates all required lookup tables and schemas before migration.
"""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_prerequisites():
    """Create all prerequisite tables and data."""
    logger.info("Creating migration prerequisites...")
    
    conn = sqlite3.connect('sport_odds.db')
    cursor = conn.cursor()
    
    try:
        # 1. Create lookup tables
        logger.info("Creating lookup tables...")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bookmakers_lookup (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS outcomes_lookup (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sports_lookup (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_types_lookup (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        """)
        
        # 2. Populate lookup tables from existing data
        logger.info("Populating bookmakers lookup...")
        cursor.execute("""
            INSERT OR IGNORE INTO bookmakers_lookup (name)
            SELECT DISTINCT bookmaker FROM odd WHERE bookmaker IS NOT NULL
        """)
        
        logger.info("Populating outcomes lookup...")
        cursor.execute("""
            INSERT OR IGNORE INTO outcomes_lookup (name)
            SELECT DISTINCT outcome FROM odd WHERE outcome IS NOT NULL
        """)
        
        logger.info("Populating sports lookup...")
        cursor.execute("""
            INSERT OR IGNORE INTO sports_lookup (name)
            SELECT DISTINCT sport FROM market WHERE sport IS NOT NULL
        """)
        
        # 3. Add default market types
        market_types = ['MONEYLINE', 'SPREAD', 'TOTAL', 'PROP', 'FUTURE']
        for mt in market_types:
            cursor.execute("INSERT OR IGNORE INTO market_types_lookup (name) VALUES (?)", (mt,))
        
        # 4. Create normalized tables if not exist
        logger.info("Creating normalized tables...")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS markets_normalized (
                market_id TEXT PRIMARY KEY,
                sport_id INTEGER,
                league TEXT,
                home_team TEXT,
                away_team TEXT,
                starts_at TEXT,
                is_finished INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sport_id) REFERENCES sports_lookup(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odds_normalized_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                bookmaker_id INTEGER NOT NULL,
                outcome_id INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                source_id INTEGER DEFAULT 0,
                market_type_id INTEGER DEFAULT 1,
                position INTEGER DEFAULT 0,
                line_x100 INTEGER,
                decimal_odds_x1000 INTEGER,
                american_odds INTEGER,
                implied_x10000 INTEGER,
                inserted_at INTEGER DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (market_id) REFERENCES markets_normalized(market_id),
                FOREIGN KEY (bookmaker_id) REFERENCES bookmakers_lookup(id),
                FOREIGN KEY (outcome_id) REFERENCES outcomes_lookup(id)
            )
        """)
        
        # 5. Create indexes
        logger.info("Creating indexes...")
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_bookmakers_name ON bookmakers_lookup(name)",
            "CREATE INDEX IF NOT EXISTS idx_outcomes_name ON outcomes_lookup(name)",
            "CREATE INDEX IF NOT EXISTS idx_odds_norm_v2_market ON odds_normalized_v2(market_id)",
            "CREATE INDEX IF NOT EXISTS idx_odds_norm_v2_composite ON odds_normalized_v2(market_id, bookmaker_id, outcome_id, updated_at)"
        ]
        
        for idx in indexes:
            cursor.execute(idx)
        
        conn.commit()
        
        # 6. Report statistics
        cursor.execute("SELECT COUNT(*) FROM bookmakers_lookup")
        bookmaker_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM outcomes_lookup")
        outcome_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sports_lookup")
        sport_count = cursor.fetchone()[0]
        
        logger.info(f"✅ Prerequisites created successfully:")
        logger.info(f"   Bookmakers: {bookmaker_count}")
        logger.info(f"   Outcomes: {outcome_count}")
        logger.info(f"   Sports: {sport_count}")
        
    except Exception as e:
        logger.error(f"Error creating prerequisites: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    create_prerequisites()