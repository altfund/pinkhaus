#!/usr/bin/env python3
"""
Create Migration Prerequisites - Fast Version

Creates required lookup tables efficiently for large database.
"""

import sqlite3
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_prerequisites_fast():
    """Create prerequisites with batch operations and minimal queries."""
    logger.info("Creating migration prerequisites (fast version)...")
    
    conn = sqlite3.connect('sport_odds.db', timeout=300.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=20000")
    conn.execute("PRAGMA temp_store=MEMORY")
    cursor = conn.cursor()
    
    try:
        # 1. Create all lookup tables first
        logger.info("Creating lookup tables...")
        
        tables = [
            """CREATE TABLE IF NOT EXISTS bookmakers_lookup (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS outcomes_lookup (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS sports_lookup (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS market_types_lookup (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS markets_normalized (
                market_id TEXT PRIMARY KEY,
                sport_id INTEGER,
                league TEXT,
                home_team TEXT,
                away_team TEXT,
                starts_at TEXT,
                is_finished INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sport_id) REFERENCES sports_lookup(id)
            )""",
            """CREATE TABLE IF NOT EXISTS odds_normalized_v2 (
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
            )"""
        ]
        
        for table_sql in tables:
            cursor.execute(table_sql)
        
        # 2. Populate lookups with limited queries
        logger.info("Populating bookmakers lookup (limited to 1000)...")
        cursor.execute("""
            INSERT OR IGNORE INTO bookmakers_lookup (name)
            SELECT DISTINCT bookmaker FROM odd 
            WHERE bookmaker IS NOT NULL 
            LIMIT 1000
        """)
        
        logger.info("Populating outcomes lookup (limited to 1000)...")
        cursor.execute("""
            INSERT OR IGNORE INTO outcomes_lookup (name)
            SELECT DISTINCT outcome FROM odd 
            WHERE outcome IS NOT NULL 
            LIMIT 1000
        """)
        
        logger.info("Populating sports lookup...")
        cursor.execute("""
            INSERT OR IGNORE INTO sports_lookup (name)
            SELECT DISTINCT sport FROM market 
            WHERE sport IS NOT NULL 
            LIMIT 100
        """)
        
        # 3. Add market types
        market_types = ['MONEYLINE', 'SPREAD', 'TOTAL', 'PROP', 'FUTURE', 'UNKNOWN']
        for mt in market_types:
            cursor.execute("INSERT OR IGNORE INTO market_types_lookup (name) VALUES (?)", (mt,))
        
        # 4. Create minimal indexes
        logger.info("Creating essential indexes...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_bookmakers_name ON bookmakers_lookup(name)",
            "CREATE INDEX IF NOT EXISTS idx_outcomes_name ON outcomes_lookup(name)",
            "CREATE INDEX IF NOT EXISTS idx_sports_name ON sports_lookup(name)"
        ]
        
        for idx in indexes:
            cursor.execute(idx)
        
        conn.commit()
        
        # 5. Report counts
        cursor.execute("SELECT COUNT(*) FROM bookmakers_lookup")
        bookmakers = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM outcomes_lookup")
        outcomes = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM sports_lookup")
        sports = cursor.fetchone()[0]
        
        logger.info(f"✅ Prerequisites created:")
        logger.info(f"   Bookmakers: {bookmakers}")
        logger.info(f"   Outcomes: {outcomes}")
        logger.info(f"   Sports: {sports}")
        
        # 6. Create migration state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS migration_state_v2 (
                id INTEGER PRIMARY KEY,
                last_processed_rowid INTEGER DEFAULT 0,
                total_rows INTEGER DEFAULT 0,
                started_at TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            INSERT OR IGNORE INTO migration_state_v2 (id, last_processed_rowid, started_at)
            VALUES (1, 0, ?)
        """, (datetime.now().isoformat(),))
        
        conn.commit()
        logger.info("✅ Migration state table created")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    create_prerequisites_fast()