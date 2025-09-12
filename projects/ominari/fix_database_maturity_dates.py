#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix empty string maturity_date values in the database.
"""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_empty_maturity_dates():
    """Update empty string maturity_dates to NULL in the database."""
    conn = sqlite3.connect('sport_odds.db')
    cursor = conn.cursor()
    
    try:
        # First, check how many records have empty string maturity_dates
        cursor.execute("""
            SELECT COUNT(*) FROM market 
            WHERE maturity_date = '' OR maturity_date IS NULL
        """)
        count = cursor.fetchone()[0]
        logger.info(f"Found {count} markets with empty or NULL maturity_date")
        
        if count > 0:
            # Update empty strings to NULL
            cursor.execute("""
                UPDATE market 
                SET maturity_date = NULL 
                WHERE maturity_date = ''
            """)
            updated = cursor.rowcount
            logger.info(f"Updated {updated} records with empty maturity_date to NULL")
            
            # Also check for other problematic date values
            cursor.execute("""
                SELECT id, source_id, maturity_date 
                FROM market 
                WHERE maturity_date IS NOT NULL 
                AND maturity_date != '' 
                LIMIT 10
            """)
            sample = cursor.fetchall()
            logger.info("Sample of non-empty maturity_dates:")
            for row in sample:
                logger.info(f"  ID {row[0]}: {row[2]}")
            
            conn.commit()
            logger.info("Database updates committed successfully")
        else:
            logger.info("No empty maturity_dates found")
            
    except Exception as e:
        logger.error(f"Error fixing database: {e}")
        conn.rollback()
    finally:
        conn.close()


def check_database_schema():
    """Check the schema of the markets table."""
    conn = sqlite3.connect('sport_odds.db')
    cursor = conn.cursor()
    
    try:
        # Get table info
        cursor.execute("PRAGMA table_info(market)")
        columns = cursor.fetchall()
        
        logger.info("Markets table schema:")
        for col in columns:
            logger.info(f"  {col[1]} - {col[2]} {'NOT NULL' if col[3] else 'NULL OK'}")
            
        # Check data types stored
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN maturity_date IS NULL THEN 1 END) as null_count,
                COUNT(CASE WHEN maturity_date = '' THEN 1 END) as empty_count,
                COUNT(CASE WHEN maturity_date IS NOT NULL AND maturity_date != '' THEN 1 END) as valid_count
            FROM market
        """)
        stats = cursor.fetchone()
        logger.info("\nMaturity date statistics:")
        logger.info(f"  Total records: {stats[0]}")
        logger.info(f"  NULL values: {stats[1]}")
        logger.info(f"  Empty strings: {stats[2]}")
        logger.info(f"  Valid values: {stats[3]}")
        
    finally:
        conn.close()


if __name__ == "__main__":
    logger.info("Checking database schema...")
    check_database_schema()
    
    logger.info("\nFixing empty maturity dates...")
    fix_empty_maturity_dates()
    
    logger.info("\nRechecking after fix...")
    check_database_schema()