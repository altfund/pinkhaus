#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find all empty string dates in all date columns.
"""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_empty_dates():
    """Find all empty date strings in the database."""
    conn = sqlite3.connect('sport_odds.db')
    cursor = conn.cursor()
    
    try:
        # Get all columns from market table
        cursor.execute("PRAGMA table_info(market)")
        columns = cursor.fetchall()
        
        # Find date/datetime columns
        date_columns = []
        for col in columns:
            col_name = col[1]
            col_type = col[2].upper()
            if 'DATE' in col_type or 'TIME' in col_type:
                date_columns.append(col_name)
        
        logger.info(f"Date columns in market table: {date_columns}")
        
        # Check each date column for empty strings
        for col in date_columns:
            cursor.execute(f"SELECT COUNT(*) FROM market WHERE {col} = ''")
            count = cursor.fetchone()[0]
            if count > 0:
                logger.info(f"\nFound {count} empty strings in column '{col}'")
                
                # Get some examples
                cursor.execute(f"SELECT source_id, {col} FROM market WHERE {col} = '' LIMIT 5")
                examples = cursor.fetchall()
                for source_id, value in examples:
                    logger.info(f"  {source_id}: '{value}'")
                    
                # Fix them
                cursor.execute(f"UPDATE market SET {col} = NULL WHERE {col} = ''")
                logger.info(f"  Updated {cursor.rowcount} records to NULL")
        
        conn.commit()
        logger.info("\nDatabase updates committed")
        
        # Verify no empty strings remain
        logger.info("\nVerifying all date columns...")
        for col in date_columns:
            cursor.execute(f"SELECT COUNT(*) FROM market WHERE {col} = ''")
            count = cursor.fetchone()[0]
            logger.info(f"  {col}: {count} empty strings")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    find_empty_dates()