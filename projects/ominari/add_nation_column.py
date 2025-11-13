#!/usr/bin/env python3
"""
Add nation/country column to markets table and populate it based on league data
"""
import os

# Set environment for PostgreSQL on port 5999
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

import psycopg2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_nation_column():
    """Add nation column if it doesn't exist"""
    conn = psycopg2.connect(
        host=os.environ['PG_HOST'],
        port=os.environ['PG_PORT'],
        user=os.environ['PG_USER'],
        password=os.environ['PG_PASSWORD'],
        database=os.environ['PG_DB']
    )
    cur = conn.cursor()
    
    # Check if column exists
    cur.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name='market' AND column_name='nation'
    """)
    
    if not cur.fetchone():
        logger.info("Adding nation column to market table...")
        cur.execute("""
            ALTER TABLE market 
            ADD COLUMN nation VARCHAR(100)
        """)
        conn.commit()
        logger.info("✅ Nation column added")
    else:
        logger.info("Nation column already exists")
    
    # Also add governing_body column
    cur.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name='market' AND column_name='governing_body'
    """)
    
    if not cur.fetchone():
        logger.info("Adding governing_body column to market table...")
        cur.execute("""
            ALTER TABLE market 
            ADD COLUMN governing_body VARCHAR(100)
        """)
        conn.commit()
        logger.info("✅ Governing body column added")
    else:
        logger.info("Governing body column already exists")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    add_nation_column()