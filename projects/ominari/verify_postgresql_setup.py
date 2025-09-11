#!/usr/bin/env python3
"""
Verify and complete PostgreSQL setup for hybrid architecture.
"""

import os
import logging
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


def verify_setup():
    """Verify PostgreSQL is set up correctly."""
    
    pg_host = os.getenv('PG_HOST', 'localhost')
    pg_port = os.getenv('PG_PORT', '5435')
    pg_user = os.getenv('PG_USER', 'ominari_user')
    pg_password = os.getenv('PG_PASSWORD', 'ominari_2025_secure')
    pg_db = os.getenv('PG_DB', 'ominari_production')
    
    try:
        conn = psycopg2.connect(
            host=pg_host,
            port=pg_port,
            user=pg_user,
            password=pg_password,
            database=pg_db
        )
        cursor = conn.cursor()
        
        logger.info("✅ Connected to PostgreSQL successfully")
        
        # Check schemas
        cursor.execute("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name IN ('blockchain', 'api', 'paper_trading', 'analytics')
            ORDER BY schema_name
        """)
        schemas = cursor.fetchall()
        
        logger.info(f"\n📋 Schemas found: {len(schemas)}")
        for schema in schemas:
            logger.info(f"   - {schema[0]}")
        
        # Check tables
        cursor.execute("""
            SELECT schemaname, tablename 
            FROM pg_tables 
            WHERE schemaname IN ('blockchain', 'api', 'paper_trading')
            ORDER BY schemaname, tablename
        """)
        tables = cursor.fetchall()
        
        logger.info(f"\n📊 Tables found: {len(tables)}")
        for schema, table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
            count = cursor.fetchone()[0]
            logger.info(f"   - {schema}.{table}: {count:,} rows")
        
        # Test insert
        logger.info("\n🧪 Testing data insertion...")
        
        # Insert test market
        test_market_id = f"test_hybrid_{datetime.now().timestamp()}"
        cursor.execute("""
            INSERT INTO blockchain.markets (
                chain_id, chain_name, market_address, market_id, 
                sport, league, home_team, away_team, start_time
            ) VALUES (
                10, 'optimism', '0xtest123', %s,
                'Soccer', 'EPL', 'Liverpool', 'Chelsea', NOW() + interval '1 day'
            )
            ON CONFLICT (market_id) DO NOTHING
            RETURNING id
        """, (test_market_id,))
        
        result = cursor.fetchone()
        if result:
            logger.info(f"   ✅ Test market created with ID: {result[0]}")
            
            # Insert test odds
            cursor.execute("""
                INSERT INTO blockchain.odds (
                    market_id, chain_id, outcome, decimal_odds, 
                    timestamp, block_number
                ) VALUES (
                    %s, 10, 'home', 2.10, NOW(), 12345678
                )
                RETURNING id
            """, (test_market_id,))
            
            odds_id = cursor.fetchone()
            logger.info(f"   ✅ Test odds created with ID: {odds_id[0]}")
        
        # Test function
        cursor.execute("SELECT * FROM get_latest_odds(%s)", (test_market_id,))
        odds_result = cursor.fetchall()
        logger.info(f"   ✅ Function test returned {len(odds_result)} rows")
        
        conn.commit()
        
        # Summary
        logger.info("\n✅ PostgreSQL setup verified successfully!")
        logger.info("\n📝 Connection details:")
        logger.info(f"   Host: {pg_host}")
        logger.info(f"   Port: {pg_port}")
        logger.info(f"   Database: {pg_db}")
        logger.info(f"   User: {pg_user}")
        
        logger.info("\n🚀 Ready for hybrid architecture:")
        logger.info("   1. SQLite for historical data (pre-6 months)")
        logger.info("   2. PostgreSQL for new blockchain data")
        logger.info("   3. Unified access through data access layer")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise


if __name__ == "__main__":
    verify_setup()