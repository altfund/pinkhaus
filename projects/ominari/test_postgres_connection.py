#!/usr/bin/env python3
"""
Test PostgreSQL connection for paper trading
"""

import os
import psycopg2
import logging

# Set PostgreSQL environment for paper trading
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_postgres_connection():
    """Test PostgreSQL connection and create paper trading tables."""
    logger.info("🧪 Testing PostgreSQL Connection for Paper Trading")
    logger.info("=" * 60)
    
    try:
        # Test basic connection
        conn = psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
        cur = conn.cursor()
        
        logger.info("✅ PostgreSQL connection successful!")
        
        # Test creating paper trading tables
        logger.info("📊 Creating paper trading tables...")
        
        # Create quotes table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS quotes (
                id SERIAL PRIMARY KEY,
                source_id TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                bid_price DECIMAL(10,4),
                bid_size DECIMAL(10,2),
                ask_price DECIMAL(10,4),
                ask_size DECIMAL(10,2),
                mid_price DECIMAL(10,4),
                spread DECIMAL(10,4),
                liquidity_score DECIMAL(5,4),
                raw_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create paper trading tables
        tables_created = []
        
        # Orders table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_orders (
                order_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                source_id TEXT NOT NULL,
                market_type TEXT,
                bet_name TEXT,
                side TEXT,
                size DECIMAL(15,6),
                limit_price DECIMAL(10,4),
                signal_name TEXT,
                expected_edge DECIMAL(6,4),
                status TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        tables_created.append("paper_orders")
        
        # Fills table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_fills (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                fill_price DECIMAL(10,4),
                fill_size DECIMAL(15,6),
                slippage DECIMAL(8,6),
                commission DECIMAL(10,4),
                market_impact DECIMAL(10,4),
                pnl DECIMAL(15,6),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                FOREIGN KEY (order_id) REFERENCES paper_orders(order_id)
            )
        """)
        tables_created.append("paper_fills")
        
        # Sessions table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_sessions (
                session_id TEXT PRIMARY KEY,
                start_time TIMESTAMP WITH TIME ZONE NOT NULL,
                end_time TIMESTAMP WITH TIME ZONE,
                initial_capital DECIMAL(15,6) NOT NULL,
                final_capital DECIMAL(15,6),
                status TEXT CHECK(status IN ('active', 'completed', 'archived')) DEFAULT 'active',
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        tables_created.append("paper_sessions")
        
        # Positions table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_positions (
                position_id TEXT PRIMARY KEY,
                session_id TEXT,
                market_id TEXT NOT NULL,
                bet_name TEXT,
                size DECIMAL(15,6) NOT NULL,
                entry_price DECIMAL(10,4) NOT NULL,
                entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
                exit_price DECIMAL(10,4),
                exit_time TIMESTAMP WITH TIME ZONE,
                status TEXT CHECK(status IN ('open', 'closed')) DEFAULT 'open',
                pnl DECIMAL(15,6),
                metadata JSONB,
                FOREIGN KEY (session_id) REFERENCES paper_sessions(session_id)
            )
        """)
        tables_created.append("paper_positions")
        
        conn.commit()
        
        logger.info(f"✅ Created {len(tables_created)} paper trading tables:")
        for table in tables_created:
            logger.info(f"   • {table}")
        
        # Test data insertion
        logger.info("\n🔄 Testing data insertion...")
        
        # Insert test session
        session_id = "TEST_SESSION_1"
        cur.execute("""
            INSERT INTO paper_sessions (session_id, start_time, initial_capital, metadata)
            VALUES (%s, NOW(), %s, %s)
            ON CONFLICT (session_id) DO UPDATE SET
            start_time = EXCLUDED.start_time
        """, (session_id, 10000.00, '{"test": true}'))
        
        # Insert test order
        order_id = "TEST_ORDER_1"
        cur.execute("""
            INSERT INTO paper_orders (order_id, timestamp, source_id, bet_name, side, size, status)
            VALUES (%s, NOW(), %s, %s, %s, %s, %s)
            ON CONFLICT (order_id) DO UPDATE SET
            timestamp = EXCLUDED.timestamp
        """, (order_id, "test_market_123", "Home", "buy", 100.0, "filled"))
        
        conn.commit()
        
        # Verify data
        cur.execute("SELECT COUNT(*) FROM paper_sessions WHERE session_id = %s", (session_id,))
        session_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM paper_orders WHERE order_id = %s", (order_id,))
        order_count = cur.fetchone()[0]
        
        logger.info(f"✅ Test data inserted successfully:")
        logger.info(f"   • Sessions: {session_count}")
        logger.info(f"   • Orders: {order_count}")
        
        # Test connection to main database
        logger.info("\n📊 Testing connection to main market data...")
        
        from database_v2 import db_manager
        from models import Market
        
        with db_manager.get_db_session() as db:
            market_count = db.query(Market).count()
            logger.info(f"✅ Found {market_count:,} markets in main database")
            
            # Get a sample market
            sample_market = db.query(Market).first()
            if sample_market:
                logger.info(f"   Sample: {sample_market.home_team} vs {sample_market.away_team}")
        
        cur.close()
        conn.close()
        
        logger.info("\n✅ PostgreSQL connection test completed successfully!")
        logger.info("🎯 Paper trading engine is ready to use PostgreSQL!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection test failed: {e}")
        return False

if __name__ == "__main__":
    test_postgres_connection()