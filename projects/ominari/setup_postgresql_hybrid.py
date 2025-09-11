#!/usr/bin/env python3
"""
PostgreSQL Setup for Hybrid Architecture

Sets up PostgreSQL for new blockchain data while SQLite handles historical data.
"""

import os
import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


class PostgreSQLSetup:
    """Setup PostgreSQL for hybrid architecture."""
    
    def __init__(self):
        self.pg_host = os.getenv('PG_HOST', 'localhost')
        self.pg_port = os.getenv('PG_PORT', '5435')  # Updated to match our Docker container
        self.pg_user = os.getenv('PG_USER', 'ominari_user')
        self.pg_password = os.getenv('PG_PASSWORD', 'ominari_2025_secure')
        self.pg_db = os.getenv('PG_DB', 'ominari_production')
    
    def create_database(self):
        """Create database if not exists."""
        try:
            # Connect to postgres database to create new db
            conn = psycopg2.connect(
                host=self.pg_host,
                port=self.pg_port,
                user=self.pg_user,
                password=self.pg_password,
                database='postgres'
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{self.pg_db}'")
            exists = cursor.fetchone()
            
            if not exists:
                logger.info(f"Creating database: {self.pg_db}")
                cursor.execute(f'CREATE DATABASE {self.pg_db}')
            else:
                logger.info(f"Database {self.pg_db} already exists")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise
    
    def setup_schema(self):
        """Set up PostgreSQL schema for blockchain data."""
        conn = psycopg2.connect(
            host=self.pg_host,
            port=self.pg_port,
            user=self.pg_user,
            password=self.pg_password,
            database=self.pg_db
        )
        cursor = conn.cursor()
        
        try:
            # Enable extensions
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS btree_gin")
            
            # Create schemas for data organization
            schemas = ['blockchain', 'api', 'paper_trading', 'analytics']
            for schema in schemas:
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            
            # Blockchain markets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS blockchain.markets (
                    id SERIAL PRIMARY KEY,
                    chain_id INTEGER NOT NULL,
                    chain_name VARCHAR(50) NOT NULL,
                    market_address VARCHAR(66) NOT NULL,
                    market_id VARCHAR(100) UNIQUE NOT NULL,
                    sport VARCHAR(50),
                    league VARCHAR(100),
                    home_team VARCHAR(200),
                    away_team VARCHAR(200),
                    start_time TIMESTAMPTZ,
                    resolved BOOLEAN DEFAULT FALSE,
                    winning_outcome VARCHAR(50),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    block_number BIGINT,
                    tx_hash VARCHAR(66),
                    UNIQUE(chain_id, market_address)
                )
            """)
            
            # Blockchain odds table with partitioning support
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS blockchain.odds (
                    id BIGSERIAL,
                    market_id VARCHAR(100) NOT NULL REFERENCES blockchain.markets(market_id),
                    chain_id INTEGER NOT NULL,
                    outcome VARCHAR(50) NOT NULL,
                    decimal_odds DECIMAL(10,4) NOT NULL,
                    implied_probability DECIMAL(5,4),
                    liquidity DECIMAL(20,6),
                    timestamp TIMESTAMPTZ NOT NULL,
                    block_number BIGINT NOT NULL,
                    tx_hash VARCHAR(66),
                    source VARCHAR(50) DEFAULT 'blockchain',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (id, timestamp)
                ) PARTITION BY RANGE (timestamp)
            """)
            
            # Create monthly partitions for odds
            current_date = datetime.now()
            for i in range(-1, 3):  # Previous month to 2 months ahead
                month = current_date.month + i
                year = current_date.year
                if month > 12:
                    month -= 12
                    year += 1
                elif month < 1:
                    month += 12
                    year -= 1
                
                partition_name = f"odds_{year}_{month:02d}"
                start_date = f"{year}-{month:02d}-01"
                if month == 12:
                    end_date = f"{year+1}-01-01"
                else:
                    end_date = f"{year}-{month+1:02d}-01"
                
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS blockchain.{partition_name}
                    PARTITION OF blockchain.odds
                    FOR VALUES FROM ('{start_date}') TO ('{end_date}')
                """)
            
            # API data tables (for comparison/fallback)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api.markets (
                    id SERIAL PRIMARY KEY,
                    external_id VARCHAR(100) UNIQUE NOT NULL,
                    sport VARCHAR(50),
                    league VARCHAR(100),
                    home_team VARCHAR(200),
                    away_team VARCHAR(200),
                    start_time TIMESTAMPTZ,
                    is_finished BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api.odds (
                    id BIGSERIAL PRIMARY KEY,
                    market_id VARCHAR(100) NOT NULL REFERENCES api.markets(external_id),
                    bookmaker VARCHAR(50) NOT NULL,
                    outcome VARCHAR(50) NOT NULL,
                    decimal_odds DECIMAL(10,4) NOT NULL,
                    american_odds INTEGER,
                    timestamp TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Paper trading tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_trading.sessions (
                    id SERIAL PRIMARY KEY,
                    session_name VARCHAR(100) UNIQUE NOT NULL,
                    strategy VARCHAR(50),
                    initial_balance DECIMAL(20,6) DEFAULT 10000.00,
                    current_balance DECIMAL(20,6),
                    data_source VARCHAR(20) DEFAULT 'blockchain',
                    started_at TIMESTAMPTZ DEFAULT NOW(),
                    ended_at TIMESTAMPTZ,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_trading.positions (
                    id BIGSERIAL PRIMARY KEY,
                    session_id INTEGER REFERENCES paper_trading.sessions(id),
                    market_id VARCHAR(100) NOT NULL,
                    chain_id INTEGER,
                    outcome VARCHAR(50) NOT NULL,
                    stake DECIMAL(20,6) NOT NULL,
                    odds DECIMAL(10,4) NOT NULL,
                    placed_at TIMESTAMPTZ DEFAULT NOW(),
                    settled_at TIMESTAMPTZ,
                    is_won BOOLEAN,
                    pnl DECIMAL(20,6),
                    tx_simulation VARCHAR(66)
                )
            """)
            
            # Analytics views
            cursor.execute("""
                CREATE OR REPLACE VIEW analytics.market_activity AS
                SELECT 
                    m.chain_name,
                    m.sport,
                    COUNT(DISTINCT m.market_id) as total_markets,
                    COUNT(o.id) as total_odds_updates,
                    AVG(o.liquidity) as avg_liquidity,
                    MAX(o.timestamp) as last_update
                FROM blockchain.markets m
                LEFT JOIN blockchain.odds o ON m.market_id = o.market_id
                GROUP BY m.chain_name, m.sport
            """)
            
            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_blockchain_markets_chain ON blockchain.markets(chain_id, chain_name)",
                "CREATE INDEX IF NOT EXISTS idx_blockchain_markets_sport ON blockchain.markets(sport)",
                "CREATE INDEX IF NOT EXISTS idx_blockchain_markets_time ON blockchain.markets(start_time)",
                "CREATE INDEX IF NOT EXISTS idx_blockchain_odds_market ON blockchain.odds(market_id, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_blockchain_odds_chain ON blockchain.odds(chain_id)",
                "CREATE INDEX IF NOT EXISTS idx_api_odds_market ON api.odds(market_id, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_paper_positions_session ON paper_trading.positions(session_id, placed_at)"
            ]
            
            for idx in indexes:
                cursor.execute(idx)
            
            # Create update trigger for timestamps
            cursor.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = NOW();
                    RETURN NEW;
                END;
                $$ language 'plpgsql'
            """)
            
            cursor.execute("""
                CREATE TRIGGER update_blockchain_markets_updated_at
                BEFORE UPDATE ON blockchain.markets
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column()
            """)
            
            conn.commit()
            logger.info("✅ PostgreSQL schema created successfully")
            
            # Report table stats
            cursor.execute("""
                SELECT schemaname, tablename 
                FROM pg_tables 
                WHERE schemaname IN ('blockchain', 'api', 'paper_trading')
                ORDER BY schemaname, tablename
            """)
            
            tables = cursor.fetchall()
            logger.info("\n📊 Created tables:")
            for schema, table in tables:
                logger.info(f"   {schema}.{table}")
            
        except Exception as e:
            logger.error(f"Error setting up schema: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    
    def create_hybrid_functions(self):
        """Create functions for hybrid data access."""
        conn = psycopg2.connect(
            host=self.pg_host,
            port=self.pg_port,
            user=self.pg_user,
            password=self.pg_password,
            database=self.pg_db
        )
        cursor = conn.cursor()
        
        try:
            # Function to get latest odds (checks both sources)
            cursor.execute("""
                CREATE OR REPLACE FUNCTION get_latest_odds(
                    p_market_id VARCHAR,
                    p_source VARCHAR DEFAULT 'any'
                )
                RETURNS TABLE(
                    outcome VARCHAR,
                    odds DECIMAL,
                    source VARCHAR,
                    ts TIMESTAMPTZ
                ) AS $$
                BEGIN
                    IF p_source = 'blockchain' OR p_source = 'any' THEN
                        RETURN QUERY
                        SELECT DISTINCT ON (o.outcome)
                            o.outcome,
                            o.decimal_odds,
                            'blockchain'::VARCHAR as source,
                            o.timestamp as ts
                        FROM blockchain.odds o
                        WHERE o.market_id = p_market_id
                        ORDER BY o.outcome, o.timestamp DESC;
                    END IF;
                    
                    IF p_source = 'api' OR (p_source = 'any' AND NOT FOUND) THEN
                        RETURN QUERY
                        SELECT DISTINCT ON (o.outcome)
                            o.outcome,
                            o.decimal_odds,
                            'api'::VARCHAR as source,
                            o.timestamp
                        FROM api.odds o
                        WHERE o.market_id = p_market_id
                        ORDER BY o.outcome, o.timestamp DESC;
                    END IF;
                END;
                $$ LANGUAGE plpgsql
            """)
            
            # Function to get market summary
            cursor.execute("""
                CREATE OR REPLACE FUNCTION get_market_summary(
                    p_chain_id INTEGER DEFAULT NULL,
                    p_sport VARCHAR DEFAULT NULL
                )
                RETURNS TABLE(
                    chain_name VARCHAR,
                    sport VARCHAR,
                    total_markets BIGINT,
                    active_markets BIGINT,
                    total_volume DECIMAL
                ) AS $$
                BEGIN
                    RETURN QUERY
                    SELECT 
                        m.chain_name,
                        m.sport,
                        COUNT(DISTINCT m.market_id),
                        COUNT(DISTINCT CASE WHEN m.resolved = FALSE THEN m.market_id END),
                        COALESCE(SUM(o.liquidity), 0)
                    FROM blockchain.markets m
                    LEFT JOIN blockchain.odds o ON m.market_id = o.market_id
                    WHERE (p_chain_id IS NULL OR m.chain_id = p_chain_id)
                        AND (p_sport IS NULL OR m.sport = p_sport)
                    GROUP BY m.chain_name, m.sport;
                END;
                $$ LANGUAGE plpgsql
            """)
            
            conn.commit()
            logger.info("✅ Hybrid access functions created")
            
        except Exception as e:
            logger.error(f"Error creating functions: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    
    def setup_replication_user(self):
        """Create user for SQLite->PostgreSQL replication."""
        conn = psycopg2.connect(
            host=self.pg_host,
            port=self.pg_port,
            user=self.pg_user,
            password=self.pg_password,
            database=self.pg_db
        )
        cursor = conn.cursor()
        
        try:
            # Create replication user
            cursor.execute("SELECT 1 FROM pg_user WHERE usename = 'ominari_replicator'")
            if not cursor.fetchone():
                cursor.execute("""
                    CREATE USER ominari_replicator WITH PASSWORD 'repl_2025_secure'
                """)
            
            # Grant permissions
            cursor.execute("""
                GRANT CONNECT ON DATABASE {} TO ominari_replicator
            """.format(self.pg_db))
            
            cursor.execute("""
                GRANT USAGE ON SCHEMA blockchain, api TO ominari_replicator
            """)
            
            cursor.execute("""
                GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA blockchain TO ominari_replicator
            """)
            
            cursor.execute("""
                GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA api TO ominari_replicator
            """)
            
            conn.commit()
            logger.info("✅ Replication user configured")
            
        except Exception as e:
            logger.error(f"Error setting up replication user: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def run(self):
        """Run complete PostgreSQL setup."""
        logger.info("🚀 Starting PostgreSQL setup for hybrid architecture")
        
        try:
            self.create_database()
            self.setup_schema()
            self.create_hybrid_functions()
            self.setup_replication_user()
            
            logger.info("\n✅ PostgreSQL setup completed successfully!")
            logger.info("📝 Next steps:")
            logger.info("   1. Configure blockchain readers to write to PostgreSQL")
            logger.info("   2. Update unified data system to query both databases")
            logger.info("   3. Start blockchain synchronization")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise


if __name__ == "__main__":
    setup = PostgreSQLSetup()
    setup.run()