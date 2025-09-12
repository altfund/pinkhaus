#!/usr/bin/env python3
"""
PostgreSQL Setup and Migration Script for Ominari Trading System

This script:
1. Creates PostgreSQL database and user
2. Sets up proper schema with optimized settings
3. Migrates data from SQLite to PostgreSQL
"""

import os
import sys
import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import subprocess
from sqlalchemy import create_engine, text
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PostgreSQLSetup:
    def __init__(self):
        self.db_name = 'ominari_trading'
        self.db_user = 'ominari'
        self.db_password = os.getenv('POSTGRES_PASSWORD', 'ominari_secure_pass_2025')
        self.db_host = os.getenv('POSTGRES_HOST', 'localhost')
        self.db_port = os.getenv('POSTGRES_PORT', '5432')
        
    def check_postgres_installed(self):
        """Check if PostgreSQL is installed."""
        try:
            result = subprocess.run(['psql', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"PostgreSQL found: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        return False
        
    def install_postgres(self):
        """Provide installation instructions for PostgreSQL."""
        logger.info("PostgreSQL not found. Please install it:")
        
        system = subprocess.run(['uname', '-s'], capture_output=True, text=True).stdout.strip()
        
        if system == 'Darwin':  # macOS
            print("\nFor macOS:")
            print("brew install postgresql")
            print("brew services start postgresql")
        elif system == 'Linux':
            print("\nFor Ubuntu/Debian:")
            print("sudo apt update")
            print("sudo apt install postgresql postgresql-contrib")
            print("sudo systemctl start postgresql")
            print("\nFor RHEL/CentOS:")
            print("sudo yum install postgresql-server postgresql-contrib")
            print("sudo postgresql-setup initdb")
            print("sudo systemctl start postgresql")
        else:
            print("\nPlease install PostgreSQL for your system from:")
            print("https://www.postgresql.org/download/")
            
    def create_database(self):
        """Create database and user."""
        logger.info("Creating PostgreSQL database and user...")
        
        try:
            # Connect as superuser
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                user='postgres',
                password=os.getenv('POSTGRES_SUPERUSER_PASSWORD', '')
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Create user
            cursor.execute(f"""
                SELECT 1 FROM pg_user WHERE usename = '{self.db_user}'
            """)
            if not cursor.fetchone():
                cursor.execute(f"""
                    CREATE USER {self.db_user} WITH PASSWORD '{self.db_password}'
                """)
                logger.info(f"Created user: {self.db_user}")
            else:
                logger.info(f"User {self.db_user} already exists")
                
            # Create database
            cursor.execute(f"""
                SELECT 1 FROM pg_database WHERE datname = '{self.db_name}'
            """)
            if not cursor.fetchone():
                cursor.execute(f"""
                    CREATE DATABASE {self.db_name} OWNER {self.db_user}
                    ENCODING 'UTF8'
                    LC_COLLATE = 'en_US.UTF-8'
                    LC_CTYPE = 'en_US.UTF-8'
                    TEMPLATE template0
                """)
                logger.info(f"Created database: {self.db_name}")
            else:
                logger.info(f"Database {self.db_name} already exists")
                
            # Grant privileges
            cursor.execute(f"""
                GRANT ALL PRIVILEGES ON DATABASE {self.db_name} TO {self.db_user}
            """)
            
            cursor.close()
            conn.close()
            
            return True
            
        except psycopg2.OperationalError as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            logger.info("\nPlease ensure PostgreSQL is running and you have superuser access")
            logger.info("You may need to:")
            logger.info("1. Start PostgreSQL service")
            logger.info("2. Set POSTGRES_SUPERUSER_PASSWORD environment variable")
            logger.info("3. Update pg_hba.conf to allow connections")
            return False
            
    def create_schema(self):
        """Create optimized schema for trading system."""
        logger.info("Creating optimized schema...")
        
        # Connect to new database
        engine = create_engine(
            f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        )
        
        with engine.connect() as conn:
            # Enable extensions
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_stat_statements"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS btree_gin"))
            conn.commit()
            
            # Create optimized tables
            
            # Markets table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS markets (
                    id SERIAL PRIMARY KEY,
                    source VARCHAR(50) NOT NULL,
                    source_id VARCHAR(255) NOT NULL,
                    sport VARCHAR(50) NOT NULL,
                    league_name VARCHAR(100),
                    home_team VARCHAR(200) NOT NULL,
                    away_team VARCHAR(200) NOT NULL,
                    market_type VARCHAR(50) NOT NULL,
                    maturity_date TIMESTAMPTZ NOT NULL,
                    is_finished BOOLEAN DEFAULT FALSE,
                    home_score INTEGER,
                    away_score INTEGER,
                    last_update TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(source, source_id, market_type)
                )
            """))
            
            # Odds table (normalized)
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS odds_normalized (
                    market_id VARCHAR(68) NOT NULL,
                    bookmaker_id SMALLINT NOT NULL,
                    outcome_id SMALLINT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    source_id SMALLINT NOT NULL,
                    market_type_id SMALLINT NOT NULL,
                    position SMALLINT NOT NULL,
                    line_x100 SMALLINT,
                    decimal_odds_x1000 SMALLINT NOT NULL,
                    american_odds SMALLINT,
                    implied_x10000 SMALLINT,
                    PRIMARY KEY (market_id, bookmaker_id, outcome_id, updated_at)
                ) PARTITION BY RANGE (updated_at)
            """))
            
            # Create partitions for last 90 days
            current_ts = int(time.time())
            day_seconds = 86400
            
            for i in range(90):
                start_ts = current_ts - ((i + 1) * day_seconds)
                end_ts = current_ts - (i * day_seconds)
                partition_name = f"odds_normalized_{time.strftime('%Y%m%d', time.localtime(end_ts))}"
                
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {partition_name}
                    PARTITION OF odds_normalized
                    FOR VALUES FROM ({start_ts}) TO ({end_ts})
                """))
                
            # Positions table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS positions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(100) NOT NULL,
                    market_source VARCHAR(50) NOT NULL,
                    market_source_id VARCHAR(255) NOT NULL,
                    bookmaker VARCHAR(50) NOT NULL,
                    market_type VARCHAR(50) NOT NULL,
                    outcome VARCHAR(100) NOT NULL,
                    odds DECIMAL(10, 4) NOT NULL,
                    stake DECIMAL(10, 2) NOT NULL,
                    potential_payout DECIMAL(10, 2) NOT NULL,
                    placed_at TIMESTAMPTZ NOT NULL,
                    settled_at TIMESTAMPTZ,
                    result VARCHAR(20),
                    pnl DECIMAL(10, 2),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            
            # Create indexes
            logger.info("Creating indexes...")
            
            indexes = [
                "CREATE INDEX idx_markets_maturity ON markets(maturity_date) WHERE NOT is_finished",
                "CREATE INDEX idx_markets_source ON markets(source, source_id)",
                "CREATE INDEX idx_markets_sport_league ON markets(sport, league_name)",
                "CREATE INDEX idx_odds_market_updated ON odds_normalized(market_id, updated_at DESC)",
                "CREATE INDEX idx_odds_bookmaker ON odds_normalized(bookmaker_id, updated_at DESC)",
                "CREATE INDEX idx_positions_session ON positions(session_id, placed_at DESC)",
                "CREATE INDEX idx_positions_result ON positions(result) WHERE result IS NOT NULL",
            ]
            
            for idx in indexes:
                try:
                    conn.execute(text(idx))
                except Exception as e:
                    logger.warning(f"Index may already exist: {e}")
                    
            conn.commit()
            
        logger.info("Schema created successfully")
        
    def optimize_settings(self):
        """Apply PostgreSQL optimizations for large datasets."""
        logger.info("Applying PostgreSQL optimizations...")
        
        engine = create_engine(
            f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        )
        
        with engine.connect() as conn:
            # Get total RAM (simplified - assumes 8GB available for PostgreSQL)
            ram_gb = 8
            
            optimizations = [
                f"ALTER SYSTEM SET shared_buffers = '{ram_gb // 4}GB'",
                f"ALTER SYSTEM SET effective_cache_size = '{ram_gb * 3 // 4}GB'",
                "ALTER SYSTEM SET maintenance_work_mem = '1GB'",
                "ALTER SYSTEM SET checkpoint_completion_target = 0.9",
                "ALTER SYSTEM SET wal_buffers = '16MB'",
                "ALTER SYSTEM SET default_statistics_target = 100",
                "ALTER SYSTEM SET random_page_cost = 1.1",  # For SSD
                "ALTER SYSTEM SET effective_io_concurrency = 200",  # For SSD
                "ALTER SYSTEM SET work_mem = '128MB'",
                "ALTER SYSTEM SET max_wal_size = '4GB'",
                "ALTER SYSTEM SET min_wal_size = '1GB'",
                "ALTER SYSTEM SET max_worker_processes = 8",
                "ALTER SYSTEM SET max_parallel_workers_per_gather = 4",
                "ALTER SYSTEM SET max_parallel_workers = 8",
            ]
            
            for setting in optimizations:
                try:
                    conn.execute(text(setting))
                    logger.info(f"Applied: {setting}")
                except Exception as e:
                    logger.warning(f"Could not apply {setting}: {e}")
                    
            conn.commit()
            
        logger.info("Optimizations applied. Restart PostgreSQL for changes to take effect.")
        
    def create_migration_script(self):
        """Create script to migrate from SQLite to PostgreSQL."""
        migration_script = '''#!/bin/bash
# SQLite to PostgreSQL migration script

echo "Starting migration from SQLite to PostgreSQL..."

# Export from SQLite
echo "Exporting markets..."
sqlite3 sport_odds.db <<EOF
.mode csv
.headers on
.output markets_export.csv
SELECT * FROM market LIMIT 1000000;
EOF

echo "Exporting recent odds..."
sqlite3 sport_odds.db <<EOF
.mode csv
.headers on
.output odds_export.csv
SELECT * FROM odd 
WHERE updated_at > datetime('now', '-30 days')
LIMIT 10000000;
EOF

# Import to PostgreSQL
echo "Importing to PostgreSQL..."
psql -U {db_user} -d {db_name} <<EOF
\\copy markets FROM 'markets_export.csv' WITH CSV HEADER;
\\copy odds_normalized FROM 'odds_export.csv' WITH CSV HEADER;
EOF

echo "Migration complete!"
        '''.format(db_user=self.db_user, db_name=self.db_name)
        
        with open('migrate_to_postgres.sh', 'w') as f:
            f.write(migration_script)
        os.chmod('migrate_to_postgres.sh', 0o755)
        
        logger.info("Created migration script: migrate_to_postgres.sh")
        
    def generate_connection_string(self):
        """Generate SQLAlchemy connection string."""
        conn_str = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        
        logger.info("\n" + "="*60)
        logger.info("PostgreSQL setup complete!")
        logger.info("="*60)
        logger.info("\nAdd this to your .env file:")
        logger.info(f"DATABASE_URL={conn_str}")
        logger.info("\nOr set as environment variable:")
        logger.info(f"export DATABASE_URL='{conn_str}'")
        logger.info("\nTo migrate data:")
        logger.info("./migrate_to_postgres.sh")
        logger.info("="*60)
        
        return conn_str


def main():
    """Main setup process."""
    setup = PostgreSQLSetup()
    
    # Check if PostgreSQL is installed
    if not setup.check_postgres_installed():
        setup.install_postgres()
        sys.exit(1)
        
    # Create database and user
    if not setup.create_database():
        sys.exit(1)
        
    # Create schema
    setup.create_schema()
    
    # Apply optimizations
    setup.optimize_settings()
    
    # Create migration script
    setup.create_migration_script()
    
    # Show connection string
    setup.generate_connection_string()


if __name__ == "__main__":
    main()