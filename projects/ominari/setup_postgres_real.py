#!/usr/bin/env python3
"""
Set up real PostgreSQL database for Ominari
Creates a local PostgreSQL instance and migrates blockchain data.
"""

import subprocess
import os
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path.cwd()
POSTGRES_DIR = PROJECT_DIR / ".postgres"
POSTGRES_DATA = POSTGRES_DIR / "data"
POSTGRES_LOG = POSTGRES_DIR / "postgres.log"
POSTGRES_SOCKET = POSTGRES_DIR / "socket"

# Configuration
PG_CONFIG = {
    'port': '5432',
    'user': 'ominari_user',
    'password': 'ominari_2025_secure',
    'database': 'ominari_production'
}

def run_cmd(cmd, check=True, capture_output=True):
    """Run a shell command."""
    if isinstance(cmd, str):
        cmd = cmd.split()
    result = subprocess.run(cmd, capture_output=capture_output, text=True, check=False)
    if check and result.returncode != 0:
        logger.error(f"Command failed: {' '.join(cmd)}")
        logger.error(f"stderr: {result.stderr}")
        raise Exception(f"Command failed: {result.stderr}")
    return result

def setup_local_postgres():
    """Set up a local PostgreSQL instance."""
    logger.info("🐘 Setting up Local PostgreSQL for Ominari")
    
    # Create directories
    logger.info("Creating PostgreSQL directories...")
    POSTGRES_DIR.mkdir(exist_ok=True)
    POSTGRES_DATA.mkdir(exist_ok=True)
    POSTGRES_SOCKET.mkdir(exist_ok=True)
    
    # Check if PostgreSQL is installed
    result = run_cmd(['which', 'initdb'], check=False)
    if result.returncode != 0:
        logger.error("PostgreSQL is not installed. Installing via apt...")
        run_cmd(['sudo', 'apt-get', 'update', '-y'])
        run_cmd(['sudo', 'apt-get', 'install', '-y', 'postgresql', 'postgresql-client'])
    
    # Initialize database if needed
    if not (POSTGRES_DATA / 'PG_VERSION').exists():
        logger.info("Initializing PostgreSQL database...")
        run_cmd([
            'initdb', '-D', str(POSTGRES_DATA),
            '-U', 'postgres',
            '--locale=C',
            '--encoding=UTF8'
        ])
        
        # Configure PostgreSQL
        pg_conf = POSTGRES_DATA / 'postgresql.conf'
        with open(pg_conf, 'a') as f:
            f.write(f"\n# Ominari Configuration\n")
            f.write(f"port = {PG_CONFIG['port']}\n")
            f.write(f"listen_addresses = 'localhost'\n")
            f.write(f"unix_socket_directories = '{POSTGRES_SOCKET}'\n")
            f.write(f"shared_buffers = 256MB\n")
            f.write(f"max_connections = 100\n")
        
        # Configure authentication
        pg_hba = POSTGRES_DATA / 'pg_hba.conf'
        with open(pg_hba, 'w') as f:
            f.write("# TYPE  DATABASE        USER            ADDRESS                 METHOD\n")
            f.write("local   all             postgres                                trust\n")
            f.write("local   all             all                                     md5\n")
            f.write("host    all             all             127.0.0.1/32            md5\n")
            f.write("host    all             all             ::1/128                 md5\n")
    
    # Start PostgreSQL
    logger.info("Starting PostgreSQL...")
    pg_ctl = run_cmd(['which', 'pg_ctl']).stdout.strip()
    
    # Check if already running
    status = run_cmd([pg_ctl, '-D', str(POSTGRES_DATA), 'status'], check=False)
    if status.returncode != 0:
        run_cmd([
            pg_ctl, '-D', str(POSTGRES_DATA),
            '-l', str(POSTGRES_LOG),
            '-o', f'-k {POSTGRES_SOCKET}',
            'start'
        ])
        time.sleep(3)
    
    # Create user and database
    logger.info("Creating Ominari database and user...")
    
    # Use Unix socket for initial setup
    os.environ['PGHOST'] = str(POSTGRES_SOCKET)
    os.environ['PGPORT'] = PG_CONFIG['port']
    
    # Create user
    run_cmd([
        'psql', '-U', 'postgres', '-c',
        f"CREATE USER {PG_CONFIG['user']} WITH PASSWORD '{PG_CONFIG['password']}';"
    ], check=False)
    
    # Create database
    run_cmd([
        'psql', '-U', 'postgres', '-c',
        f"CREATE DATABASE {PG_CONFIG['database']} OWNER {PG_CONFIG['user']};"
    ], check=False)
    
    # Grant privileges
    run_cmd([
        'psql', '-U', 'postgres', '-c',
        f"GRANT ALL PRIVILEGES ON DATABASE {PG_CONFIG['database']} TO {PG_CONFIG['user']};"
    ])
    
    logger.info("✅ PostgreSQL is ready!")
    
    # Create environment file
    env_content = f"""# PostgreSQL Configuration for Ominari
export PG_HOST=localhost
export PG_PORT={PG_CONFIG['port']}
export PG_USER={PG_CONFIG['user']}
export PG_PASSWORD={PG_CONFIG['password']}
export PG_DB={PG_CONFIG['database']}
export PGHOST={POSTGRES_SOCKET}
export PGPORT={PG_CONFIG['port']}
"""
    
    with open('.env.postgres', 'w') as f:
        f.write(env_content)
    
    logger.info("✅ Environment saved to .env.postgres")
    
    return True

def migrate_blockchain_data():
    """Migrate blockchain data from SQLite to PostgreSQL."""
    logger.info("\n📊 Migrating blockchain data...")
    
    # Source environment
    os.environ.update({
        'PG_HOST': 'localhost',
        'PG_PORT': PG_CONFIG['port'],
        'PG_USER': PG_CONFIG['user'],
        'PG_PASSWORD': PG_CONFIG['password'],
        'PG_DB': PG_CONFIG['database'],
        'PGHOST': str(POSTGRES_SOCKET),
        'PGPORT': PG_CONFIG['port']
    })
    
    # Run migration script
    try:
        from blockchain_postgres_writer import migrate_blockchain_to_postgres
        migrate_blockchain_to_postgres()
        logger.info("✅ Blockchain data migrated!")
    except ImportError:
        logger.warning("Migration script not found, creating one...")
        create_migration_script()
        run_cmd(['uv', 'run', 'python', 'migrate_blockchain_to_postgres.py'])

def create_migration_script():
    """Create a migration script to move data to PostgreSQL."""
    script = '''#!/usr/bin/env python3
"""Migrate blockchain data to PostgreSQL"""

import os
os.environ['FORCE_POSTGRESQL'] = '1'

from sqlalchemy import create_engine, text
from models import Base, Market, Odd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PostgreSQL connection
pg_url = f"postgresql://{os.environ['PG_USER']}:{os.environ['PG_PASSWORD']}@localhost:{os.environ['PG_PORT']}/{os.environ['PG_DB']}"
pg_engine = create_engine(pg_url)

# SQLite connection
sqlite_engine = create_engine('sqlite:///sport_odds.db')

logger.info("Creating PostgreSQL schema...")
Base.metadata.create_all(pg_engine)

logger.info("Migrating blockchain markets...")
# Get blockchain markets from SQLite
with sqlite_engine.connect() as src:
    markets = src.execute(text("""
        SELECT * FROM market 
        WHERE source LIKE 'blockchain_%'
        LIMIT 1000
    """)).fetchall()
    
    logger.info(f"Found {len(markets)} blockchain markets")
    
    # Insert into PostgreSQL
    with pg_engine.connect() as dst:
        for market in markets:
            dst.execute(text("""
                INSERT INTO market (source_id, source, sport, league_name, 
                                  home_team, away_team, market_type, 
                                  maturity_date, is_finished, updated_at)
                VALUES (:source_id, :source, :sport, :league_name,
                        :home_team, :away_team, :market_type,
                        :maturity_date, :is_finished, :updated_at)
                ON CONFLICT (source_id) DO NOTHING
            """), dict(market._mapping))
        dst.commit()
    
    # Migrate odds
    with sqlite_engine.connect() as src:
        odds = src.execute(text("""
            SELECT o.* FROM odd o
            JOIN market m ON o.source_id = m.source_id
            WHERE m.source LIKE 'blockchain_%'
            LIMIT 5000
        """)).fetchall()
        
        logger.info(f"Found {len(odds)} blockchain odds")
        
        with pg_engine.connect() as dst:
            for odd in odds:
                dst.execute(text("""
                    INSERT INTO odd (source_id, position, outcome, decimal_odds,
                                   market_type, source, bookmaker, updated_at)
                    VALUES (:source_id, :position, :outcome, :decimal_odds,
                            :market_type, :source, :bookmaker, :updated_at)
                    ON CONFLICT DO NOTHING
                """), dict(odd._mapping))
            dst.commit()

logger.info("✅ Migration complete!")
'''
    
    with open('migrate_blockchain_to_postgres.py', 'w') as f:
        f.write(script)

def main():
    """Main setup function."""
    logger.info("=" * 60)
    logger.info("PostgreSQL Setup for Real Blockchain Data")
    logger.info("=" * 60)
    
    try:
        # Set up PostgreSQL
        if setup_local_postgres():
            # Test connection
            logger.info("\nTesting PostgreSQL connection...")
            os.environ['PGPASSWORD'] = PG_CONFIG['password']
            result = run_cmd([
                'psql', '-h', 'localhost', '-p', PG_CONFIG['port'],
                '-U', PG_CONFIG['user'], '-d', PG_CONFIG['database'],
                '-c', 'SELECT version();'
            ], check=False)
            
            if result.returncode == 0:
                logger.info("✅ PostgreSQL connection successful!")
                
                # Migrate data
                migrate_blockchain_data()
                
                logger.info("\n✨ Setup Complete!")
                logger.info("\nNext steps:")
                logger.info("1. Source environment: source .env.postgres")
                logger.info("2. Start web monitor: uv run python web_monitor.py")
                logger.info("3. Access dashboard: http://localhost:8888/unified")
                logger.info(f"\nPostgreSQL is running on port {PG_CONFIG['port']}")
                logger.info(f"Data directory: {POSTGRES_DATA}")
                logger.info(f"Log file: {POSTGRES_LOG}")
            else:
                logger.error("PostgreSQL connection failed!")
                logger.error(result.stderr)
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()