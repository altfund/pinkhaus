#!/usr/bin/env python3
"""
Start PostgreSQL via flox and deploy Ominari with real blockchain data
"""

import os
import subprocess
import time
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use port 5435 to avoid conflict
POSTGRES_PORT = "5435"

def setup_flox_postgres():
    """Set up PostgreSQL in flox environment on port 5435."""
    logger.info("🐘 Setting up PostgreSQL via flox on port 5435...")
    
    # Set up directories
    pgdata = Path(".flox/postgres/data")
    pghost = Path(".flox/postgres/socket")
    pglog = Path(".flox/postgres/log")
    
    pgdata.mkdir(parents=True, exist_ok=True)
    pghost.mkdir(parents=True, exist_ok=True)
    pglog.mkdir(parents=True, exist_ok=True)
    
    # Update PostgreSQL config to use port 5435
    pg_conf = pgdata / "postgresql.conf"
    if pg_conf.exists():
        logger.info("Updating PostgreSQL to use port 5435...")
        config = pg_conf.read_text()
        config = config.replace("port = 5432", f"port = {POSTGRES_PORT}")
        pg_conf.write_text(config)
    
    # Set environment
    env = os.environ.copy()
    env.update({
        'PGDATA': str(pgdata),
        'PGHOST': str(pghost),
        'PGPORT': POSTGRES_PORT,
        'PATH': f"{os.environ.get('PATH')}:{Path.home()}/.flox/run/x86_64-linux.ominari.dev/bin"
    })
    
    # Stop any existing PostgreSQL on this port
    subprocess.run(['pg_ctl', '-D', str(pgdata), 'stop'], env=env, capture_output=True)
    time.sleep(2)
    
    # Start PostgreSQL in flox environment
    logger.info("Starting PostgreSQL in flox environment...")
    cmd = ['flox', 'activate', '--', 'pg_ctl', '-D', str(pgdata), '-l', str(pglog / 'postgres.log'), 'start']
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("✅ PostgreSQL started successfully")
        time.sleep(3)
        
        # Create database and user
        logger.info("Creating Ominari database...")
        
        # Run in flox environment
        create_cmds = [
            f"createuser -p {POSTGRES_PORT} -h {pghost} -U postgres ominari_user 2>/dev/null || true",
            f"psql -p {POSTGRES_PORT} -h {pghost} -U postgres -c \"ALTER USER ominari_user WITH PASSWORD 'ominari_2025_secure';\"",
            f"createdb -p {POSTGRES_PORT} -h {pghost} -U postgres -O ominari_user ominari_production 2>/dev/null || true"
        ]
        
        for cmd in create_cmds:
            subprocess.run(['flox', 'activate', '--', 'bash', '-c', cmd], env=env, capture_output=True)
        
        logger.info("✅ Database configured")
        return True
    else:
        logger.error(f"Failed to start PostgreSQL: {result.stderr}")
        return False

def migrate_blockchain_data():
    """Migrate blockchain data to PostgreSQL."""
    logger.info("📊 Migrating blockchain data to PostgreSQL...")
    
    # Create migration script
    migration_script = '''
import os
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5435',
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})

from sqlalchemy import create_engine, text
from models import Base, Market, Odd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PostgreSQL connection
pg_url = "postgresql://ominari_user:ominari_2025_secure@localhost:5435/ominari_production"
pg_engine = create_engine(pg_url)

# Create schema
logger.info("Creating PostgreSQL schema...")
Base.metadata.create_all(pg_engine)

# Copy blockchain data from SQLite
sqlite_engine = create_engine('sqlite:///sport_odds.db')

with sqlite_engine.connect() as src:
    # Get blockchain markets
    markets = src.execute(text("""
        SELECT source_id, source, sport, league_name, home_team, away_team, 
               market_type, maturity_date, is_finished, updated_at
        FROM market 
        WHERE source LIKE 'blockchain_%'
    """)).fetchall()
    
    logger.info(f"Migrating {len(markets)} blockchain markets...")
    
    # Insert into PostgreSQL
    with pg_engine.connect() as dst:
        for market in markets:
            try:
                dst.execute(text("""
                    INSERT INTO market (source_id, source, sport, league_name, 
                                      home_team, away_team, market_type, 
                                      maturity_date, is_finished, updated_at)
                    VALUES (:source_id, :source, :sport, :league_name,
                            :home_team, :away_team, :market_type,
                            :maturity_date, :is_finished, :updated_at)
                    ON CONFLICT (source_id) DO NOTHING
                """), dict(market._mapping))
            except Exception as e:
                logger.debug(f"Skipping market: {e}")
        dst.commit()
    
    # Migrate odds
    odds = src.execute(text("""
        SELECT o.source_id, o.position, o.outcome, o.decimal_odds,
               o.market_type, o.source, o.bookmaker, o.updated_at
        FROM odd o
        JOIN market m ON o.source_id = m.source_id
        WHERE m.source LIKE 'blockchain_%'
    """)).fetchall()
    
    logger.info(f"Migrating {len(odds)} blockchain odds...")
    
    with pg_engine.connect() as dst:
        for odd in odds:
            try:
                dst.execute(text("""
                    INSERT INTO odd (source_id, position, outcome, decimal_odds,
                                   market_type, source, bookmaker, updated_at)
                    VALUES (:source_id, :position, :outcome, :decimal_odds,
                            :market_type, :source, :bookmaker, :updated_at)
                    ON CONFLICT DO NOTHING
                """), dict(odd._mapping))
            except Exception as e:
                logger.debug(f"Skipping odd: {e}")
        dst.commit()

logger.info("✅ Migration complete!")
'''
    
    with open('temp_migrate.py', 'w') as f:
        f.write(migration_script)
    
    # Run migration
    result = subprocess.run(['uv', 'run', 'python', 'temp_migrate.py'], capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("✅ Data migration successful")
        os.remove('temp_migrate.py')
        return True
    else:
        logger.error(f"Migration failed: {result.stderr}")
        return False

def start_web_monitor():
    """Start the web monitor with PostgreSQL."""
    logger.info("🚀 Starting Web Monitor with PostgreSQL...")
    
    # Update environment for port 5435
    env_vars = {
        'PG_HOST': 'localhost',
        'PG_PORT': POSTGRES_PORT,
        'PG_USER': 'ominari_user',
        'PG_PASSWORD': 'ominari_2025_secure',
        'PG_DB': 'ominari_production',
        'USE_POSTGRESQL': '1'
    }
    
    os.environ.update(env_vars)
    
    # Save environment
    with open('.env.postgres', 'w') as f:
        for k, v in env_vars.items():
            f.write(f"export {k}={v}\n")
    
    logger.info("Starting web monitor on http://localhost:8888")
    logger.info("Dashboard: http://localhost:8888/unified")
    
    # Run web monitor
    subprocess.run(['uv', 'run', 'python', 'web_monitor.py'])

def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("🚀 Ominari with PostgreSQL via Flox")
    logger.info("=" * 60)
    
    # Set up PostgreSQL
    if not setup_flox_postgres():
        logger.error("Failed to set up PostgreSQL")
        sys.exit(1)
    
    # Migrate data
    if not migrate_blockchain_data():
        logger.warning("Data migration failed, but continuing...")
    
    # Start web monitor
    start_web_monitor()

if __name__ == "__main__":
    main()