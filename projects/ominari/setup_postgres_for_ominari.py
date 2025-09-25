#!/usr/bin/env python3
"""
Setup PostgreSQL for Ominari Trading System
Sets up database, user, and permissions for Ominari.
"""

import subprocess
import logging
import sys
import os
import psycopg2
from psycopg2 import sql

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PG_CONFIG = {
    'host': 'localhost',
    'port': '5432',  # Use default PostgreSQL port
    'user': 'ominari_user',
    'password': 'ominari_2025_secure',
    'database': 'ominari_production'
}

def run_command(cmd, check=True):
    """Run a shell command."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.stderr}")
        return None

def setup_postgresql():
    """Set up PostgreSQL database and user for Ominari."""
    logger.info("🐘 Setting up PostgreSQL for Ominari Trading System")
    
    # Check if PostgreSQL is running
    pg_status = run_command("systemctl is-active postgresql", check=False)
    if pg_status != "active":
        logger.info("PostgreSQL is not running. Starting it...")
        run_command("sudo systemctl start postgresql")
    
    # Connect as postgres superuser to create database and user
    logger.info("Creating Ominari database and user...")
    
    try:
        # First, create the database and user using psql
        create_user_cmd = f"""
        sudo -u postgres psql -c "CREATE USER {PG_CONFIG['user']} WITH PASSWORD '{PG_CONFIG['password']}';" 2>/dev/null || true
        """
        run_command(create_user_cmd)
        
        create_db_cmd = f"""
        sudo -u postgres psql -c "CREATE DATABASE {PG_CONFIG['database']} OWNER {PG_CONFIG['user']};" 2>/dev/null || true
        """
        run_command(create_db_cmd)
        
        # Grant all privileges
        grant_cmd = f"""
        sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE {PG_CONFIG['database']} TO {PG_CONFIG['user']};"
        """
        run_command(grant_cmd)
        
        logger.info("✅ Database and user created successfully")
        
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        return False
    
    # Test connection
    try:
        conn = psycopg2.connect(
            host=PG_CONFIG['host'],
            port=PG_CONFIG['port'],
            database=PG_CONFIG['database'],
            user=PG_CONFIG['user'],
            password=PG_CONFIG['password']
        )
        conn.close()
        logger.info("✅ Successfully connected to PostgreSQL as ominari_user")
        
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        return False
    
    # Update environment variables
    logger.info("Setting up environment variables...")
    
    env_vars = f"""
# PostgreSQL Configuration for Ominari
export PG_HOST=localhost
export PG_PORT=5432
export PG_USER={PG_CONFIG['user']}
export PG_PASSWORD={PG_CONFIG['password']}
export PG_DB={PG_CONFIG['database']}
"""
    
    # Write to a file that can be sourced
    with open('.env.postgres', 'w') as f:
        f.write(env_vars)
    
    logger.info("✅ Environment variables saved to .env.postgres")
    logger.info("   Source it with: source .env.postgres")
    
    # Create a modified database_v2.py that uses port 5432
    logger.info("Creating database configuration for port 5432...")
    create_db_config()
    
    return True

def create_db_config():
    """Create a database configuration that uses the correct port."""
    config_content = '''#!/usr/bin/env python3
"""
PostgreSQL Database Configuration for Ominari
Uses standard PostgreSQL port 5432.
"""

import os

# PostgreSQL Configuration
PG_CONFIG = {
    'host': os.getenv('PG_HOST', 'localhost'),
    'port': os.getenv('PG_PORT', '5432'),  # Standard PostgreSQL port
    'user': os.getenv('PG_USER', 'ominari_user'),
    'password': os.getenv('PG_PASSWORD', 'ominari_2025_secure'),
    'database': os.getenv('PG_DB', 'ominari_production')
}

DB_URL = f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['database']}"

# Export the connection string
POSTGRESQL_URL = DB_URL
'''
    
    with open('db_config_postgres.py', 'w') as f:
        f.write(config_content)
    
    logger.info("✅ Database configuration created in db_config_postgres.py")

def main():
    """Main setup function."""
    logger.info("=" * 60)
    logger.info("PostgreSQL Setup for Ominari Trading System")
    logger.info("=" * 60)
    
    # Check if running with appropriate permissions
    if os.geteuid() != 0 and not run_command("sudo -n true", check=False):
        logger.warning("This script needs sudo access to create PostgreSQL database and user.")
        logger.info("You may be prompted for your sudo password.")
    
    if setup_postgresql():
        logger.info("\n✨ PostgreSQL setup complete!")
        logger.info("\nNext steps:")
        logger.info("1. Source the environment: source .env.postgres")
        logger.info("2. Run blockchain sync: python blockchain_hybrid_sync.py")
        logger.info("3. Access dashboard: http://localhost:8888/unified")
        
        # Show connection info
        logger.info(f"\nConnection Details:")
        logger.info(f"  Host: {PG_CONFIG['host']}")
        logger.info(f"  Port: {PG_CONFIG['port']}")
        logger.info(f"  Database: {PG_CONFIG['database']}")
        logger.info(f"  User: {PG_CONFIG['user']}")
        
    else:
        logger.error("PostgreSQL setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()