#!/usr/bin/env python3
"""
Simple PostgreSQL configuration for Ominari
Uses Docker Compose for easy setup.
"""

import subprocess
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_postgres_docker():
    """Set up PostgreSQL using Docker Compose."""
    logger.info("🐘 Setting up PostgreSQL for Ominari using Docker")
    
    # Check if Docker is available
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        logger.info(f"Docker version: {result.stdout.strip()}")
    except FileNotFoundError:
        logger.error("Docker is not installed. Please install Docker first.")
        return False
    
    # Create Docker network if it doesn't exist
    logger.info("Creating Docker network...")
    subprocess.run(['docker', 'network', 'create', 'ominari-network'], 
                   capture_output=True, text=True)
    
    # Start PostgreSQL using docker-compose
    logger.info("Starting PostgreSQL container...")
    cmd = ['docker-compose', '-f', 'docker-compose.postgres.yml', 'up', '-d', 'postgres']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed to start PostgreSQL: {result.stderr}")
        return False
    
    logger.info("✅ PostgreSQL container started")
    
    # Wait for PostgreSQL to be ready
    logger.info("Waiting for PostgreSQL to be ready...")
    time.sleep(10)
    
    # Check if PostgreSQL is running
    check_cmd = ['docker-compose', '-f', 'docker-compose.postgres.yml', 'ps']
    result = subprocess.run(check_cmd, capture_output=True, text=True)
    logger.info(f"Container status:\n{result.stdout}")
    
    # Create environment file
    env_content = """# PostgreSQL Configuration for Ominari
export PG_HOST=localhost
export PG_PORT=5432
export PG_USER=ominari_user
export PG_PASSWORD=ominari_2025_secure
export PG_DB=ominari_production
"""
    
    with open('.env.postgres', 'w') as f:
        f.write(env_content)
    
    logger.info("✅ Environment variables saved to .env.postgres")
    
    # Update database_v2.py to use correct port
    update_database_config()
    
    return True

def update_database_config():
    """Update database configuration to use port 5432."""
    logger.info("Updating database configuration...")
    
    # Read current database_v2.py
    with open('database_v2.py', 'r') as f:
        content = f.read()
    
    # Replace port 5435 with 5432
    content = content.replace("'port': '5435'", "'port': '5432'")
    content = content.replace('port": "5435"', 'port": "5432"')
    
    # Write updated content
    with open('database_v2.py', 'w') as f:
        f.write(content)
    
    logger.info("✅ Updated database_v2.py to use port 5432")

def main():
    """Main setup function."""
    logger.info("=" * 60)
    logger.info("PostgreSQL Setup for Ominari (Docker)")
    logger.info("=" * 60)
    
    if setup_postgres_docker():
        logger.info("\n✨ PostgreSQL setup complete!")
        logger.info("\nNext steps:")
        logger.info("1. Source the environment: source .env.postgres")
        logger.info("2. Run blockchain sync: uv run python blockchain_hybrid_sync.py")
        logger.info("3. Check database: docker-compose -f docker-compose.postgres.yml logs postgres")
        logger.info("\nConnection info:")
        logger.info("  Host: localhost")
        logger.info("  Port: 5432")
        logger.info("  Database: ominari_production")
        logger.info("  User: ominari_user")
        logger.info("  Password: ominari_2025_secure")
    else:
        logger.error("Setup failed!")

if __name__ == "__main__":
    main()