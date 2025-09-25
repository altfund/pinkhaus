#!/usr/bin/env python3
"""
Run Ominari with PostgreSQL
This script sets up the environment to use PostgreSQL and starts the web monitor.
"""

import os
import sys
import subprocess
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up environment for PostgreSQL."""
    # PostgreSQL configuration
    os.environ.update({
        'PG_HOST': 'localhost',
        'PG_PORT': '5432',
        'PG_USER': 'ominari_user',
        'PG_PASSWORD': 'ominari_2025_secure',
        'PG_DB': 'ominari_production',
        'USE_POSTGRESQL': '1',
        'FORCE_POSTGRESQL': '1'
    })
    
    logger.info("Environment configured for PostgreSQL")
    
    # Save to file for future use
    with open('.env.postgres', 'w') as f:
        f.write("""# PostgreSQL Configuration
export PG_HOST=localhost
export PG_PORT=5432
export PG_USER=ominari_user
export PG_PASSWORD=ominari_2025_secure
export PG_DB=ominari_production
export USE_POSTGRESQL=1
export FORCE_POSTGRESQL=1
""")
    
    logger.info("Configuration saved to .env.postgres")

def test_postgres_connection():
    """Test if we can connect to PostgreSQL."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.environ['PG_HOST'],
            port=os.environ['PG_PORT'],
            database=os.environ['PG_DB'],
            user=os.environ['PG_USER'],
            password=os.environ['PG_PASSWORD']
        )
        conn.close()
        logger.info("✅ PostgreSQL connection successful!")
        return True
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        return False

def get_blockchain_data_from_sqlite():
    """Get blockchain data from SQLite for display."""
    try:
        from sqlalchemy import create_engine, text
        
        # Connect to SQLite
        engine = create_engine('sqlite:///sport_odds.db')
        
        with engine.connect() as conn:
            # Get blockchain market count
            result = conn.execute(text("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN sport = 'Soccer' THEN 1 ELSE 0 END) as soccer
                FROM market
                WHERE source LIKE 'blockchain_%'
            """)).fetchone()
            
            total_markets = result[0] or 0
            soccer_markets = result[1] or 0
            
            # Get sample markets
            markets = conn.execute(text("""
                SELECT m.source_id, m.home_team, m.away_team, m.sport, m.maturity_date,
                       GROUP_CONCAT(o.outcome || ':' || ROUND(o.decimal_odds, 2)) as odds
                FROM market m
                LEFT JOIN odd o ON m.source_id = o.source_id
                WHERE m.source LIKE 'blockchain_%' 
                  AND m.sport = 'Soccer'
                  AND m.maturity_date > datetime('now')
                GROUP BY m.source_id
                ORDER BY m.maturity_date
                LIMIT 10
            """)).fetchall()
            
            logger.info(f"\n📊 Blockchain Data Summary:")
            logger.info(f"Total blockchain markets: {total_markets}")
            logger.info(f"Soccer markets: {soccer_markets}")
            logger.info(f"\n🔥 Live Soccer Markets (from blockchain):")
            
            for market in markets:
                logger.info(f"  • {market[1]} vs {market[2]} - {market[5] or 'No odds yet'}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error reading blockchain data: {e}")
        return False

def start_web_monitor():
    """Start the web monitor with PostgreSQL configuration."""
    logger.info("\n🚀 Starting Web Monitor...")
    
    # Kill any existing web monitor
    subprocess.run(['pkill', '-f', 'web_monitor.py'], capture_output=True)
    time.sleep(2)
    
    # Start web monitor
    cmd = ['uv', 'run', 'python', 'web_monitor.py']
    
    logger.info(f"Running: {' '.join(cmd)}")
    logger.info("Web monitor starting on http://localhost:8888/")
    logger.info("Access unified dashboard at: http://localhost:8888/unified")
    
    # Run in foreground so we can see logs
    subprocess.run(cmd)

def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("🏃 Running Ominari with Real Blockchain Data")
    logger.info("=" * 60)
    
    # Set up environment
    setup_environment()
    
    # Show blockchain data summary
    get_blockchain_data_from_sqlite()
    
    # Test PostgreSQL (optional - will fall back to SQLite if not available)
    postgres_available = test_postgres_connection()
    
    if not postgres_available:
        logger.warning("\n⚠️  PostgreSQL not available, will use SQLite")
        logger.info("To set up PostgreSQL:")
        logger.info("1. Run: sudo -u postgres psql < create_postgres_db.sql")
        logger.info("2. Then run this script again")
        logger.info("\n📊 Using SQLite with real blockchain data...")
    else:
        logger.info("\n✅ Using PostgreSQL for better performance!")
    
    # Start web monitor
    logger.info("\n" + "=" * 60)
    start_web_monitor()

if __name__ == "__main__":
    main()