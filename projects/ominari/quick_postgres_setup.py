#!/usr/bin/env python3
"""
Quick PostgreSQL setup and web monitor restart
"""

import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def restart_web_monitor():
    """Restart web monitor with proper PostgreSQL configuration."""
    
    # Kill existing web monitor
    logger.info("Stopping existing web monitor...")
    subprocess.run(['pkill', '-f', 'web_monitor.py'], capture_output=True)
    
    # Set up environment
    env = os.environ.copy()
    env.update({
        'PG_HOST': 'localhost',
        'PG_PORT': '5435',
        'PG_USER': 'ominari_user',
        'PG_PASSWORD': 'ominari_2025_secure',
        'PG_DB': 'ominari_production',
        'USE_POSTGRESQL': '1'
    })
    
    # Insert a few sample markets manually to test
    logger.info("Creating sample blockchain markets...")
    from sqlalchemy import create_engine, text
    
    pg_engine = create_engine('postgresql://ominari_user:ominari_2025_secure@localhost:5435/ominari_production')
    
    with pg_engine.connect() as conn:
        # Insert sample blockchain market
        conn.execute(text("""
            INSERT INTO market (source_id, source, sport, league_name, 
                              home_team, away_team, market_type, is_finished)
            VALUES ('blockchain_test_001', 'blockchain_optimism_v2', 'Soccer', 'Test League',
                    'Real Madrid', 'Barcelona', 'winner', false)
            ON CONFLICT (source_id) DO NOTHING
        """))
        
        # Insert sample odds
        conn.execute(text("""
            INSERT INTO odd (source_id, outcome, decimal_odds, market_type,
                           source, bookmaker, position)
            VALUES ('blockchain_test_001', 'Home', 2.10, 'moneyline',
                    'blockchain_optimism_v2', 'overtime', 0),
                   ('blockchain_test_001', 'Away', 3.20, 'moneyline',
                    'blockchain_optimism_v2', 'overtime', 1),
                   ('blockchain_test_001', 'Draw', 3.40, 'moneyline',
                    'blockchain_optimism_v2', 'overtime', 2)
            ON CONFLICT DO NOTHING
        """))
        
        conn.commit()
        logger.info("✅ Sample blockchain data created")
        
        # Verify
        count = conn.execute(text(
            "SELECT COUNT(*) FROM market WHERE source LIKE 'blockchain_%'"
        )).scalar()
        logger.info(f"PostgreSQL now has {count} blockchain markets")
    
    # Start web monitor
    logger.info("🚀 Starting web monitor with PostgreSQL...")
    logger.info("Dashboard will be at: http://localhost:8888/unified")
    
    # Run web monitor with proper environment
    cmd = ['uv', 'run', 'python', 'web_monitor.py']
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    restart_web_monitor()