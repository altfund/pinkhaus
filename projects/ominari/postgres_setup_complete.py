#!/usr/bin/env python3
"""
Complete PostgreSQL setup with schema and data migration
"""

import os
import logging
from sqlalchemy import create_engine, text
from models import Base, Market, Odd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment for PostgreSQL on port 5435
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5435',
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})

def setup_postgres_and_migrate():
    """Create schema and migrate blockchain data to PostgreSQL."""
    
    # PostgreSQL connection
    pg_url = "postgresql://ominari_user:ominari_2025_secure@localhost:5435/ominari_production"
    pg_engine = create_engine(pg_url)
    
    # Create all tables
    logger.info("Creating database schema...")
    Base.metadata.create_all(pg_engine)
    logger.info("✅ Schema created successfully")
    
    # Connect to SQLite for data source
    sqlite_engine = create_engine('sqlite:///sport_odds.db')
    
    # Migrate blockchain markets
    logger.info("Migrating blockchain markets from SQLite...")
    
    with sqlite_engine.connect() as src:
        # Count total markets
        total_count = src.execute(text(
            "SELECT COUNT(*) FROM market WHERE source LIKE 'blockchain_%'"
        )).scalar()
        logger.info(f"Found {total_count} blockchain markets to migrate")
        
        # Get markets in batches
        batch_size = 100
        migrated = 0
        
        for offset in range(0, total_count, batch_size):
            markets = src.execute(text("""
                SELECT source_id, source, sport, league_name, home_team, away_team, 
                       market_type, maturity_date, is_finished, updated_at
                FROM market 
                WHERE source LIKE 'blockchain_%'
                LIMIT :limit OFFSET :offset
            """), {"limit": batch_size, "offset": offset}).fetchall()
            
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
                        migrated += 1
                    except Exception as e:
                        logger.debug(f"Skipping market: {e}")
                dst.commit()
            
            logger.info(f"Migrated {migrated}/{total_count} markets...")
    
    # Migrate odds
    logger.info("Migrating blockchain odds...")
    
    with sqlite_engine.connect() as src:
        # Count odds
        odds_count = src.execute(text("""
            SELECT COUNT(*) FROM odd o
            JOIN market m ON o.source_id = m.source_id
            WHERE m.source LIKE 'blockchain_%'
        """)).scalar()
        logger.info(f"Found {odds_count} blockchain odds to migrate")
        
        # Get odds in batches
        migrated_odds = 0
        
        for offset in range(0, odds_count, batch_size * 5):
            odds = src.execute(text("""
                SELECT o.source_id, o.position, o.outcome, o.decimal_odds,
                       o.market_type, o.source, o.bookmaker, o.updated_at
                FROM odd o
                JOIN market m ON o.source_id = m.source_id
                WHERE m.source LIKE 'blockchain_%'
                LIMIT :limit OFFSET :offset
            """), {"limit": batch_size * 5, "offset": offset}).fetchall()
            
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
                        migrated_odds += 1
                    except Exception as e:
                        logger.debug(f"Skipping odd: {e}")
                dst.commit()
            
            logger.info(f"Migrated {migrated_odds}/{odds_count} odds...")
    
    # Verify migration
    with pg_engine.connect() as conn:
        market_count = conn.execute(text(
            "SELECT COUNT(*) FROM market WHERE source LIKE 'blockchain_%'"
        )).scalar()
        soccer_count = conn.execute(text(
            "SELECT COUNT(*) FROM market WHERE source LIKE 'blockchain_%' AND sport = 'Soccer'"
        )).scalar()
        odds_count = conn.execute(text(
            "SELECT COUNT(*) FROM odd WHERE source LIKE 'blockchain_%'"
        )).scalar()
        
        logger.info("\n✅ Migration Complete!")
        logger.info(f"PostgreSQL now contains:")
        logger.info(f"  - {market_count} blockchain markets")
        logger.info(f"  - {soccer_count} soccer markets")
        logger.info(f"  - {odds_count} odds records")
        
        # Show sample data
        sample = conn.execute(text("""
            SELECT m.home_team || ' vs ' || m.away_team as match,
                   STRING_AGG(o.outcome || ': ' || ROUND(o.decimal_odds::numeric, 2), ', ') as odds
            FROM market m
            LEFT JOIN odd o ON m.source_id = o.source_id
            WHERE m.source LIKE 'blockchain_%' 
              AND m.sport = 'Soccer'
              AND m.maturity_date > CURRENT_TIMESTAMP
            GROUP BY m.source_id, m.home_team, m.away_team
            ORDER BY m.maturity_date
            LIMIT 5
        """)).fetchall()
        
        logger.info("\n🔥 Sample Live Blockchain Markets:")
        for match, odds in sample:
            logger.info(f"  • {match} - {odds or 'No odds yet'}")

if __name__ == "__main__":
    setup_postgres_and_migrate()