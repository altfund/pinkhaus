#!/usr/bin/env python3
"""
Migrate all data from SQLite to PostgreSQL
Transfers markets and odds from sport_odds.db to PostgreSQL database
"""

import sqlite3
import os
from database_v2 import db_manager
from models import Market, Odd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set PostgreSQL environment variables
os.environ['PGHOST'] = 'localhost'
os.environ['PGPORT'] = '5999'
os.environ['PGUSER'] = 'ominari_user'
os.environ['PGPASSWORD'] = 'ominari_2025_secure'
os.environ['PGDATABASE'] = 'ominari_production'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'
os.environ['FORCE_POSTGRESQL'] = '1'

def migrate_markets():
    """Migrate all markets from SQLite to PostgreSQL"""
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect('sport_odds.db')
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cursor = sqlite_conn.cursor()
    
    # Get all markets from SQLite
    logger.info("📊 Fetching markets from SQLite...")
    sqlite_cursor.execute("SELECT COUNT(*) FROM market")
    total_markets = sqlite_cursor.fetchone()[0]
    logger.info(f"Found {total_markets:,} markets in SQLite")
    
    sqlite_cursor.execute("""
        SELECT source_id, home_team, away_team, sport, source, 
               maturity_date, league_name, is_finished, home_score, away_score,
               updated_at
        FROM market
    """)
    
    markets_migrated = 0
    markets_skipped = 0
    
    with db_manager.get_db_session() as pg_db:
        logger.info("🔄 Starting market migration to PostgreSQL...")
        
        for row in sqlite_cursor.fetchall():
            try:
                # Check if market already exists in PostgreSQL
                existing = pg_db.query(Market).filter(
                    Market.source_id == row['source_id']
                ).first()
                
                if existing:
                    markets_skipped += 1
                    continue
                
                # Parse datetime fields
                maturity_date = None
                if row['maturity_date']:
                    try:
                        maturity_date = datetime.fromisoformat(row['maturity_date'].replace('Z', '+00:00'))
                    except:
                        pass
                
                updated_at = None
                if row['updated_at']:
                    try:
                        updated_at = datetime.fromisoformat(row['updated_at'].replace('Z', '+00:00'))
                    except:
                        updated_at = datetime.now()
                else:
                    updated_at = datetime.now()
                
                # Create new market in PostgreSQL
                market = Market(
                    source_id=row['source_id'],
                    home_team=row['home_team'],
                    away_team=row['away_team'],
                    sport=row['sport'],
                    source=row['source'],
                    maturity_date=maturity_date,
                    league_name=row['league_name'],
                    is_finished=bool(row['is_finished']),
                    home_score=row['home_score'],
                    away_score=row['away_score'],
                    created_at=updated_at,  # Use updated_at as created_at
                    updated_at=updated_at
                )
                
                pg_db.add(market)
                markets_migrated += 1
                
                if markets_migrated % 1000 == 0:
                    logger.info(f"Migrated {markets_migrated:,} markets...")
                    pg_db.commit()
                    
            except Exception as e:
                logger.error(f"Error migrating market {row['source_id']}: {e}")
                continue
        
        # Final commit
        pg_db.commit()
        logger.info(f"✅ Market migration complete: {markets_migrated:,} migrated, {markets_skipped:,} skipped")
    
    sqlite_conn.close()
    return markets_migrated

def migrate_odds():
    """Migrate all odds from SQLite to PostgreSQL"""
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect('sport_odds.db')
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cursor = sqlite_conn.cursor()
    
    # Get all odds from SQLite
    logger.info("🎲 Fetching odds from SQLite...")
    sqlite_cursor.execute("SELECT COUNT(*) FROM odd")
    total_odds = sqlite_cursor.fetchone()[0]
    logger.info(f"Found {total_odds:,} odds in SQLite")
    
    sqlite_cursor.execute("""
        SELECT source_id, outcome, decimal_odds, american_odds, position,
               updated_at
        FROM odd
    """)
    
    odds_migrated = 0
    odds_skipped = 0
    
    with db_manager.get_db_session() as pg_db:
        logger.info("🔄 Starting odds migration to PostgreSQL...")
        
        for row in sqlite_cursor.fetchall():
            try:
                # Check if odd already exists in PostgreSQL
                existing = pg_db.query(Odd).filter(
                    Odd.source_id == row['source_id'],
                    Odd.outcome == row['outcome']
                ).first()
                
                if existing:
                    odds_skipped += 1
                    continue
                
                # Parse datetime fields
                updated_at = None
                if row['updated_at']:
                    try:
                        updated_at = datetime.fromisoformat(row['updated_at'].replace('Z', '+00:00'))
                    except:
                        updated_at = datetime.now()
                else:
                    updated_at = datetime.now()
                
                # Create new odd in PostgreSQL
                odd = Odd(
                    source_id=row['source_id'],
                    outcome=row['outcome'],
                    decimal_odds=row['decimal_odds'],
                    american_odds=row['american_odds'],
                    position=row['position'],
                    created_at=updated_at,  # Use updated_at as created_at
                    updated_at=updated_at
                )
                
                pg_db.add(odd)
                odds_migrated += 1
                
                if odds_migrated % 5000 == 0:
                    logger.info(f"Migrated {odds_migrated:,} odds...")
                    pg_db.commit()
                    
            except Exception as e:
                logger.error(f"Error migrating odd for {row['source_id']}: {e}")
                continue
        
        # Final commit
        pg_db.commit()
        logger.info(f"✅ Odds migration complete: {odds_migrated:,} migrated, {odds_skipped:,} skipped")
    
    sqlite_conn.close()
    return odds_migrated

def verify_migration():
    """Verify migration was successful"""
    logger.info("🔍 Verifying migration...")
    
    with db_manager.get_db_session() as pg_db:
        # Count markets
        market_count = pg_db.query(Market).count()
        logger.info(f"PostgreSQL markets: {market_count:,}")
        
        # Count odds
        odds_count = pg_db.query(Odd).count()
        logger.info(f"PostgreSQL odds: {odds_count:,}")
        
        # Count by source
        from sqlalchemy import func
        sources = pg_db.query(
            Market.source, 
            func.count(Market.source_id).label('count')
        ).group_by(Market.source).all()
        
        logger.info("📊 Markets by source:")
        for source, count in sources:
            logger.info(f"  {source}: {count:,}")
        
        # Count sports
        sports = pg_db.query(
            Market.sport,
            func.count(Market.source_id).label('count')
        ).group_by(Market.sport).order_by(func.count(Market.source_id).desc()).limit(10).all()
        
        logger.info("🏆 Top sports:")
        for sport, count in sports:
            logger.info(f"  {sport}: {count:,}")

def main():
    logger.info("🚀 Starting SQLite to PostgreSQL migration...")
    
    try:
        # Migrate markets
        markets_migrated = migrate_markets()
        
        # Migrate odds  
        odds_migrated = migrate_odds()
        
        # Verify migration
        verify_migration()
        
        logger.info(f"🎉 Migration complete!")
        logger.info(f"📊 Total migrated: {markets_migrated:,} markets, {odds_migrated:,} odds")
        logger.info("🎯 Your dashboard at http://localhost:8888 should now show all data!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == '__main__':
    main()