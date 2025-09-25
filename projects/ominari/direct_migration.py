#!/usr/bin/env python3
"""
Direct SQLite to PostgreSQL migration using psycopg2
Bypasses ORM to use direct connections
"""

import sqlite3
import psycopg2
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PostgreSQL connection parameters
PG_PARAMS = {
    'host': 'localhost',
    'port': 5999,
    'user': 'ominari_user',
    'password': 'ominari_2025_secure',
    'database': 'ominari_production'
}

def migrate_data():
    """Migrate all data from SQLite to PostgreSQL"""
    
    # Connect to SQLite
    logger.info("📊 Connecting to SQLite...")
    sqlite_conn = sqlite3.connect('sport_odds.db')
    sqlite_conn.row_factory = sqlite3.Row
    
    # Connect to PostgreSQL
    logger.info("🐘 Connecting to PostgreSQL...")
    pg_conn = psycopg2.connect(**PG_PARAMS)
    pg_conn.autocommit = False
    
    try:
        with sqlite_conn, pg_conn:
            sqlite_cursor = sqlite_conn.cursor()
            pg_cursor = pg_conn.cursor()
            
            # Count existing data
            sqlite_cursor.execute("SELECT COUNT(*) FROM market")
            sqlite_markets = sqlite_cursor.fetchone()[0]
            
            pg_cursor.execute("SELECT COUNT(*) FROM market")
            pg_markets = pg_cursor.fetchone()[0]
            
            logger.info(f"SQLite markets: {sqlite_markets:,}")
            logger.info(f"PostgreSQL markets: {pg_markets:,}")
            
            # Migrate markets
            logger.info("🔄 Migrating markets...")
            sqlite_cursor.execute("""
                SELECT source_id, home_team, away_team, sport, source, 
                       maturity_date, league_name, is_finished, home_score, away_score, updated_at
                FROM market
            """)
            
            markets_migrated = 0
            for row in sqlite_cursor.fetchall():
                try:
                    # Check if market already exists
                    pg_cursor.execute("SELECT 1 FROM market WHERE source_id = %s", (row['source_id'],))
                    if pg_cursor.fetchone():
                        continue
                    
                    # Parse datetime
                    maturity_date = None
                    if row['maturity_date']:
                        try:
                            maturity_date = datetime.fromisoformat(row['maturity_date'].replace('Z', '+00:00'))
                        except:
                            pass
                    
                    updated_at = datetime.now()
                    if row['updated_at']:
                        try:
                            updated_at = datetime.fromisoformat(row['updated_at'].replace('Z', '+00:00'))
                        except:
                            pass
                    
                    # Insert into PostgreSQL (matching actual schema)
                    pg_cursor.execute("""
                        INSERT INTO market (
                            source_id, home_team, away_team, sport, source,
                            maturity_date, league_name, is_finished, home_score, away_score,
                            updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        row['source_id'], row['home_team'], row['away_team'], row['sport'], row['source'],
                        maturity_date, row['league_name'], bool(row['is_finished']), 
                        row['home_score'], row['away_score'], updated_at
                    ))
                    
                    markets_migrated += 1
                    
                    if markets_migrated % 1000 == 0:
                        logger.info(f"Migrated {markets_migrated:,} markets...")
                        pg_conn.commit()
                        
                except Exception as e:
                    logger.error(f"Error migrating market {row['source_id']}: {e}")
                    continue
            
            # Final commit for markets
            pg_conn.commit()
            logger.info(f"✅ Markets migrated: {markets_migrated:,}")
            
            # Migrate odds
            logger.info("🎲 Migrating odds...")
            sqlite_cursor.execute("""
                SELECT source_id, outcome, decimal_odds, american_odds, position, updated_at
                FROM odd
            """)
            
            odds_migrated = 0
            for row in sqlite_cursor.fetchall():
                try:
                    # Check if odd already exists
                    pg_cursor.execute(
                        "SELECT 1 FROM odd WHERE source_id = %s AND outcome = %s", 
                        (row['source_id'], row['outcome'])
                    )
                    if pg_cursor.fetchone():
                        continue
                    
                    # Parse datetime
                    updated_at = datetime.now()
                    if row['updated_at']:
                        try:
                            updated_at = datetime.fromisoformat(row['updated_at'].replace('Z', '+00:00'))
                        except:
                            pass
                    
                    # Insert into PostgreSQL (matching actual schema)
                    pg_cursor.execute("""
                        INSERT INTO odd (
                            source_id, outcome, decimal_odds, american_odds, position,
                            market_type, source, bookmaker, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        row['source_id'], row['outcome'], row['decimal_odds'], 
                        row['american_odds'], row['position'], 
                        'moneyline', 'api_import', 'ominari', updated_at
                    ))
                    
                    odds_migrated += 1
                    
                    if odds_migrated % 5000 == 0:
                        logger.info(f"Migrated {odds_migrated:,} odds...")
                        pg_conn.commit()
                        
                except Exception as e:
                    logger.error(f"Error migrating odd {row['source_id']}/{row['outcome']}: {e}")
                    continue
            
            # Final commit for odds
            pg_conn.commit()
            logger.info(f"✅ Odds migrated: {odds_migrated:,}")
            
            # Verify migration
            pg_cursor.execute("SELECT COUNT(*) FROM market")
            final_markets = pg_cursor.fetchone()[0]
            
            pg_cursor.execute("SELECT COUNT(*) FROM odd")
            final_odds = pg_cursor.fetchone()[0]
            
            logger.info(f"🎉 Migration complete!")
            logger.info(f"📊 Final counts: {final_markets:,} markets, {final_odds:,} odds")
            
            # Show source breakdown
            pg_cursor.execute("SELECT source, COUNT(*) FROM market GROUP BY source ORDER BY COUNT(*) DESC")
            logger.info("📈 Markets by source:")
            for source, count in pg_cursor.fetchall():
                logger.info(f"  {source}: {count:,}")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        pg_conn.rollback()
        raise
    finally:
        sqlite_conn.close()
        pg_conn.close()

if __name__ == '__main__':
    logger.info("🚀 Starting direct SQLite to PostgreSQL migration...")
    migrate_data()
    logger.info("🎯 Your dashboard should now show all migrated data!")