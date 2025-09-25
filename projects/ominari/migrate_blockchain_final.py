#!/usr/bin/env python3
"""
Final blockchain data migration to PostgreSQL
"""

import logging
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_all_data():
    """Migrate blockchain data with proper error handling."""
    
    sqlite_engine = create_engine('sqlite:///sport_odds.db')
    postgres_engine = create_engine('postgresql://ominari_user:ominari_2025_secure@localhost:5435/ominari_production')
    
    logger.info("Starting blockchain data migration...")
    
    migrated_markets = 0
    migrated_odds = 0
    
    with sqlite_engine.connect() as sqlite_conn:
        # Get all blockchain markets
        markets = sqlite_conn.execute(text("""
            SELECT source_id, source, sport, league_name, home_team, away_team,
                   market_type, maturity_date, is_finished, updated_at
            FROM market
            WHERE source LIKE 'blockchain_%'
        """)).fetchall()
        
        logger.info(f"Processing {len(markets)} blockchain markets...")
        
        with postgres_engine.connect() as pg_conn:
            for market in markets:
                source_id = market[0]
                
                # Truncate source_id if too long (keep first 60 chars + hash)
                if len(source_id) > 66:
                    import hashlib
                    hash_suffix = hashlib.md5(source_id.encode()).hexdigest()[:6]
                    source_id = source_id[:60] + hash_suffix
                
                try:
                    pg_conn.execute(text("""
                        INSERT INTO market (source_id, source, sport, league_name,
                                          home_team, away_team, market_type,
                                          maturity_date, is_finished, updated_at)
                        VALUES (:source_id, :source, :sport, :league_name,
                                :home_team, :away_team, :market_type,
                                :maturity_date, :is_finished, :updated_at)
                        ON CONFLICT (source_id) DO UPDATE SET
                          updated_at = EXCLUDED.updated_at
                    """), {
                        'source_id': source_id,
                        'source': market[1],
                        'sport': market[2],
                        'league_name': market[3],
                        'home_team': market[4],
                        'away_team': market[5],
                        'market_type': market[6],
                        'maturity_date': market[7],
                        'is_finished': market[8],
                        'updated_at': market[9]
                    })
                    migrated_markets += 1
                    
                    if migrated_markets % 100 == 0:
                        logger.info(f"Migrated {migrated_markets} markets...")
                        
                except Exception as e:
                    logger.warning(f"Failed to migrate market {source_id[:50]}...: {e}")
            
            pg_conn.commit()
            logger.info(f"✅ Migrated {migrated_markets} markets")
        
        # Now migrate odds
        logger.info("Migrating odds...")
        odds = sqlite_conn.execute(text("""
            SELECT o.source_id, o.outcome, o.decimal_odds, o.market_type,
                   o.source, o.bookmaker, o.updated_at, o.position
            FROM odd o
            JOIN market m ON o.source_id = m.source_id
            WHERE m.source LIKE 'blockchain_%'
        """)).fetchall()
        
        logger.info(f"Processing {len(odds)} blockchain odds...")
        
        with postgres_engine.connect() as pg_conn:
            for odd in odds:
                original_source_id = odd[0]
                
                # Apply same truncation as markets
                if len(original_source_id) > 66:
                    import hashlib
                    hash_suffix = hashlib.md5(original_source_id.encode()).hexdigest()[:6]
                    source_id = original_source_id[:60] + hash_suffix
                else:
                    source_id = original_source_id
                
                try:
                    pg_conn.execute(text("""
                        INSERT INTO odd (source_id, outcome, decimal_odds, market_type,
                                       source, bookmaker, updated_at, position)
                        VALUES (:source_id, :outcome, :decimal_odds, :market_type,
                                :source, :bookmaker, :updated_at, :position)
                        ON CONFLICT DO NOTHING
                    """), {
                        'source_id': source_id,
                        'outcome': odd[1],
                        'decimal_odds': float(odd[2]) if odd[2] else 0.0,
                        'market_type': odd[3],
                        'source': odd[4],
                        'bookmaker': odd[5],
                        'updated_at': odd[6],
                        'position': int(odd[7]) if odd[7] is not None else 0
                    })
                    migrated_odds += 1
                    
                    if migrated_odds % 100 == 0:
                        logger.info(f"Migrated {migrated_odds} odds...")
                        
                except Exception as e:
                    logger.warning(f"Failed to migrate odd for {source_id[:30]}...: {e}")
            
            pg_conn.commit()
            logger.info(f"✅ Migrated {migrated_odds} odds")
    
    # Final verification
    with postgres_engine.connect() as pg_conn:
        market_count = pg_conn.execute(text(
            "SELECT COUNT(*) FROM market WHERE source LIKE 'blockchain_%'"
        )).scalar()
        
        soccer_count = pg_conn.execute(text(
            "SELECT COUNT(*) FROM market WHERE source LIKE 'blockchain_%' AND sport = 'Soccer'"
        )).scalar()
        
        odds_count = pg_conn.execute(text(
            "SELECT COUNT(*) FROM odd WHERE source LIKE 'blockchain_%'"
        )).scalar()
        
        logger.info(f"\n🎉 MIGRATION COMPLETE!")
        logger.info(f"PostgreSQL now has:")
        logger.info(f"  • {market_count:,} blockchain markets")
        logger.info(f"  • {soccer_count:,} soccer markets")  
        logger.info(f"  • {odds_count:,} odds records")
        
        if market_count > 0:
            # Show sample data
            sample = pg_conn.execute(text("""
                SELECT m.home_team || ' vs ' || m.away_team as match,
                       COUNT(o.source_id) as odds_count
                FROM market m
                LEFT JOIN odd o ON m.source_id = o.source_id
                WHERE m.source LIKE 'blockchain_%' 
                  AND m.sport = 'Soccer'
                  AND m.maturity_date > CURRENT_TIMESTAMP
                GROUP BY m.source_id, m.home_team, m.away_team, m.maturity_date
                ORDER BY m.maturity_date
                LIMIT 10
            """)).fetchall()
            
            logger.info(f"\n🔥 Sample Live Soccer Markets:")
            for match, odds_count in sample:
                logger.info(f"  • {match} ({odds_count} odds)")

if __name__ == "__main__":
    migrate_all_data()