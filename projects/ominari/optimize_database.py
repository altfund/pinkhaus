#!/usr/bin/env python3
"""Optimize the main database to reduce size from 200GB+."""

import logging
import time
from datetime import datetime, timedelta, timezone
from sqlalchemy import text, create_engine, Index
from database_v2 import db_manager
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """Optimize the sport odds database."""
    
    def __init__(self):
        self.start_time = time.time()
        
    def create_optimized_schema(self):
        """Create optimized tables."""
        logger.info("Creating optimized schema...")
        
        with db_manager.get_db_session() as db:
            # Create teams table
            db.execute(text("""
                CREATE TABLE IF NOT EXISTS teams (
                    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_name VARCHAR(100) UNIQUE NOT NULL
                )
            """))
            
            # Create optimized odds table for recent data
            db.execute(text("""
                CREATE TABLE IF NOT EXISTS odds_recent (
                    market_id VARCHAR(68) NOT NULL,
                    outcome TINYINT NOT NULL,
                    odds_x1000 INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (market_id, outcome, updated_at)
                ) WITHOUT ROWID
            """))
            
            # Create final odds summary table
            db.execute(text("""
                CREATE TABLE IF NOT EXISTS odds_final (
                    market_id VARCHAR(68) NOT NULL,
                    outcome TINYINT NOT NULL,
                    final_odds_x1000 INTEGER NOT NULL,
                    min_odds_x1000 INTEGER NOT NULL,
                    max_odds_x1000 INTEGER NOT NULL,
                    avg_odds_x1000 INTEGER NOT NULL,
                    update_count INTEGER NOT NULL,
                    last_updated INTEGER NOT NULL,
                    PRIMARY KEY (market_id, outcome)
                ) WITHOUT ROWID
            """))
            
            # Create archive table for old odds
            db.execute(text("""
                CREATE TABLE IF NOT EXISTS odds_archive (
                    market_id VARCHAR(68) NOT NULL,
                    outcome TINYINT NOT NULL,
                    odds_x1000 INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
            """))
            
            # Create indexes
            db.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_odds_archive_market 
                ON odds_archive(market_id)
            """))
            
            db.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_odds_recent_updated 
                ON odds_recent(updated_at)
            """))
            
            db.commit()
            logger.info("Optimized schema created")
            
    def normalize_teams(self):
        """Normalize team names to save space."""
        logger.info("Normalizing team names...")
        
        with db_manager.get_db_session() as db:
            # Get unique teams
            result = db.execute(text("""
                SELECT DISTINCT home_team FROM market 
                WHERE home_team IS NOT NULL
                UNION
                SELECT DISTINCT away_team FROM market 
                WHERE away_team IS NOT NULL
                LIMIT 10000
            """))
            
            teams = set()
            for row in result:
                teams.add(row[0])
                
            logger.info(f"Found {len(teams)} unique teams")
            
            # Insert teams
            for team_name in teams:
                try:
                    db.execute(text("""
                        INSERT OR IGNORE INTO teams (team_name) 
                        VALUES (:team_name)
                    """), {"team_name": team_name})
                except:
                    pass
                    
            db.commit()
            logger.info("Teams normalized")
            
    def calculate_odds_summary(self):
        """Calculate odds summaries for each market."""
        logger.info("Calculating odds summaries...")
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
        
        with db_manager.get_db_session() as db:
            # Get markets that need summaries
            result = db.execute(text("""
                SELECT DISTINCT source_id 
                FROM market 
                WHERE maturity_date > :cutoff
                LIMIT 1000
            """), {"cutoff": cutoff_date})
            
            market_ids = [row[0] for row in result]
            logger.info(f"Processing {len(market_ids)} recent markets")
            
            for i, market_id in enumerate(market_ids):
                if i % 100 == 0:
                    logger.info(f"Progress: {i}/{len(market_ids)}")
                    
                # Calculate summary for each outcome
                for outcome_idx, outcome_name in enumerate(['option_1', 'option_2', 'option_3']):
                    try:
                        summary = db.execute(text(f"""
                            SELECT 
                                MIN(odds_{outcome_name}),
                                MAX(odds_{outcome_name}),
                                AVG(odds_{outcome_name}),
                                COUNT(*),
                                MAX(updated_at)
                            FROM odd
                            WHERE source_id = :market_id
                            AND odds_{outcome_name} IS NOT NULL
                            AND odds_{outcome_name} > 1
                        """), {"market_id": market_id}).first()
                        
                        if summary and summary[0]:
                            min_odds, max_odds, avg_odds, count, last_update = summary
                            
                            # Get final odds (most recent)
                            final_result = db.execute(text(f"""
                                SELECT odds_{outcome_name}
                                FROM odd
                                WHERE source_id = :market_id
                                AND odds_{outcome_name} IS NOT NULL
                                ORDER BY updated_at DESC
                                LIMIT 1
                            """), {"market_id": market_id}).first()
                            
                            final_odds = final_result[0] if final_result else avg_odds
                            
                            # Insert summary
                            db.execute(text("""
                                INSERT OR REPLACE INTO odds_final
                                (market_id, outcome, final_odds_x1000, min_odds_x1000, 
                                 max_odds_x1000, avg_odds_x1000, update_count, last_updated)
                                VALUES (:market_id, :outcome, :final, :min, :max, :avg, :count, :updated)
                            """), {
                                "market_id": market_id,
                                "outcome": outcome_idx,
                                "final": int(final_odds * 1000),
                                "min": int(min_odds * 1000),
                                "max": int(max_odds * 1000),
                                "avg": int(avg_odds * 1000),
                                "count": count,
                                "updated": int(last_update.timestamp()) if last_update else 0
                            })
                    except Exception as e:
                        logger.error(f"Error processing {market_id} outcome {outcome_name}: {e}")
                        
                if i % 100 == 0:
                    db.commit()
                    
            db.commit()
            logger.info("Odds summaries calculated")
            
    def create_compatibility_views(self):
        """Create views for backward compatibility."""
        logger.info("Creating compatibility views...")
        
        with db_manager.get_db_session() as db:
            # Create view that mimics old odd table structure
            db.execute(text("""
                CREATE VIEW IF NOT EXISTS odd_view AS
                SELECT 
                    m.source_id,
                    o1.final_odds_x1000 / 1000.0 as odds_option_1,
                    o2.final_odds_x1000 / 1000.0 as odds_option_2,
                    o3.final_odds_x1000 / 1000.0 as odds_option_3,
                    datetime(o1.last_updated, 'unixepoch') as updated_at
                FROM market m
                LEFT JOIN odds_final o1 ON m.source_id = o1.market_id AND o1.outcome = 0
                LEFT JOIN odds_final o2 ON m.source_id = o2.market_id AND o2.outcome = 1
                LEFT JOIN odds_final o3 ON m.source_id = o3.market_id AND o3.outcome = 2
            """))
            
            db.commit()
            logger.info("Compatibility views created")
            
    def analyze_savings(self):
        """Analyze space savings."""
        logger.info("Analyzing space savings...")
        
        with db_manager.get_db_session() as db:
            # Count records in new tables
            final_count = db.execute(text(
                "SELECT COUNT(*) FROM odds_final"
            )).scalar()
            
            recent_count = db.execute(text(
                "SELECT COUNT(*) FROM odds_recent"
            )).scalar()
            
            logger.info(f"Odds final records: {final_count:,}")
            logger.info(f"Odds recent records: {recent_count:,}")
            
            # Estimate savings
            original_size = 515_919_464 * 50  # ~50 bytes per record
            new_size = (final_count * 40) + (recent_count * 20)  # Optimized sizes
            
            savings_gb = (original_size - new_size) / (1024**3)
            logger.info(f"Estimated storage savings: {savings_gb:.1f} GB")
            
    def run_optimization(self):
        """Run the full optimization process."""
        logger.info("Starting database optimization...")
        
        # Step 1: Create schema
        self.create_optimized_schema()
        
        # Step 2: Normalize teams
        self.normalize_teams()
        
        # Step 3: Calculate summaries
        self.calculate_odds_summary()
        
        # Step 4: Create views
        self.create_compatibility_views()
        
        # Step 5: Analyze results
        self.analyze_savings()
        
        elapsed = time.time() - self.start_time
        logger.info(f"Optimization completed in {elapsed:.1f} seconds")
        
        logger.info("\nNext steps:")
        logger.info("1. Test the new views to ensure compatibility")
        logger.info("2. Migrate recent odds data to odds_recent table")
        logger.info("3. Archive old odds data")
        logger.info("4. Update application to use new tables")
        logger.info("5. Drop the original odds table to reclaim space")


if __name__ == "__main__":
    optimizer = DatabaseOptimizer()
    optimizer.run_optimization()