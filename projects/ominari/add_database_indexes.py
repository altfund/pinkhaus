#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add database indexes to improve query performance for backtesting.
Includes indexes specifically optimized for the vectorized backtest queries.
"""

import sqlite3
import time
import logging

DB_NAME = "sport_odds.db"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_existing_indexes(cursor):
    """Check which indexes already exist."""
    cursor.execute("""
        SELECT name 
        FROM sqlite_master 
        WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
    """)
    return {row[0] for row in cursor.fetchall()}

def add_indexes():
    """Add indexes to improve query performance."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Enable WAL mode for better concurrency
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    
    logger.info("Checking existing indexes...")
    existing = check_existing_indexes(cursor)
    logger.info(f"Found {len(existing)} existing indexes")
    
    # Define indexes optimized for backtest queries
    indexes = [
        # Primary performance indexes for odd table
        ("idx_odd_source_updated", "odd", "(source_id, updated_at DESC)",
         "Primary index for fetching latest odds per market"),
        
        ("idx_odd_updated_at", "odd", "(updated_at)",
         "Time-based queries and window filtering"),
        
        ("idx_odd_updated_source_outcome", "odd", "(updated_at, source_id, outcome)",
         "Composite for time window market snapshots"),
        
        # Covering index for common query pattern
        ("idx_odd_covering", "odd", 
         "(source_id, updated_at DESC, market_type, outcome, decimal_odds)",
         "Covering index to avoid table lookups"),
         
        # Market table indexes
        ("idx_market_source", "market", "(source_id)",
         "Primary market lookup"),
         
        ("idx_market_maturity", "market", "(maturity_date)",
         "Finding markets that closed in time window"),
         
        ("idx_market_composite", "market", "(source_id, maturity_date, sport)",
         "Composite for filtered queries"),
    ]
    
    # Create indexes
    created = 0
    failed = 0
    
    for idx_name, table, columns, description in indexes:
        if idx_name in existing:
            logger.info(f"Skipping {idx_name} - already exists")
            continue
            
        try:
            logger.info(f"\nCreating {idx_name}: {description}")
            logger.info(f"  SQL: CREATE INDEX {idx_name} ON {table}{columns}")
            
            start = time.time()
            cursor.execute(f"CREATE INDEX {idx_name} ON {table}{columns}")
            elapsed = time.time() - start
            
            logger.info(f"  Created in {elapsed:.2f} seconds")
            created += 1
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            failed += 1
    
    # Update statistics
    if created > 0:
        logger.info("\nUpdating table statistics...")
        start = time.time()
        cursor.execute("ANALYZE")
        logger.info(f"Analysis completed in {time.time() - start:.2f} seconds")
    
    # Verify indexes
    logger.info("\nVerifying indexes...")
    cursor.execute("""
        SELECT name, tbl_name 
        FROM sqlite_master 
        WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
        ORDER BY tbl_name, name
    """)
    
    index_count = {}
    for idx_name, table in cursor.fetchall():
        index_count[table] = index_count.get(table, 0) + 1
        
    for table, count in index_count.items():
        logger.info(f"  {table}: {count} indexes")
    
    conn.commit()
    conn.close()
    
    logger.info(f"\nSummary: Created {created} indexes, {failed} failed")
    
    if created > 0:
        logger.info("\nIMPORTANT: First queries may be slow as indexes are populated.")
        logger.info("Subsequent queries should be much faster.")

def analyze_query_plans():
    """Analyze query execution plans to verify index usage."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    logger.info("\n" + "="*60)
    logger.info("QUERY PLAN ANALYSIS")
    logger.info("="*60)
    
    # Test queries used in backtest
    test_queries = [
        (
            "Latest odds per market",
            """
            WITH latest AS (
                SELECT source_id, MAX(updated_at) AS t
                FROM odd
                WHERE updated_at <= '2025-08-31T12:00:00'
                GROUP BY source_id
            )
            SELECT o.* FROM odd o
            JOIN latest l ON o.source_id = l.source_id AND o.updated_at = l.t
            LIMIT 10
            """
        ),
        (
            "Time window markets", 
            """
            SELECT DISTINCT source_id
            FROM odd
            WHERE updated_at BETWEEN '2025-08-30T00:00:00' AND '2025-08-31T00:00:00'
            LIMIT 10
            """
        ),
        (
            "Market maturity check",
            """
            SELECT source_id 
            FROM market
            WHERE maturity_date BETWEEN '2025-08-30T00:00:00' AND '2025-08-31T00:00:00'
            LIMIT 10
            """
        )
    ]
    
    for desc, query in test_queries:
        logger.info(f"\n{desc}:")
        try:
            plan = cursor.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
            for row in plan:
                logger.info(f"  {row}")
        except Exception as e:
            logger.error(f"  Error: {e}")
    
    conn.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize database for backtesting")
    parser.add_argument("--analyze", action="store_true", 
                       help="Analyze query plans after creating indexes")
    args = parser.parse_args()
    
    add_indexes()
    
    if args.analyze:
        analyze_query_plans()