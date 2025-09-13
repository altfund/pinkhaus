#!/usr/bin/env python3
"""Test migration speed with current setup."""

import time
from sqlalchemy import text
from database_v2 import db_manager
from datetime import datetime, timezone

def test_speed():
    """Test how fast we can migrate records."""
    
    with db_manager.get_db_session() as db:
        # Get current lookups
        bookmakers = dict(db.execute(text("SELECT name, id FROM lu_bookmakers")).fetchall())
        sources = dict(db.execute(text("SELECT name, id FROM lu_sources")).fetchall())
        market_types = dict(db.execute(text("SELECT name, id FROM lu_market_types")).fetchall())
        
        print(f"Lookups loaded:")
        print(f"  Bookmakers: {len(bookmakers)}")
        print(f"  Sources: {len(sources)}")
        print(f"  Market types: {len(market_types)}")
        
        # Test with 1000 records
        print("\nTesting migration speed...")
        start_time = time.time()
        
        records = db.execute(text("""
            SELECT 
                source_id, bookmaker, source, market_type, outcome,
                COALESCE(position, 0) as position, line, decimal_odds,
                american_odds, normalized_implied, updated_at
            FROM odd
            LIMIT 1000
        """)).fetchall()
        
        fetch_time = time.time() - start_time
        print(f"Fetched 1000 records in {fetch_time:.2f} seconds")
        
        # Transform records
        transform_start = time.time()
        migrated = 0
        skipped = 0
        
        for record in records:
            # Skip if lookups missing
            if (record[1] not in bookmakers or 
                record[2] not in sources or 
                record[3] not in market_types):
                skipped += 1
                continue
            
            # Map outcome
            outcome_id = {'option_1': 0, 'option_2': 1, 'option_3': 2}.get(record[4])
            if outcome_id is None:
                skipped += 1
                continue
            
            migrated += 1
        
        transform_time = time.time() - transform_start
        
        print(f"Transformed in {transform_time:.2f} seconds")
        print(f"  Migrated: {migrated}")
        print(f"  Skipped: {skipped}")
        
        # Calculate rates
        total_time = fetch_time + transform_time
        rate = migrated / total_time if total_time > 0 else 0
        
        print(f"\nRate: {rate:.0f} records/second")
        print(f"Estimated time for 515M records: {515_919_464 / rate / 3600:.1f} hours")
        
        # Check what we're skipping
        if skipped > 0:
            print("\nChecking why records were skipped...")
            
            # Get unique values we don't have
            missing = db.execute(text("""
                SELECT DISTINCT bookmaker, source, market_type
                FROM odd
                WHERE bookmaker NOT IN (SELECT name FROM lu_bookmakers)
                   OR source NOT IN (SELECT name FROM lu_sources)
                   OR market_type NOT IN (SELECT name FROM lu_market_types)
                LIMIT 10
            """)).fetchall()
            
            print("Missing lookup values:")
            for bm, src, mt in missing:
                if bm not in bookmakers:
                    print(f"  Bookmaker: {bm}")
                if src not in sources:
                    print(f"  Source: {src}")
                if mt not in market_types:
                    print(f"  Market type: {mt}")

if __name__ == "__main__":
    test_speed()