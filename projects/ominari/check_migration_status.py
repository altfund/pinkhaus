#!/usr/bin/env python3
"""Quick status check for the migration."""

import json
import os
from sqlalchemy import create_engine, text
from datetime import datetime

def check_status():
    # Check if migration is running
    if os.path.exists('migration.pid'):
        with open('migration.pid', 'r') as f:
            pid = f.read().strip()
        
        # Check if process is running
        try:
            os.kill(int(pid), 0)
            print(f"✓ Migration is running (PID: {pid})")
        except OSError:
            print("✗ Migration process not found")
    else:
        print("✗ No migration running")
    
    # Check state file
    if os.path.exists('migration_state.json'):
        with open('migration_state.json', 'r') as f:
            state = json.load(f)
        
        print(f"\nMigration State:")
        print(f"  Last ID processed: {state['last_id']:,}")
        print(f"  Total migrated: {state['total_migrated']:,}")
        print(f"  Total skipped: {state['total_skipped']:,}")
        print(f"  Total processed: {state['total_processed']:,}")
        
        if state['start_time']:
            elapsed = datetime.now().timestamp() - state['start_time']
            rate = state['total_migrated'] / elapsed if elapsed > 0 else 0
            print(f"  Average rate: {rate:.0f} records/sec")
            print(f"  Running for: {elapsed/3600:.1f} hours")
    
    # Quick database check
    try:
        engine = create_engine('sqlite:///sport_odds.db', connect_args={'timeout': 5})
        with engine.connect() as conn:
            # Count normalized records
            result = conn.execute(text("SELECT COUNT(*) FROM odds_normalized"))
            count = result.scalar()
            print(f"\nNormalized table: {count:,} records")
            
            # Get sample
            result = conn.execute(text("""
                SELECT market_id, bookmaker_id, outcome_id, decimal_odds_x1000
                FROM odds_normalized
                ORDER BY updated_at DESC
                LIMIT 3
            """))
            
            print("\nLatest records:")
            for row in result:
                print(f"  {row[0][:16]}... | Bookmaker: {row[1]} | Outcome: {row[2]} | Odds: {row[3]/1000:.3f}")
                
    except Exception as e:
        print(f"\nDatabase check failed: {e}")

if __name__ == "__main__":
    check_status()