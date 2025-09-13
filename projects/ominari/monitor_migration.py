#!/usr/bin/env python3
"""Monitor the migration progress in real-time."""

import time
import os
from datetime import datetime
from sqlalchemy import text
from database_v2 import db_manager
import psutil

def get_migration_stats():
    """Get current migration statistics."""
    with db_manager.get_db_session() as db:
        # Original table count
        orig_count = db.execute(text(
            "SELECT COUNT(*) FROM odd"
        )).scalar()
        
        # Normalized table count  
        norm_count = db.execute(text(
            "SELECT COUNT(*) FROM odds_normalized"
        )).scalar()
        
        # Latest migrated timestamp
        latest = db.execute(text(
            "SELECT MAX(updated_at) FROM odds_normalized"
        )).scalar()
        
        latest_date = datetime.fromtimestamp(latest) if latest else None
        
        # Database file size
        db_size = os.path.getsize('sport_odds.db') / (1024**3)
        
    return {
        'original_count': orig_count,
        'normalized_count': norm_count,
        'latest_date': latest_date,
        'db_size_gb': db_size,
        'progress_pct': (norm_count / orig_count * 100) if orig_count > 0 else 0
    }

def monitor_loop():
    """Monitor migration progress."""
    print("MIGRATION MONITOR")
    print("=" * 80)
    
    start_stats = get_migration_stats()
    start_count = start_stats['normalized_count']
    start_time = time.time()
    
    while True:
        stats = get_migration_stats()
        elapsed = time.time() - start_time
        
        # Calculate rate
        migrated = stats['normalized_count'] - start_count
        rate = migrated / elapsed if elapsed > 0 else 0
        
        # System resources
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        
        # Clear screen and display
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("MIGRATION MONITOR")
        print("=" * 80)
        print(f"Progress: {stats['progress_pct']:.2f}%")
        print(f"Original records: {stats['original_count']:,}")
        print(f"Migrated records: {stats['normalized_count']:,}")
        print(f"Latest date: {stats['latest_date']}")
        print(f"Database size: {stats['db_size_gb']:.1f} GB")
        print()
        print(f"Migration rate: {rate:,.0f} records/sec")
        print(f"Time elapsed: {elapsed/3600:.1f} hours")
        print(f"Est. time remaining: {((stats['original_count'] - stats['normalized_count']) / rate / 3600):.1f} hours" if rate > 0 else "N/A")
        print()
        print("System Resources:")
        print(f"  CPU: {cpu}%")
        print(f"  Memory: {memory.percent}% ({memory.used/(1024**3):.1f}/{memory.total/(1024**3):.1f} GB)")
        print(f"  Disk Read: {disk_io.read_bytes/(1024**3):.1f} GB")
        print(f"  Disk Write: {disk_io.write_bytes/(1024**3):.1f} GB")
        print()
        print("Press Ctrl+C to exit")
        
        time.sleep(5)  # Update every 5 seconds

if __name__ == "__main__":
    try:
        monitor_loop()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")