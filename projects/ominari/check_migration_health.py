#!/usr/bin/env python3
"""
Check Migration Health Status

Determines if the migration is running, stuck, or stopped.
"""

import os
import sqlite3
import json
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_migration_status():
    """Check comprehensive migration status."""
    
    logger.info("🔍 Checking Migration Health Status...")
    logger.info("=" * 60)
    
    status = {
        'is_running': False,
        'last_activity': None,
        'progress': 0.0,
        'errors': [],
        'recommendation': None
    }
    
    # 1. Check PID file
    pid_file = 'migration.pid'
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as f:
            pid = f.read().strip()
        
        # Check if process is running
        try:
            os.kill(int(pid), 0)  # Signal 0 = check if process exists
            status['is_running'] = True
            logger.info(f"✅ Migration process {pid} is RUNNING")
        except ProcessLookupError:
            status['is_running'] = False
            logger.info(f"❌ Migration process {pid} is NOT RUNNING")
        except ValueError:
            logger.error(f"Invalid PID in file: {pid}")
    
    # 2. Check log files for recent activity
    log_files = [
        'migration_output.log',
        'migration_background.log', 
        'optimized_migration.log',
        'simple_migration.log'
    ]
    
    latest_timestamp = None
    latest_log = None
    
    for log_file in log_files:
        if os.path.exists(log_file):
            mtime = os.path.getmtime(log_file)
            mod_time = datetime.fromtimestamp(mtime, timezone.utc)
            
            if latest_timestamp is None or mtime > latest_timestamp:
                latest_timestamp = mtime
                latest_log = log_file
            
            logger.info(f"   {log_file}: Last modified {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if latest_timestamp:
        time_since_update = (datetime.now(timezone.utc) - 
                            datetime.fromtimestamp(latest_timestamp, timezone.utc))
        hours_since = time_since_update.total_seconds() / 3600
        
        status['last_activity'] = {
            'file': latest_log,
            'timestamp': datetime.fromtimestamp(latest_timestamp, timezone.utc).isoformat(),
            'hours_ago': round(hours_since, 1)
        }
        
        logger.info(f"\n⏰ Last activity: {hours_since:.1f} hours ago in {latest_log}")
    
    # 3. Check database for migration progress
    try:
        conn = sqlite3.connect('sport_odds.db')
        cursor = conn.cursor()
        
        # Check if normalized tables exist
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name LIKE '%_normalized'
        """)
        normalized_tables = cursor.fetchall()
        
        if normalized_tables:
            # Check row counts
            for table in normalized_tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                logger.info(f"   {table_name}: {count:,} rows")
                
                if table_name == 'odds_normalized':
                    # Calculate progress
                    cursor.execute("SELECT COUNT(*) FROM odd")
                    total_odds = cursor.fetchone()[0]
                    progress = (count / total_odds) * 100 if total_odds > 0 else 0
                    status['progress'] = progress
                    logger.info(f"\n📊 Migration Progress: {progress:.2f}% ({count:,}/{total_odds:,})")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Database check error: {e}")
        status['errors'].append(str(e))
    
    # 4. Analyze errors in logs
    if latest_log and os.path.exists(latest_log):
        try:
            # Check last 100 lines for errors
            with open(latest_log, 'r') as f:
                lines = f.readlines()[-100:]
            
            error_count = sum(1 for line in lines if 'ERROR' in line)
            if error_count > 0:
                logger.warning(f"\n⚠️ Found {error_count} errors in recent logs")
                status['errors'].append(f"{error_count} errors in {latest_log}")
                
                # Look for specific error patterns
                unique_errors = set()
                for line in lines:
                    if 'ERROR' in line:
                        if 'UNIQUE constraint failed' in line:
                            unique_errors.add('Duplicate data conflicts')
                        elif 'locked' in line:
                            unique_errors.add('Database lock issues')
                        elif 'memory' in line.lower():
                            unique_errors.add('Memory issues')
                
                for error in unique_errors:
                    logger.error(f"   - {error}")
                    
        except Exception as e:
            logger.error(f"Log analysis error: {e}")
    
    # 5. Make recommendation
    if not status['is_running']:
        if status['last_activity'] and status['last_activity']['hours_ago'] < 1:
            status['recommendation'] = "RECENTLY_STOPPED"
            logger.warning("\n🔴 Migration STOPPED recently (within last hour)")
        else:
            status['recommendation'] = "STOPPED"
            logger.error("\n🔴 Migration is STOPPED")
    else:
        if status['errors']:
            status['recommendation'] = "RUNNING_WITH_ERRORS"
            logger.warning("\n🟡 Migration RUNNING but encountering errors")
        else:
            status['recommendation'] = "RUNNING_OK"
            logger.info("\n🟢 Migration RUNNING normally")
    
    # 6. Final summary
    logger.info("\n📋 SUMMARY:")
    logger.info(f"   Status: {'RUNNING' if status['is_running'] else 'STOPPED'}")
    logger.info(f"   Progress: {status['progress']:.2f}%")
    
    if status['last_activity']:
        logger.info(f"   Last Activity: {status['last_activity']['hours_ago']} hours ago")
    
    if status['errors']:
        logger.info(f"   Errors: {len(status['errors'])}")
    
    # Save status report
    with open('migration_health_status.json', 'w') as f:
        json.dump(status, f, indent=2)
    
    return status


def restart_migration_if_needed(status):
    """Provide instructions to restart migration if needed."""
    
    if status['recommendation'] in ['STOPPED', 'RECENTLY_STOPPED']:
        logger.info("\n🔧 RESTART INSTRUCTIONS:")
        logger.info("To restart the migration, run one of these commands:")
        logger.info("")
        logger.info("1. Safe background migration (recommended):")
        logger.info("   python background_migration.py &")
        logger.info("")
        logger.info("2. Optimized migration with monitoring:")
        logger.info("   python optimized_background_migration.py &")
        logger.info("")
        logger.info("3. Simple incremental migration:")
        logger.info("   python simple_background_migration.py &")
        logger.info("")
        logger.info("⚠️ However, given the 0.18% progress after significant time,")
        logger.info("   consider using the HYBRID approach instead:")
        logger.info("   python hybrid_database_setup.py")


if __name__ == "__main__":
    status = check_migration_status()
    restart_migration_if_needed(status)
    
    print("\n" + "="*60)
    if status['is_running']:
        print("✅ Migration is RUNNING")
        if status['progress'] > 0:
            # Estimate completion time
            # Assuming 0.18% took ~2 days
            days_per_percent = 2 / 0.18
            days_remaining = (100 - status['progress']) * days_per_percent
            print(f"⏳ Estimated completion: {days_remaining:.0f} days")
    else:
        print("❌ Migration is STOPPED")
        print("🚨 The migration process has stopped and needs attention!")