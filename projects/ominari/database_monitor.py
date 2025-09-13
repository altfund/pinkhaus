#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database monitoring and maintenance utilities.
"""

import os
import time
import sqlite3
import logging
from datetime import datetime
import psutil
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseMonitor:
    """Monitor and maintain the SQLite database."""
    
    def __init__(self, db_path: str = "sport_odds.db"):
        self.db_path = db_path
        self.wal_path = f"{db_path}-wal"
        self.shm_path = f"{db_path}-shm"
        self.running = True
        
    def check_database_health(self) -> dict:
        """Check various database health metrics."""
        health = {
            'timestamp': datetime.now().isoformat(),
            'database_exists': os.path.exists(self.db_path),
            'database_size': 0,
            'wal_size': 0,
            'shm_size': 0,
            'total_size': 0,
            'locked': False,
            'wal_mode': False,
            'page_size': 0,
            'page_count': 0,
            'connections': 0,
            'issues': []
        }
        
        if not health['database_exists']:
            health['issues'].append("Database file not found")
            return health
        
        # File sizes
        health['database_size'] = os.path.getsize(self.db_path)
        if os.path.exists(self.wal_path):
            health['wal_size'] = os.path.getsize(self.wal_path)
        if os.path.exists(self.shm_path):
            health['shm_size'] = os.path.getsize(self.shm_path)
        health['total_size'] = health['database_size'] + health['wal_size'] + health['shm_size']
        
        # Check if database is locked
        try:
            conn = sqlite3.connect(self.db_path, timeout=1.0)
            cursor = conn.cursor()
            
            # Get basic info
            cursor.execute("PRAGMA journal_mode;")
            health['wal_mode'] = cursor.fetchone()[0] == 'wal'
            
            cursor.execute("PRAGMA page_size;")
            health['page_size'] = cursor.fetchone()[0]
            
            cursor.execute("PRAGMA page_count;")
            health['page_count'] = cursor.fetchone()[0]
            
            # Check for locks
            cursor.execute("PRAGMA lock_status;")
            locks = cursor.fetchall()
            health['locks'] = locks
            
            conn.close()
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                health['locked'] = True
                health['issues'].append("Database is locked")
        except Exception as e:
            health['issues'].append(f"Error checking database: {e}")
        
        # Count connections (Linux specific)
        try:
            connections = 0
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    for f in proc.open_files():
                        if self.db_path in f.path:
                            connections += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            health['connections'] = connections
        except Exception:
            pass
        
        # Check WAL size warnings
        if health['wal_size'] > 1_000_000_000:  # 1GB
            health['issues'].append(f"WAL file very large: {health['wal_size'] / 1e9:.1f}GB")
        
        if health['database_size'] > 100_000_000_000:  # 100GB
            health['issues'].append(f"Database very large: {health['database_size'] / 1e9:.1f}GB")
        
        return health
    
    def force_wal_checkpoint(self) -> bool:
        """Force a WAL checkpoint."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()
            
            logger.info("Forcing WAL checkpoint...")
            cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                logger.info(f"Checkpoint result: busy={result[0]}, log={result[1]}, checkpointed={result[2]}")
                return result[0] == 0  # Not busy
            return False
            
        except Exception as e:
            logger.error(f"Failed to checkpoint: {e}")
            return False
    
    def kill_blocking_processes(self):
        """Kill processes that may be blocking the database."""
        killed = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'ominari' in cmdline and 'database_monitor' not in cmdline:
                        for f in proc.open_files():
                            if self.db_path in f.path:
                                logger.warning(f"Killing process {proc.pid}: {proc.info['name']}")
                                proc.kill()
                                killed.append(proc.pid)
                                break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            logger.error(f"Error killing processes: {e}")
        
        return killed
    
    def optimize_database(self):
        """Run database optimization."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=60.0)
            cursor = conn.cursor()
            
            logger.info("Running database optimization...")
            cursor.execute("PRAGMA optimize;")
            cursor.execute("ANALYZE;")
            
            # Rebuild indexes if needed
            cursor.execute("PRAGMA integrity_check;")
            integrity = cursor.fetchall()
            if integrity[0][0] != 'ok':
                logger.warning("Integrity check failed, rebuilding indexes...")
                cursor.execute("REINDEX;")
            
            conn.close()
            logger.info("Optimization complete")
            
        except Exception as e:
            logger.error(f"Failed to optimize: {e}")
    
    def monitor_loop(self, interval: int = 60):
        """Main monitoring loop."""
        logger.info(f"Starting database monitor (checking every {interval}s)")
        
        def signal_handler(sig, frame):
            logger.info("Received interrupt signal, shutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        while self.running:
            try:
                health = self.check_database_health()
                
                # Log status
                logger.info(f"DB: {health['database_size']/1e9:.1f}GB, "
                           f"WAL: {health['wal_size']/1e6:.1f}MB, "
                           f"Connections: {health['connections']}, "
                           f"Locked: {health['locked']}")
                
                if health['issues']:
                    logger.warning(f"Issues: {', '.join(health['issues'])}")
                
                # Take action if needed
                if health['locked'] and health['connections'] == 0:
                    logger.warning("Database locked with no connections, attempting recovery...")
                    self.kill_blocking_processes()
                    time.sleep(5)
                    self.force_wal_checkpoint()
                
                elif health['wal_size'] > 500_000_000:  # 500MB
                    logger.info("Large WAL file, attempting checkpoint...")
                    if not self.force_wal_checkpoint():
                        logger.warning("Checkpoint failed, database may be busy")
                
                # Periodic optimization (once per day)
                current_hour = datetime.now().hour
                if current_hour == 3:  # 3 AM
                    self.optimize_database()
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            # Sleep with interrupt checking
            for _ in range(interval):
                if not self.running:
                    break
                time.sleep(1)
        
        logger.info("Database monitor stopped")
    
    def print_report(self):
        """Print a detailed database report."""
        health = self.check_database_health()
        
        print("\n" + "="*60)
        print("DATABASE HEALTH REPORT")
        print("="*60)
        print(f"Timestamp: {health['timestamp']}")
        print("\nFile Sizes:")
        print(f"  Database: {health['database_size']/1e9:.2f} GB")
        print(f"  WAL:      {health['wal_size']/1e6:.2f} MB")
        print(f"  SHM:      {health['shm_size']/1e3:.2f} KB")
        print(f"  Total:    {health['total_size']/1e9:.2f} GB")
        print("\nDatabase Info:")
        print(f"  WAL Mode:    {health['wal_mode']}")
        print(f"  Page Size:   {health['page_size']} bytes")
        print(f"  Page Count:  {health['page_count']:,}")
        print(f"  Connections: {health['connections']}")
        print(f"  Locked:      {health['locked']}")
        
        if health['issues']:
            print("\nIssues:")
            for issue in health['issues']:
                print(f"  - {issue}")
        else:
            print("\nNo issues detected")
        
        print("="*60 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database monitoring utility")
    parser.add_argument('command', choices=['monitor', 'report', 'checkpoint', 'optimize', 'unlock'],
                       help="Command to execute")
    parser.add_argument('--interval', type=int, default=60,
                       help="Monitoring interval in seconds (default: 60)")
    parser.add_argument('--force', action='store_true',
                       help="Force operation even if risky")
    
    args = parser.parse_args()
    
    monitor = DatabaseMonitor()
    
    if args.command == 'monitor':
        monitor.monitor_loop(args.interval)
    elif args.command == 'report':
        monitor.print_report()
    elif args.command == 'checkpoint':
        success = monitor.force_wal_checkpoint()
        sys.exit(0 if success else 1)
    elif args.command == 'optimize':
        monitor.optimize_database()
    elif args.command == 'unlock':
        if args.force:
            killed = monitor.kill_blocking_processes()
            print(f"Killed {len(killed)} processes")
        else:
            print("Use --force to kill blocking processes")
            sys.exit(1)


if __name__ == "__main__":
    main()