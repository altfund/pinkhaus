#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Background database indexer that creates indexes incrementally without blocking.
Designed to handle very large databases by creating indexes in the background.
"""

import sqlite3
import time
import logging
import json
import os
import subprocess
from datetime import datetime

DB_NAME = "sport_odds.db"
STATE_FILE = "indexing_state.json"
LOG_FILE = "background_indexing.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BackgroundIndexer:
    """Manages background index creation with state persistence."""
    
    def __init__(self, db_path=DB_NAME):
        self.db_path = db_path
        self.state_file = STATE_FILE
        self.state = self.load_state()
        self.indexes = [
            # Most critical indexes first
            ("idx_odd_updated_at", "odd", "(updated_at)",
             "Time-based queries - CRITICAL for backtest"),
             
            ("idx_market_source", "market", "(source_id)",
             "Primary market lookup"),
             
            ("idx_odd_source_updated", "odd", "(source_id, updated_at DESC)",
             "Latest odds per market - may take hours"),
             
            ("idx_market_maturity", "market", "(maturity_date)",
             "Finding closed markets"),
             
            ("idx_odd_rowid_desc", "odd", "(rowid DESC)",
             "Fast recent data access"),
             
            # Less critical, can be added later
            ("idx_odd_updated_source_outcome", "odd", "(updated_at, source_id, outcome)",
             "Composite for snapshots - optional"),
        ]
        
    def load_state(self):
        """Load indexing state from file."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'completed': [],
            'in_progress': None,
            'failed': {},
            'start_time': None
        }
    
    def save_state(self):
        """Save current state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def check_existing_indexes(self):
        """Get list of existing indexes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
        """)
        existing = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        return existing
    
    def create_index_background(self, idx_name, table, columns):
        """Create index using a separate SQLite process."""
        logger.info(f"Starting background creation of {idx_name}")
        
        # Create SQL script
        sql_script = f"""
-- Background index creation for {idx_name}
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=50000;
PRAGMA temp_store=MEMORY;

-- Create the index
CREATE INDEX IF NOT EXISTS {idx_name} ON {table}{columns};

-- Analyze to update statistics
ANALYZE {table};
"""
        
        script_file = f"create_{idx_name}.sql"
        with open(script_file, 'w') as f:
            f.write(sql_script)
        
        # Run in background using sqlite3 command
        try:
            # Use nice to lower priority
            cmd = [
                'nice', '-n', '19',  # Lowest priority
                'sqlite3', self.db_path,
                f'.read {script_file}'
            ]
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            # Run with timeout of 4 hours per index
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=4 * 3600  # 4 hours
            )
            
            # Clean up script file
            os.remove(script_file)
            
            if result.returncode == 0:
                logger.info(f"Successfully created {idx_name}")
                return True
            else:
                logger.error(f"Failed to create {idx_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout creating {idx_name} after 4 hours")
            return False
        except Exception as e:
            logger.error(f"Error creating {idx_name}: {e}")
            return False
        finally:
            # Clean up script file if it exists
            if os.path.exists(script_file):
                os.remove(script_file)
    
    def run_incremental(self):
        """Run indexing incrementally, saving state between indexes."""
        logger.info("Starting incremental background indexing")
        logger.info(f"Database: {self.db_path} ({self.get_db_size()})")
        
        # Check what already exists
        existing = self.check_existing_indexes()
        logger.info(f"Found {len(existing)} existing indexes")
        
        # Update completed list
        for idx_name, _, _, _ in self.indexes:
            if idx_name in existing and idx_name not in self.state['completed']:
                self.state['completed'].append(idx_name)
                logger.info(f"  {idx_name} already exists")
        
        self.save_state()
        
        # Process remaining indexes
        for idx_name, table, columns, description in self.indexes:
            if idx_name in self.state['completed']:
                continue
                
            if idx_name in self.state.get('failed', {}):
                logger.info(f"Skipping {idx_name} (previously failed)")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Creating {idx_name}: {description}")
            logger.info("This may take a while for large tables...")
            
            self.state['in_progress'] = idx_name
            self.save_state()
            
            start_time = time.time()
            success = self.create_index_background(idx_name, table, columns)
            elapsed = time.time() - start_time
            
            if success:
                self.state['completed'].append(idx_name)
                self.state['in_progress'] = None
                logger.info(f"Completed {idx_name} in {elapsed/60:.1f} minutes")
            else:
                self.state['failed'][idx_name] = {
                    'time': datetime.now().isoformat(),
                    'duration': elapsed
                }
                self.state['in_progress'] = None
                logger.error(f"Failed {idx_name} after {elapsed/60:.1f} minutes")
            
            self.save_state()
            
            # Brief pause between indexes
            time.sleep(5)
        
        logger.info(f"\n{'='*60}")
        logger.info("Indexing complete!")
        logger.info(f"Successful: {len(self.state['completed'])}")
        logger.info(f"Failed: {len(self.state.get('failed', {}))}")
        
        # Run final analyze
        self.run_analyze()
    
    def run_analyze(self):
        """Run ANALYZE on the database."""
        logger.info("\nRunning ANALYZE to update statistics...")
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("ANALYZE")
            conn.close()
            logger.info("ANALYZE completed")
        except Exception as e:
            logger.error(f"ANALYZE failed: {e}")
    
    def get_db_size(self):
        """Get human-readable database size."""
        size = os.path.getsize(self.db_path)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
    
    def status(self):
        """Print current indexing status."""
        logger.info("\nIndexing Status:")
        logger.info(f"Completed: {len(self.state['completed'])}")
        for idx in self.state['completed']:
            logger.info(f"  ✓ {idx}")
        
        if self.state.get('in_progress'):
            logger.info(f"\nIn Progress: {self.state['in_progress']}")
        
        failed = self.state.get('failed', {})
        if failed:
            logger.info(f"\nFailed: {len(failed)}")
            for idx, info in failed.items():
                logger.info(f"  ✗ {idx} (at {info['time']})")
        
        # Check remaining
        remaining = []
        for idx_name, _, _, _ in self.indexes:
            if (idx_name not in self.state['completed'] and 
                idx_name not in failed and
                idx_name != self.state.get('in_progress')):
                remaining.append(idx_name)
        
        if remaining:
            logger.info(f"\nRemaining: {len(remaining)}")
            for idx in remaining:
                logger.info(f"  - {idx}")


def run_as_daemon():
    """Run indexer as a background daemon process."""
    import signal
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)
    
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("Starting background indexer daemon...")
    
    # Run indexer
    indexer = BackgroundIndexer()
    indexer.run_incremental()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Background database indexer for large SQLite databases"
    )
    parser.add_argument(
        "--daemon", "-d", action="store_true",
        help="Run as background daemon"
    )
    parser.add_argument(
        "--status", "-s", action="store_true",
        help="Show current indexing status"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Reset indexing state and start over"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Only create the most essential indexes"
    )
    
    args = parser.parse_args()
    
    indexer = BackgroundIndexer()
    
    if args.status:
        indexer.status()
    elif args.reset:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
            logger.info("Reset indexing state")
        indexer.run_incremental()
    elif args.quick:
        # Only create the most critical indexes
        indexer.indexes = indexer.indexes[:3]
        indexer.run_incremental()
    elif args.daemon:
        run_as_daemon()
    else:
        # Run in foreground
        indexer.run_incremental()


if __name__ == "__main__":
    main()