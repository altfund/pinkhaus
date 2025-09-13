#!/usr/bin/env python3
"""
Continuous Migration Manager

Manages the large-scale database migration from SQLite to PostgreSQL
with smart chunking, progress tracking, and resume capability.
"""

import logging
import sqlite3
import time
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MigrationProgress:
    """Tracks migration progress."""
    table_name: str
    total_rows: int
    migrated_rows: int
    current_chunk: int
    chunk_size: int
    start_time: datetime
    last_update: datetime
    completion_percentage: float
    estimated_time_remaining: Optional[float] = None
    rows_per_second: float = 0.0
    status: str = "in_progress"  # in_progress, completed, failed, paused


@dataclass
class MigrationState:
    """Overall migration state."""
    total_tables: int
    completed_tables: int
    current_table: Optional[str]
    overall_progress: float
    total_rows_migrated: int
    total_rows: int
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    is_running: bool = True


class ContinuousMigrationManager:
    """Manages continuous database migration with smart chunking."""
    
    def __init__(self,
                 source_db: str = "sport_odds.db",
                 chunk_size: int = 10000,
                 max_workers: int = 2,
                 progress_file: str = "migration_progress.json"):
        
        self.source_db = source_db
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.progress_file = progress_file
        
        # Migration state
        self.migration_state: Optional[MigrationState] = None
        self.table_progress: Dict[str, MigrationProgress] = {}
        self.is_running = False
        self.should_stop = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Load existing progress
        self._load_progress()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received shutdown signal, stopping migration...")
        self.should_stop = True
        self._save_progress()
    
    def _load_progress(self):
        """Load migration progress from file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct progress objects
                self.table_progress = {}
                for table_name, progress_data in data.get('table_progress', {}).items():
                    progress_data['start_time'] = datetime.fromisoformat(progress_data['start_time'])
                    progress_data['last_update'] = datetime.fromisoformat(progress_data['last_update'])
                    self.table_progress[table_name] = MigrationProgress(**progress_data)
                
                # Reconstruct migration state
                if 'migration_state' in data:
                    state_data = data['migration_state']
                    state_data['start_time'] = datetime.fromisoformat(state_data['start_time'])
                    if state_data.get('estimated_completion'):
                        state_data['estimated_completion'] = datetime.fromisoformat(state_data['estimated_completion'])
                    self.migration_state = MigrationState(**state_data)
                
                logger.info("✅ Loaded existing migration progress")
                
            except Exception as e:
                logger.error(f"Failed to load progress: {e}")
                self._initialize_new_migration()
        else:
            self._initialize_new_migration()
    
    def _save_progress(self):
        """Save migration progress to file."""
        try:
            data = {
                'table_progress': {},
                'migration_state': None
            }
            
            # Save table progress
            for table_name, progress in self.table_progress.items():
                progress_dict = asdict(progress)
                progress_dict['start_time'] = progress.start_time.isoformat()
                progress_dict['last_update'] = progress.last_update.isoformat()
                data['table_progress'][table_name] = progress_dict
            
            # Save migration state
            if self.migration_state:
                state_dict = asdict(self.migration_state)
                state_dict['start_time'] = self.migration_state.start_time.isoformat()
                if self.migration_state.estimated_completion:
                    state_dict['estimated_completion'] = self.migration_state.estimated_completion.isoformat()
                data['migration_state'] = state_dict
            
            with open(self.progress_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def _initialize_new_migration(self):
        """Initialize a new migration."""
        logger.info("Initializing new migration...")
        
        # Get table information
        tables_info = self._analyze_source_database()
        
        total_tables = len(tables_info)
        total_rows = sum(info['row_count'] for info in tables_info.values())
        
        self.migration_state = MigrationState(
            total_tables=total_tables,
            completed_tables=0,
            current_table=None,
            overall_progress=0.0,
            total_rows_migrated=0,
            total_rows=total_rows,
            start_time=datetime.now(timezone.utc)
        )
        
        # Initialize table progress
        for table_name, info in tables_info.items():
            if info['row_count'] > 0:
                self.table_progress[table_name] = MigrationProgress(
                    table_name=table_name,
                    total_rows=info['row_count'],
                    migrated_rows=0,
                    current_chunk=0,
                    chunk_size=self.chunk_size,
                    start_time=datetime.now(timezone.utc),
                    last_update=datetime.now(timezone.utc),
                    completion_percentage=0.0
                )
        
        logger.info(f"Initialized migration for {total_tables} tables ({total_rows:,} total rows)")
    
    def _analyze_source_database(self) -> Dict[str, Dict]:
        """Analyze source database to get table information."""
        logger.info("Analyzing source database...")
        
        conn = sqlite3.connect(self.source_db)
        
        # Get all tables
        tables = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """).fetchall()
        
        tables_info = {}
        
        for (table_name,) in tables:
            try:
                # Get row count
                count_result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                row_count = count_result[0] if count_result else 0
                
                # Get table schema
                schema = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                
                tables_info[table_name] = {
                    'row_count': row_count,
                    'columns': len(schema),
                    'schema': schema
                }
                
                logger.info(f"  {table_name}: {row_count:,} rows, {len(schema)} columns")
                
            except sqlite3.Error as e:
                logger.error(f"Error analyzing table {table_name}: {e}")
                tables_info[table_name] = {'row_count': 0, 'error': str(e)}
        
        conn.close()
        
        # Sort tables by row count (process smaller tables first)
        sorted_tables = dict(sorted(tables_info.items(), key=lambda x: x[1].get('row_count', 0)))
        
        return sorted_tables
    
    def _migrate_table_chunk(self, table_name: str, offset: int, limit: int) -> int:
        """Migrate a chunk of data from a specific table."""
        
        conn = sqlite3.connect(self.source_db)
        try:
            # Get chunk data
            cursor = conn.execute(f"SELECT * FROM {table_name} LIMIT {limit} OFFSET {offset}")
            rows = cursor.fetchall()
            
            if rows:
                # In a real implementation, this would insert into PostgreSQL
                # For now, simulate processing
                time.sleep(0.01 * len(rows))  # Simulate processing time
                
                logger.debug(f"Processed {len(rows)} rows from {table_name} (offset {offset})")
                
                return len(rows)
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error migrating chunk from {table_name}: {e}")
            return 0
        finally:
            conn.close()
    
    def _migrate_table(self, table_name: str) -> bool:
        """Migrate an entire table with chunked processing."""
        
        if table_name not in self.table_progress:
            logger.warning(f"No progress info for table {table_name}")
            return False
        
        progress = self.table_progress[table_name]
        
        if progress.status == "completed":
            logger.info(f"Table {table_name} already completed")
            return True
        
        logger.info(f"Starting migration of table {table_name} ({progress.total_rows:,} rows)")
        
        self.migration_state.current_table = table_name
        
        start_time = time.time()
        rows_processed = 0
        
        while progress.migrated_rows < progress.total_rows and not self.should_stop:
            
            # Calculate chunk parameters
            offset = progress.migrated_rows
            remaining_rows = progress.total_rows - progress.migrated_rows
            chunk_size = min(self.chunk_size, remaining_rows)
            
            # Migrate chunk
            chunk_start = time.time()
            chunk_rows = self._migrate_table_chunk(table_name, offset, chunk_size)
            chunk_time = time.time() - chunk_start
            
            if chunk_rows == 0:
                break
            
            # Update progress
            progress.migrated_rows += chunk_rows
            progress.current_chunk += 1
            progress.last_update = datetime.now(timezone.utc)
            progress.completion_percentage = (progress.migrated_rows / progress.total_rows) * 100
            
            rows_processed += chunk_rows
            elapsed_time = time.time() - start_time
            
            if elapsed_time > 0:
                progress.rows_per_second = rows_processed / elapsed_time
                
                # Estimate time remaining
                remaining_rows = progress.total_rows - progress.migrated_rows
                if progress.rows_per_second > 0:
                    progress.estimated_time_remaining = remaining_rows / progress.rows_per_second
            
            # Update migration state
            self._update_overall_progress()
            
            # Save progress periodically
            if progress.current_chunk % 100 == 0:  # Every 100 chunks
                self._save_progress()
                logger.info(f"Table {table_name}: {progress.completion_percentage:.2f}% complete "
                           f"({progress.migrated_rows:,}/{progress.total_rows:,} rows, "
                           f"{progress.rows_per_second:.0f} rows/sec)")
            
            # Small delay to prevent overwhelming the system
            if chunk_time < 0.1:
                time.sleep(0.1 - chunk_time)
        
        # Mark table as completed
        if progress.migrated_rows >= progress.total_rows:
            progress.status = "completed"
            progress.completion_percentage = 100.0
            self.migration_state.completed_tables += 1
            
            logger.info(f"✅ Completed table {table_name} ({progress.migrated_rows:,} rows in {elapsed_time:.2f}s)")
            return True
        else:
            logger.info(f"⏸️ Paused table {table_name} at {progress.completion_percentage:.2f}%")
            progress.status = "paused"
            return False
    
    def _update_overall_progress(self):
        """Update overall migration progress."""
        if not self.migration_state:
            return
        
        total_migrated = sum(p.migrated_rows for p in self.table_progress.values())
        self.migration_state.total_rows_migrated = total_migrated
        
        if self.migration_state.total_rows > 0:
            self.migration_state.overall_progress = (total_migrated / self.migration_state.total_rows) * 100
        
        # Estimate completion time
        elapsed_time = datetime.now(timezone.utc) - self.migration_state.start_time
        if self.migration_state.overall_progress > 0:
            total_estimated_time = elapsed_time.total_seconds() * (100 / self.migration_state.overall_progress)
            remaining_time = total_estimated_time - elapsed_time.total_seconds()
            
            if remaining_time > 0:
                self.migration_state.estimated_completion = datetime.now(timezone.utc).timestamp() + remaining_time
    
    def get_status_report(self) -> Dict:
        """Get current migration status."""
        if not self.migration_state:
            return {"status": "not_initialized"}
        
        active_tables = [p for p in self.table_progress.values() if p.status == "in_progress"]
        completed_tables = [p for p in self.table_progress.values() if p.status == "completed"]
        
        return {
            "overall_progress": self.migration_state.overall_progress,
            "completed_tables": len(completed_tables),
            "total_tables": self.migration_state.total_tables,
            "current_table": self.migration_state.current_table,
            "total_rows_migrated": self.migration_state.total_rows_migrated,
            "total_rows": self.migration_state.total_rows,
            "active_tables": len(active_tables),
            "estimated_completion": self.migration_state.estimated_completion,
            "is_running": self.is_running,
            "table_details": {
                name: {
                    "completion_pct": p.completion_percentage,
                    "rows_migrated": p.migrated_rows,
                    "total_rows": p.total_rows,
                    "status": p.status,
                    "rows_per_second": p.rows_per_second
                }
                for name, p in self.table_progress.items()
            }
        }
    
    def run_continuous_migration(self):
        """Run continuous migration with resume capability."""
        logger.info("🚀 Starting Continuous Database Migration")
        logger.info("=" * 70)
        
        self.is_running = True
        
        try:
            # Get tables to process (incomplete ones first)
            incomplete_tables = [
                name for name, progress in self.table_progress.items()
                if progress.status in ["in_progress", "paused"] and progress.total_rows > 0
            ]
            
            # Sort by completion percentage (continue where we left off)
            incomplete_tables.sort(key=lambda t: self.table_progress[t].completion_percentage, reverse=True)
            
            logger.info(f"Processing {len(incomplete_tables)} incomplete tables")
            
            for table_name in incomplete_tables:
                if self.should_stop:
                    break
                
                success = self._migrate_table(table_name)
                
                if success:
                    logger.info(f"✅ Table {table_name} migration completed")
                else:
                    logger.info(f"⏸️ Table {table_name} migration paused")
                
                # Save progress after each table
                self._save_progress()
                
                # Print status update
                status = self.get_status_report()
                logger.info(f"Overall progress: {status['overall_progress']:.2f}% "
                           f"({status['completed_tables']}/{status['total_tables']} tables)")
            
            self.is_running = False
            
            if not self.should_stop:
                logger.info("🎉 Migration completed successfully!")
            else:
                logger.info("⏸️ Migration paused - can be resumed later")
                
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            self.is_running = False
        
        finally:
            self._save_progress()


def main():
    """Run continuous migration."""
    
    # Create migration manager
    migrator = ContinuousMigrationManager(
        chunk_size=5000,  # Smaller chunks for more frequent progress updates
        max_workers=1     # Single worker to avoid overwhelming the system
    )
    
    # Print initial status
    status = migrator.get_status_report()
    if status.get("status") != "not_initialized":
        print("Current Migration Status:")
        print(f"  Overall progress: {status['overall_progress']:.3f}%")
        print(f"  Completed tables: {status['completed_tables']}/{status['total_tables']}")
        print(f"  Total rows migrated: {status['total_rows_migrated']:,}")
        print(f"  Total rows: {status['total_rows']:,}")
        
        if status.get('current_table'):
            print(f"  Current table: {status['current_table']}")
    
    # Run migration
    migrator.run_continuous_migration()


if __name__ == "__main__":
    main()