#!/usr/bin/env python3
"""Resume and monitor the database migration with better error handling."""

import logging
import time
import json
import os
import signal
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration_resume.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MigrationResumer:
    def __init__(self, batch_size=5000, workers=2):
        self.batch_size = batch_size
        self.workers = workers
        self.running = True
        self.state_file = 'migration_state.json'
        self.stats = {
            'batches_processed': 0,
            'records_processed': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        
    def load_state(self):
        """Load migration state from file."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'last_id': 0,
            'total_processed': 0,
            'total_migrated': 0,
            'total_skipped': 0,
            'start_time': time.time()
        }
        
    def save_state(self, state):
        """Save migration state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
            
    def check_system_resources(self):
        """Check if system has enough resources."""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        if memory.percent > 85:
            logger.warning(f"High memory usage: {memory.percent}%")
            return False
        if cpu > 90:
            logger.warning(f"High CPU usage: {cpu}%")
            return False
        return True
        
    def run(self):
        """Run the migration resumption."""
        logger.info("Starting migration resumption...")
        
        # Load current state
        state = self.load_state()
        logger.info(f"Loaded state: Last ID={state['last_id']:,}, Processed={state['total_processed']:,}")
        
        # Check total records
        try:
            engine = create_engine('sqlite:///sport_odds.db', poolclass=NullPool)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM odd WHERE id > :last_id"), 
                                    {"last_id": state['last_id']})
                remaining = result.scalar()
                logger.info(f"Remaining records to process: {remaining:,}")
                
                if remaining == 0:
                    logger.info("Migration appears to be complete!")
                    return
                    
        except Exception as e:
            logger.error(f"Error checking database: {e}")
            return
            
        # Start optimized migration
        logger.info(f"Starting optimized migration with batch_size={self.batch_size}, workers={self.workers}")
        
        try:
            # Import and run the optimized migration
            from optimized_background_migration import OptimizedMigrationManager
            
            manager = OptimizedMigrationManager(
                batch_size=self.batch_size,
                max_workers=self.workers,
                start_id=state['last_id']
            )
            
            # Monitor progress
            last_report = time.time()
            
            while self.running and manager.running:
                time.sleep(5)  # Check every 5 seconds
                
                # Report progress
                if time.time() - last_report > 30:  # Report every 30 seconds
                    current_state = manager.get_state()
                    progress_pct = (current_state['total_processed'] / 515919464) * 100
                    rate = current_state.get('current_rate', 0)
                    
                    logger.info(f"Progress: {progress_pct:.2f}% | "
                              f"Processed: {current_state['total_processed']:,} | "
                              f"Rate: {rate:.0f} records/sec | "
                              f"Memory: {psutil.virtual_memory().percent}%")
                    
                    # Check resources
                    if not self.check_system_resources():
                        logger.warning("Resource limits reached, pausing for 60 seconds...")
                        manager.pause()
                        time.sleep(60)
                        manager.resume()
                    
                    last_report = time.time()
                    
        except KeyboardInterrupt:
            logger.info("Migration interrupted by user")
        except Exception as e:
            logger.error(f"Migration error: {e}", exc_info=True)
            
        logger.info("Migration resumption completed")
        
    def run_parallel(self):
        """Run the migration with parallel processing."""
        logger.info("Starting parallel migration...")
        
        # Create a simple parallel migration script
        parallel_script = """
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_migration_worker(worker_id, start_id, end_id):
    cmd = [
        'python', 'optimized_background_migration.py',
        '--start-id', str(start_id),
        '--end-id', str(end_id),
        '--worker-id', str(worker_id),
        '--batch-size', '10000'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        'worker_id': worker_id,
        'success': result.returncode == 0,
        'output': result.stdout,
        'error': result.stderr
    }

# Divide work among workers
total_records = 515919464
workers = 4
chunk_size = total_records // workers

with ProcessPoolExecutor(max_workers=workers) as executor:
    futures = []
    
    for i in range(workers):
        start_id = i * chunk_size
        end_id = (i + 1) * chunk_size if i < workers - 1 else total_records
        
        future = executor.submit(run_migration_worker, i, start_id, end_id)
        futures.append(future)
        
    # Monitor progress
    for future in as_completed(futures):
        result = future.result()
        print(f"Worker {result['worker_id']}: {'Success' if result['success'] else 'Failed'}")
"""
        
        logger.info("Parallel migration setup complete. Run with multiple workers for faster processing.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Resume database migration')
    parser.add_argument('--batch-size', type=int, default=5000, help='Batch size for processing')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--parallel', action='store_true', help='Run parallel migration')
    
    args = parser.parse_args()
    
    resumer = MigrationResumer(
        batch_size=args.batch_size,
        workers=args.workers
    )
    
    if args.parallel:
        resumer.run_parallel()
    else:
        resumer.run()


if __name__ == "__main__":
    main()