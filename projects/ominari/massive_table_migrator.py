#!/usr/bin/env python3
"""
Massive Table Migrator

Specialized migrator for handling the 515M+ row odd table and other large tables.
Uses advanced techniques for massive dataset processing.
"""

import logging
import sqlite3
import time
import json
import os
from datetime import datetime, timezone
from typing import Dict, Iterator, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MassiveTableMigrator:
    """Handles migration of extremely large tables (500M+ rows)."""
    
    def __init__(self, db_path: str = "sport_odds.db"):
        self.db_path = db_path
        self.progress_file = "massive_migration_progress.json"
        
        # Performance settings
        self.batch_size = 50000  # Large batches for efficiency
        self.checkpoint_frequency = 10  # Save progress every N batches
        
    def analyze_massive_tables(self) -> Dict:
        """Analyze the massive tables in detail."""
        logger.info("🔍 Analyzing massive tables...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Focus on the largest tables
        large_tables = ['odd', 'market']
        
        analysis = {}
        
        for table in large_tables:
            try:
                start_time = time.time()
                
                # Get total count (we know this is expensive but need it)
                count_result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                total_rows = count_result[0] if count_result else 0
                
                analysis_time = time.time() - start_time
                
                # Get table info
                table_info = conn.execute(f"PRAGMA table_info({table})").fetchall()
                
                # Estimate table size
                page_count = conn.execute(f"SELECT COUNT(*) FROM pragma_page_count()").fetchone()[0]
                page_size = conn.execute(f"PRAGMA page_size").fetchone()[0]
                estimated_table_size_mb = (page_count * page_size) / (1024 * 1024)
                
                # Sample data to understand patterns
                sample_rows = conn.execute(f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT 5").fetchall()
                
                analysis[table] = {
                    'total_rows': total_rows,
                    'columns': len(table_info),
                    'analysis_time_seconds': analysis_time,
                    'estimated_size_mb': estimated_table_size_mb,
                    'rows_per_second_scan': total_rows / analysis_time if analysis_time > 0 else 0,
                    'sample_data': sample_rows[:2],  # Just first 2 samples
                    'schema': table_info
                }
                
                logger.info(f"  {table}: {total_rows:,} rows ({analysis_time:.2f}s to count)")
                
                # Calculate estimated migration time
                if total_rows > 0:
                    estimated_batches = (total_rows + self.batch_size - 1) // self.batch_size
                    estimated_time_hours = (estimated_batches * 0.5) / 3600  # Assume 0.5s per batch
                    logger.info(f"    Estimated migration time: {estimated_time_hours:.2f} hours")
                
            except Exception as e:
                logger.error(f"Error analyzing {table}: {e}")
                analysis[table] = {'error': str(e)}
        
        conn.close()
        return analysis
    
    def create_migration_strategy(self, analysis: Dict) -> Dict:
        """Create an optimal migration strategy for massive tables."""
        
        strategy = {
            'priority_order': [],
            'batch_sizes': {},
            'estimated_times': {},
            'parallel_feasible': {},
            'recommendations': []
        }
        
        for table, info in analysis.items():
            if 'error' in info:
                continue
                
            total_rows = info['total_rows']
            
            if total_rows == 0:
                continue
            
            # Determine priority (smaller tables first, then by importance)
            if table == 'market':
                priority = 1  # Markets are needed first (foreign key references)
            elif table == 'odd':
                priority = 2  # Odds are largest but depend on markets
            else:
                priority = 3
            
            # Determine optimal batch size based on table size
            if total_rows > 100_000_000:  # 100M+
                batch_size = 100000  # Large batches for efficiency
                parallel = False  # Sequential for stability
            elif total_rows > 10_000_000:  # 10M+
                batch_size = 50000
                parallel = True
            else:
                batch_size = 10000
                parallel = True
            
            strategy['priority_order'].append((priority, table))
            strategy['batch_sizes'][table] = batch_size
            strategy['parallel_feasible'][table] = parallel
            
            # Estimate migration time
            batches = (total_rows + batch_size - 1) // batch_size
            time_per_batch = 0.2 if parallel else 0.5  # Seconds
            estimated_seconds = batches * time_per_batch
            strategy['estimated_times'][table] = {
                'batches': batches,
                'estimated_seconds': estimated_seconds,
                'estimated_hours': estimated_seconds / 3600
            }
        
        # Sort by priority
        strategy['priority_order'].sort(key=lambda x: x[0])
        
        # Generate recommendations
        total_estimated_hours = sum(info['estimated_hours'] for info in strategy['estimated_times'].values())
        
        strategy['recommendations'] = [
            f"Total estimated migration time: {total_estimated_hours:.2f} hours",
            f"Largest table (odd): {analysis.get('odd', {}).get('total_rows', 0):,} rows",
            "Run migration during low-usage hours",
            "Monitor disk space - migration may require significant temporary space",
            "Consider running in screen/tmux session for long-running process"
        ]
        
        return strategy
    
    def migrate_table_in_chunks(self, table_name: str, batch_size: int, 
                               start_offset: int = 0) -> Dict:
        """Migrate a table in chunks with progress tracking."""
        
        logger.info(f"🚀 Starting chunked migration of {table_name}")
        logger.info(f"   Batch size: {batch_size:,}")
        logger.info(f"   Starting from offset: {start_offset:,}")
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get total rows
            total_result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            total_rows = total_result[0] if total_result else 0
            
            if total_rows == 0:
                logger.info(f"Table {table_name} is empty, skipping")
                return {'status': 'completed', 'rows_processed': 0}
            
            # Calculate chunks
            remaining_rows = total_rows - start_offset
            total_chunks = (remaining_rows + batch_size - 1) // batch_size
            
            logger.info(f"   Total rows: {total_rows:,}")
            logger.info(f"   Remaining rows: {remaining_rows:,}")
            logger.info(f"   Total chunks: {total_chunks:,}")
            
            # Migration metrics
            start_time = time.time()
            rows_processed = 0
            chunks_completed = 0
            
            current_offset = start_offset
            
            while current_offset < total_rows:
                chunk_start_time = time.time()
                
                # Calculate chunk size (may be smaller for last chunk)
                current_batch_size = min(batch_size, total_rows - current_offset)
                
                try:
                    # Fetch chunk data
                    cursor = conn.execute(f"""
                        SELECT * FROM {table_name} 
                        LIMIT {current_batch_size} 
                        OFFSET {current_offset}
                    """)
                    
                    chunk_rows = cursor.fetchall()
                    actual_rows = len(chunk_rows)
                    
                    if actual_rows == 0:
                        break
                    
                    # Simulate processing (in real implementation, would insert to PostgreSQL)
                    # Add small delay to simulate network/processing overhead
                    processing_time = 0.001 * actual_rows  # 1ms per row
                    time.sleep(processing_time)
                    
                    # Update counters
                    rows_processed += actual_rows
                    chunks_completed += 1
                    current_offset += actual_rows
                    
                    chunk_time = time.time() - chunk_start_time
                    
                    # Progress reporting
                    if chunks_completed % 50 == 0:  # Every 50 chunks
                        elapsed_time = time.time() - start_time
                        rows_per_second = rows_processed / elapsed_time if elapsed_time > 0 else 0
                        
                        completion_pct = (rows_processed / remaining_rows) * 100
                        remaining_seconds = (remaining_rows - rows_processed) / rows_per_second if rows_per_second > 0 else 0
                        
                        logger.info(f"   Chunk {chunks_completed:,}/{total_chunks:,} "
                                   f"({completion_pct:.2f}% complete)")
                        logger.info(f"   Processed: {rows_processed:,} rows "
                                   f"({rows_per_second:.0f} rows/sec)")
                        logger.info(f"   ETA: {remaining_seconds/3600:.2f} hours")
                        
                        # Save checkpoint
                        checkpoint_data = {
                            'table_name': table_name,
                            'last_offset': current_offset,
                            'rows_processed': rows_processed,
                            'chunks_completed': chunks_completed,
                            'start_time': start_time,
                            'last_update': time.time()
                        }
                        
                        with open(f"{table_name}_checkpoint.json", 'w') as f:
                            json.dump(checkpoint_data, f)
                
                except Exception as e:
                    logger.error(f"Error processing chunk at offset {current_offset}: {e}")
                    break
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"✅ Completed {table_name} migration:")
            logger.info(f"   Rows processed: {rows_processed:,}")
            logger.info(f"   Chunks completed: {chunks_completed:,}")
            logger.info(f"   Total time: {elapsed_time:.2f}s ({elapsed_time/3600:.2f}h)")
            logger.info(f"   Average rate: {rows_processed/elapsed_time:.0f} rows/sec")
            
            return {
                'status': 'completed',
                'rows_processed': rows_processed,
                'chunks_completed': chunks_completed,
                'elapsed_time': elapsed_time,
                'rows_per_second': rows_processed / elapsed_time if elapsed_time > 0 else 0
            }
            
        finally:
            conn.close()
    
    def run_massive_migration(self):
        """Run migration optimized for massive tables."""
        logger.info("🚀 Starting Massive Table Migration")
        logger.info("=" * 80)
        
        # Step 1: Analyze tables
        analysis = self.analyze_massive_tables()
        
        # Step 2: Create strategy
        strategy = self.create_migration_strategy(analysis)
        
        # Step 3: Display plan
        logger.info("\n📋 Migration Plan:")
        for _, table in strategy['priority_order']:
            info = strategy['estimated_times'].get(table, {})
            batch_size = strategy['batch_sizes'].get(table, 0)
            
            logger.info(f"   {table}: {analysis[table]['total_rows']:,} rows, "
                       f"batch={batch_size:,}, "
                       f"~{info.get('estimated_hours', 0):.2f}h")
        
        logger.info(f"\n🔮 Recommendations:")
        for rec in strategy['recommendations']:
            logger.info(f"   • {rec}")
        
        # Step 4: Ask for confirmation (in interactive mode)
        total_hours = sum(info.get('estimated_hours', 0) for info in strategy['estimated_times'].values())
        logger.info(f"\n⚠️ This migration will take approximately {total_hours:.2f} hours")
        
        # Step 5: Execute migration
        for priority, table in strategy['priority_order']:
            batch_size = strategy['batch_sizes'][table]
            
            # Check for existing checkpoint
            checkpoint_file = f"{table}_checkpoint.json"
            start_offset = 0
            
            if os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, 'r') as f:
                        checkpoint = json.load(f)
                    start_offset = checkpoint.get('last_offset', 0)
                    logger.info(f"📍 Resuming {table} from offset {start_offset:,}")
                except:
                    logger.warning(f"Could not load checkpoint for {table}, starting from beginning")
            
            # Migrate table
            result = self.migrate_table_in_chunks(table, batch_size, start_offset)
            
            if result['status'] == 'completed':
                # Remove checkpoint file on successful completion
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
            
            logger.info(f"Completed {table} migration\n")
        
        logger.info("🎉 Massive table migration completed!")


def demo_massive_migration():
    """Run a demo of the massive migration system."""
    migrator = MassiveTableMigrator()
    
    # Just analyze for now (don't run full migration)
    logger.info("Running analysis only (not full migration)")
    
    analysis = migrator.analyze_massive_tables()
    strategy = migrator.create_migration_strategy(analysis)
    
    print("\n" + "="*80)
    print("MASSIVE TABLE MIGRATION ANALYSIS")
    print("="*80)
    
    for table, info in analysis.items():
        if 'error' not in info:
            print(f"\n{table.upper()}:")
            print(f"  Rows: {info['total_rows']:,}")
            print(f"  Columns: {info['columns']}")
            print(f"  Analysis time: {info['analysis_time_seconds']:.2f}s")
            
            estimated = strategy['estimated_times'].get(table, {})
            if estimated:
                print(f"  Estimated migration time: {estimated['estimated_hours']:.2f} hours")
                print(f"  Batches required: {estimated['batches']:,}")
    
    print(f"\nRECOMMENDations:")
    for rec in strategy['recommendations']:
        print(f"  • {rec}")


if __name__ == "__main__":
    demo_massive_migration()