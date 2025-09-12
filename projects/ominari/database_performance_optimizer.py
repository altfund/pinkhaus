#!/usr/bin/env python3
"""
Database Performance Optimizer

Comprehensive database optimization for the 216GB sport_odds.db:
- Adds strategic indexes for common queries
- Analyzes query performance
- Implements chunked processing
- Optimizes database settings
- Monitors performance improvements
"""

import sqlite3
import logging
import time
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path

from database_v2 import db_manager
from models import Market, Odd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabasePerformanceOptimizer:
    """Optimizes database performance for large-scale operations."""
    
    def __init__(self, db_path: str = "sport_odds.db"):
        self.db_path = db_path
        self.optimization_log = []
        
    def analyze_database_structure(self) -> Dict:
        """Analyze current database structure and performance."""
        logger.info("🔍 Analyzing database structure and performance...")
        
        conn = sqlite3.connect(self.db_path)
        
        analysis = {
            'database_size': self._get_database_size(),
            'table_stats': self._get_table_statistics(conn),
            'index_stats': self._get_index_statistics(conn),
            'query_performance': self._analyze_query_performance(conn),
            'recommendations': []
        }
        
        conn.close()
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_optimization_recommendations(analysis)
        
        return analysis
    
    def _get_database_size(self) -> Dict:
        """Get database file sizes."""
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        
        # Check for WAL file
        wal_path = self.db_path + "-wal"
        wal_size = os.path.getsize(wal_path) if os.path.exists(wal_path) else 0
        
        return {
            'main_db_bytes': db_size,
            'main_db_gb': db_size / (1024**3),
            'wal_bytes': wal_size,
            'wal_mb': wal_size / (1024**2),
            'total_bytes': db_size + wal_size,
            'total_gb': (db_size + wal_size) / (1024**3)
        }
    
    def _get_table_statistics(self, conn: sqlite3.Connection) -> Dict:
        """Get statistics for all tables."""
        tables = ['market', 'odd', 'betting_session', 'bet', 'blockchain_markets', 'blockchain_odds', 'blockchain_trades']
        stats = {}
        
        for table in tables:
            try:
                # Get row count
                count_result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                row_count = count_result[0] if count_result else 0
                
                # Get table info
                table_info = conn.execute(f"PRAGMA table_info({table})").fetchall()
                
                stats[table] = {
                    'row_count': row_count,
                    'columns': len(table_info),
                    'column_info': table_info
                }
                
            except sqlite3.Error as e:
                stats[table] = {'error': str(e), 'row_count': 0}
        
        return stats
    
    def _get_index_statistics(self, conn: sqlite3.Connection) -> Dict:
        """Get existing index information."""
        indexes = conn.execute("SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index'").fetchall()
        
        index_stats = {
            'total_indexes': len(indexes),
            'indexes': []
        }
        
        for name, table, sql in indexes:
            if name and not name.startswith('sqlite_'):  # Skip auto-indexes
                index_stats['indexes'].append({
                    'name': name,
                    'table': table,
                    'sql': sql
                })
        
        return index_stats
    
    def _analyze_query_performance(self, conn: sqlite3.Connection) -> Dict:
        """Analyze performance of common queries."""
        
        # Common query patterns for sports betting
        test_queries = [
            ("market_by_sport", "SELECT COUNT(*) FROM market WHERE sport = 'Soccer'"),
            ("recent_odds", "SELECT COUNT(*) FROM odd WHERE updated_at > datetime('now', '-1 day')"),
            ("active_markets", "SELECT COUNT(*) FROM market WHERE is_finished = 0"),
            ("blockchain_markets", "SELECT COUNT(*) FROM market WHERE source LIKE 'blockchain_%'"),
            ("market_with_odds", "SELECT COUNT(*) FROM market m JOIN odd o ON m.source_id = o.source_id LIMIT 1000")
        ]
        
        performance = {}
        
        for query_name, sql in test_queries:
            try:
                start_time = time.time()
                result = conn.execute(sql).fetchone()
                end_time = time.time()
                
                performance[query_name] = {
                    'execution_time': end_time - start_time,
                    'result_count': result[0] if result else 0,
                    'status': 'success'
                }
                
            except Exception as e:
                performance[query_name] = {
                    'execution_time': 0,
                    'error': str(e),
                    'status': 'failed'
                }
        
        return performance
    
    def _generate_optimization_recommendations(self, analysis: Dict) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Check database size
        if analysis['database_size']['total_gb'] > 50:
            recommendations.append("Large database detected - consider partitioning or archiving old data")
        
        # Check WAL size
        if analysis['database_size']['wal_mb'] > 1000:
            recommendations.append("Large WAL file detected - consider checkpointing more frequently")
        
        # Check for missing indexes
        market_stats = analysis['table_stats'].get('market', {})
        odd_stats = analysis['table_stats'].get('odd', {})
        
        if market_stats.get('row_count', 0) > 100000:
            recommendations.append("Large market table - ensure indexes on sport, source, maturity_date")
        
        if odd_stats.get('row_count', 0) > 500000:
            recommendations.append("Large odd table - ensure indexes on source_id, bookmaker, updated_at")
        
        # Check query performance
        query_perf = analysis['query_performance']
        for query_name, perf in query_perf.items():
            if perf.get('execution_time', 0) > 1.0:
                recommendations.append(f"Slow query detected: {query_name} ({perf['execution_time']:.2f}s)")
        
        return recommendations
    
    def create_performance_indexes(self) -> Dict:
        """Create strategic indexes for better performance."""
        logger.info("📊 Creating performance indexes...")
        
        # Define strategic indexes
        indexes_to_create = [
            # Market table indexes
            ("idx_market_sport_maturity", "market", "CREATE INDEX IF NOT EXISTS idx_market_sport_maturity ON market(sport, maturity_date)"),
            ("idx_market_source_finished", "market", "CREATE INDEX IF NOT EXISTS idx_market_source_finished ON market(source, is_finished)"),
            ("idx_market_league_sport", "market", "CREATE INDEX IF NOT EXISTS idx_market_league_sport ON market(league_name, sport)"),
            ("idx_market_updated_at", "market", "CREATE INDEX IF NOT EXISTS idx_market_updated_at ON market(updated_at)"),
            
            # Odd table indexes  
            ("idx_odd_source_bookmaker", "odd", "CREATE INDEX IF NOT EXISTS idx_odd_source_bookmaker ON odd(source_id, bookmaker)"),
            ("idx_odd_updated_at", "odd", "CREATE INDEX IF NOT EXISTS idx_odd_updated_at ON odd(updated_at)"),
            ("idx_odd_bookmaker_type", "odd", "CREATE INDEX IF NOT EXISTS idx_odd_bookmaker_type ON odd(bookmaker, market_type)"),
            ("idx_odd_position_outcome", "odd", "CREATE INDEX IF NOT EXISTS idx_odd_position_outcome ON odd(position, outcome)"),
            
            # Blockchain-specific indexes
            ("idx_blockchain_markets_network", "blockchain_markets", "CREATE INDEX IF NOT EXISTS idx_blockchain_markets_network ON blockchain_markets(network, maturity_date)"),
            ("idx_blockchain_odds_market", "blockchain_odds", "CREATE INDEX IF NOT EXISTS idx_blockchain_odds_market ON blockchain_odds(market_address, timestamp)"),
            ("idx_blockchain_trades_market", "blockchain_trades", "CREATE INDEX IF NOT EXISTS idx_blockchain_trades_market ON blockchain_trades(market_address, timestamp)"),
            
            # Composite indexes for common joins
            ("idx_market_odd_join", "market", "CREATE INDEX IF NOT EXISTS idx_market_odd_join ON market(source_id, sport, is_finished)")
        ]
        
        conn = sqlite3.connect(self.db_path)
        results = {
            'created': [],
            'failed': [],
            'already_existed': []
        }
        
        for index_name, table, sql in indexes_to_create:
            try:
                start_time = time.time()
                conn.execute(sql)
                end_time = time.time()
                
                results['created'].append({
                    'name': index_name,
                    'table': table,
                    'creation_time': end_time - start_time
                })
                
                logger.info(f"✅ Created index {index_name} on {table} ({end_time - start_time:.2f}s)")
                
            except sqlite3.Error as e:
                if "already exists" in str(e):
                    results['already_existed'].append(index_name)
                else:
                    results['failed'].append({
                        'name': index_name,
                        'error': str(e)
                    })
                    logger.error(f"❌ Failed to create index {index_name}: {e}")
        
        conn.commit()
        conn.close()
        
        return results
    
    def optimize_database_settings(self) -> Dict:
        """Optimize database settings for performance."""
        logger.info("⚙️ Optimizing database settings...")
        
        conn = sqlite3.connect(self.db_path)
        
        optimizations = [
            ("page_size", "PRAGMA page_size = 4096"),  # Optimal page size
            ("cache_size", "PRAGMA cache_size = -2000000"),  # 2GB cache
            ("temp_store", "PRAGMA temp_store = MEMORY"),  # Use memory for temp
            ("journal_mode", "PRAGMA journal_mode = WAL"),  # WAL mode for concurrency
            ("synchronous", "PRAGMA synchronous = NORMAL"),  # Balance safety/speed
            ("wal_autocheckpoint", "PRAGMA wal_autocheckpoint = 1000"),  # Checkpoint every 1000 pages
            ("mmap_size", "PRAGMA mmap_size = 268435456"),  # 256MB memory map
        ]
        
        results = {}
        
        for setting_name, pragma_sql in optimizations:
            try:
                # Get current value
                current_result = conn.execute(f"PRAGMA {setting_name}").fetchone()
                current_value = current_result[0] if current_result else "unknown"
                
                # Apply optimization
                conn.execute(pragma_sql)
                
                # Get new value
                new_result = conn.execute(f"PRAGMA {setting_name}").fetchone()
                new_value = new_result[0] if new_result else "unknown"
                
                results[setting_name] = {
                    'previous': current_value,
                    'new': new_value,
                    'status': 'success'
                }
                
                logger.info(f"✅ {setting_name}: {current_value} → {new_value}")
                
            except sqlite3.Error as e:
                results[setting_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
                logger.error(f"❌ Failed to optimize {setting_name}: {e}")
        
        conn.close()
        return results
    
    def run_maintenance_tasks(self) -> Dict:
        """Run database maintenance tasks."""
        logger.info("🧹 Running database maintenance tasks...")
        
        conn = sqlite3.connect(self.db_path)
        
        maintenance_tasks = [
            ("analyze", "ANALYZE"),  # Update query planner statistics
            ("vacuum", "VACUUM"),    # Defragment database (warning: can take long time)
        ]
        
        results = {}
        
        for task_name, sql in maintenance_tasks:
            try:
                if task_name == "vacuum":
                    # Skip vacuum for very large databases unless explicitly requested
                    db_size_gb = self._get_database_size()['main_db_gb']
                    if db_size_gb > 50:
                        results[task_name] = {
                            'status': 'skipped',
                            'reason': f'Database too large ({db_size_gb:.1f}GB) - vacuum would take too long'
                        }
                        logger.info(f"⏭️ Skipped {task_name}: database too large")
                        continue
                
                start_time = time.time()
                conn.execute(sql)
                end_time = time.time()
                
                results[task_name] = {
                    'execution_time': end_time - start_time,
                    'status': 'success'
                }
                
                logger.info(f"✅ Completed {task_name} ({end_time - start_time:.2f}s)")
                
            except sqlite3.Error as e:
                results[task_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
                logger.error(f"❌ Failed {task_name}: {e}")
        
        conn.close()
        return results
    
    def benchmark_performance(self) -> Dict:
        """Benchmark database performance before and after optimization."""
        logger.info("⏱️ Benchmarking database performance...")
        
        # Test queries that represent common operations
        benchmark_queries = [
            ("count_markets", "SELECT COUNT(*) FROM market"),
            ("count_odds", "SELECT COUNT(*) FROM odd"),
            ("recent_markets", "SELECT COUNT(*) FROM market WHERE updated_at > datetime('now', '-7 days')"),
            ("active_soccer", "SELECT COUNT(*) FROM market WHERE sport = 'Soccer' AND is_finished = 0"),
            ("blockchain_markets", "SELECT COUNT(*) FROM market WHERE source LIKE 'blockchain_%'"),
            ("market_odds_join", "SELECT COUNT(*) FROM market m JOIN odd o ON m.source_id = o.source_id LIMIT 10000")
        ]
        
        conn = sqlite3.connect(self.db_path)
        results = {}
        
        for query_name, sql in benchmark_queries:
            times = []
            
            # Run query 3 times and take average
            for i in range(3):
                try:
                    start_time = time.time()
                    result = conn.execute(sql).fetchone()
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    
                except Exception as e:
                    times.append(None)
                    logger.error(f"Benchmark query {query_name} failed: {e}")
                    break
            
            valid_times = [t for t in times if t is not None]
            
            if valid_times:
                results[query_name] = {
                    'avg_time': sum(valid_times) / len(valid_times),
                    'min_time': min(valid_times),
                    'max_time': max(valid_times),
                    'runs': len(valid_times)
                }
            else:
                results[query_name] = {'error': 'All benchmark runs failed'}
        
        conn.close()
        return results
    
    def run_comprehensive_optimization(self) -> Dict:
        """Run complete database optimization process."""
        logger.info("🚀 Starting Comprehensive Database Optimization")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # 1. Initial analysis
        logger.info("1. Analyzing database structure...")
        initial_analysis = self.analyze_database_structure()
        
        # 2. Benchmark before optimization
        logger.info("2. Running initial performance benchmark...")
        before_benchmark = self.benchmark_performance()
        
        # 3. Optimize settings
        logger.info("3. Optimizing database settings...")
        settings_result = self.optimize_database_settings()
        
        # 4. Create indexes
        logger.info("4. Creating performance indexes...")
        index_result = self.create_performance_indexes()
        
        # 5. Run maintenance
        logger.info("5. Running maintenance tasks...")
        maintenance_result = self.run_maintenance_tasks()
        
        # 6. Benchmark after optimization
        logger.info("6. Running post-optimization benchmark...")
        after_benchmark = self.benchmark_performance()
        
        end_time = time.time()
        
        optimization_summary = {
            'total_time': end_time - start_time,
            'database_info': initial_analysis['database_size'],
            'initial_analysis': initial_analysis,
            'before_benchmark': before_benchmark,
            'optimization_results': {
                'settings': settings_result,
                'indexes': index_result,
                'maintenance': maintenance_result
            },
            'after_benchmark': after_benchmark,
            'performance_improvements': self._calculate_performance_improvements(
                before_benchmark, after_benchmark
            )
        }
        
        # Log summary
        self._log_optimization_summary(optimization_summary)
        
        return optimization_summary
    
    def _calculate_performance_improvements(self, before: Dict, after: Dict) -> Dict:
        """Calculate performance improvements."""
        improvements = {}
        
        for query_name in before.keys():
            if query_name in after:
                before_time = before[query_name].get('avg_time')
                after_time = after[query_name].get('avg_time')
                
                if before_time and after_time:
                    improvement_pct = ((before_time - after_time) / before_time) * 100
                    improvements[query_name] = {
                        'before': before_time,
                        'after': after_time,
                        'improvement_pct': improvement_pct,
                        'improvement_factor': before_time / after_time if after_time > 0 else 1.0
                    }
        
        return improvements
    
    def _log_optimization_summary(self, summary: Dict):
        """Log optimization summary."""
        logger.info("\n" + "="*80)
        logger.info("DATABASE OPTIMIZATION COMPLETE")
        logger.info("="*80)
        
        db_info = summary['database_info']
        logger.info(f"Database size: {db_info['total_gb']:.2f} GB")
        logger.info(f"Total optimization time: {summary['total_time']:.2f}s")
        
        # Index creation results
        index_results = summary['optimization_results']['indexes']
        logger.info(f"\nIndexes created: {len(index_results['created'])}")
        logger.info(f"Indexes already existed: {len(index_results['already_existed'])}")
        logger.info(f"Index creation failures: {len(index_results['failed'])}")
        
        # Performance improvements
        improvements = summary['performance_improvements']
        if improvements:
            logger.info(f"\nPerformance Improvements:")
            for query, improvement in improvements.items():
                if improvement['improvement_pct'] > 0:
                    logger.info(f"  {query}: {improvement['improvement_pct']:.1f}% faster "
                               f"({improvement['before']:.3f}s → {improvement['after']:.3f}s)")
                else:
                    logger.info(f"  {query}: {abs(improvement['improvement_pct']):.1f}% slower")


if __name__ == "__main__":
    optimizer = DatabasePerformanceOptimizer()
    results = optimizer.run_comprehensive_optimization()
    
    print(f"\n🎯 Optimization completed in {results['total_time']:.2f} seconds")
    print(f"Database: {results['database_info']['total_gb']:.2f} GB")
    print(f"Indexes created: {len(results['optimization_results']['indexes']['created'])}")
    
    if results['performance_improvements']:
        print(f"\nTop performance improvements:")
        sorted_improvements = sorted(
            results['performance_improvements'].items(),
            key=lambda x: x[1]['improvement_pct'],
            reverse=True
        )[:3]
        
        for query, improvement in sorted_improvements:
            if improvement['improvement_pct'] > 0:
                print(f"  {query}: {improvement['improvement_pct']:.1f}% faster")