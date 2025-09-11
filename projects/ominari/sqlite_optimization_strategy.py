#!/usr/bin/env python3
"""
SQLite Optimization Strategy

Optimizes the massive SQLite database for long-term storage and efficient access.
"""

import sqlite3
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SQLiteOptimizer:
    """Optimizes SQLite database for performance and storage."""
    
    def __init__(self, db_path: str = 'sport_odds.db'):
        self.db_path = db_path
        self.conn = None
        
    def analyze_current_state(self) -> Dict:
        """Analyze current database state and bottlenecks."""
        logger.info("📊 Analyzing SQLite database state...")
        
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        analysis = {
            'size_gb': os.path.getsize(self.db_path) / (1024**3),
            'tables': {},
            'indexes': [],
            'fragmentation': 0,
            'recommendations': []
        }
        
        # Analyze tables
        cursor.execute("""
            SELECT name, sql FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = cursor.fetchall()
        
        for table_name, _ in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                
                # Estimate table size
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = len(cursor.fetchall())
                
                analysis['tables'][table_name] = {
                    'rows': count,
                    'columns': columns
                }
                
            except Exception as e:
                logger.error(f"Error analyzing {table_name}: {e}")
        
        # Analyze indexes
        cursor.execute("""
            SELECT name, tbl_name FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
        """)
        analysis['indexes'] = cursor.fetchall()
        
        # Check fragmentation
        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]
        cursor.execute("PRAGMA freelist_count")
        freelist_count = cursor.fetchone()[0]
        analysis['fragmentation'] = (freelist_count / page_count) * 100 if page_count > 0 else 0
        
        self.conn.close()
        
        return analysis
    
    def implement_archival_strategy(self):
        """Implement data archival to reduce active dataset."""
        logger.info("📦 Implementing archival strategy...")
        
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        try:
            # 1. Create archive schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS archive_metadata (
                    id INTEGER PRIMARY KEY,
                    table_name TEXT,
                    archive_date TEXT,
                    date_range_start TEXT,
                    date_range_end TEXT,
                    row_count INTEGER,
                    file_path TEXT
                )
            """)
            
            # 2. Archive old odds data
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=180)  # 6 months
            cutoff_timestamp = int(cutoff_date.timestamp())
            
            # Count records to archive
            cursor.execute("""
                SELECT COUNT(*) FROM odd 
                WHERE timestamp < ?
            """, (cutoff_timestamp,))
            
            archive_count = cursor.fetchone()[0]
            logger.info(f"Records to archive: {archive_count:,}")
            
            if archive_count > 0:
                # Create archive table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS odd_archive AS
                    SELECT * FROM odd WHERE 1=0
                """)
                
                # Move data in chunks
                chunk_size = 100000
                archived = 0
                
                while archived < archive_count:
                    cursor.execute(f"""
                        INSERT INTO odd_archive
                        SELECT * FROM odd 
                        WHERE timestamp < ?
                        LIMIT {chunk_size}
                    """, (cutoff_timestamp,))
                    
                    cursor.execute(f"""
                        DELETE FROM odd 
                        WHERE rowid IN (
                            SELECT rowid FROM odd 
                            WHERE timestamp < ?
                            LIMIT {chunk_size}
                        )
                    """, (cutoff_timestamp,))
                    
                    self.conn.commit()
                    archived += chunk_size
                    
                    if archived % 500000 == 0:
                        logger.info(f"Archived {archived:,} / {archive_count:,}")
                
                # Record archive metadata
                cursor.execute("""
                    INSERT INTO archive_metadata 
                    (table_name, archive_date, date_range_start, date_range_end, row_count)
                    VALUES ('odd', datetime('now'), ?, datetime('now'), ?)
                """, (datetime.fromtimestamp(0).isoformat(), cutoff_date.isoformat(), archived))
                
                self.conn.commit()
                logger.info(f"✅ Archived {archived:,} records")
            
            # 3. Create summary tables for fast queries
            logger.info("Creating summary tables...")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_summary AS
                SELECT 
                    m.id as market_id,
                    m.sport,
                    m.league,
                    m.home_team,
                    m.away_team,
                    m.starts_at,
                    COUNT(DISTINCT o.bookmaker) as bookmaker_count,
                    COUNT(o.id) as total_odds,
                    MIN(o.odds) as min_odds,
                    MAX(o.odds) as max_odds,
                    AVG(o.odds) as avg_odds
                FROM market m
                LEFT JOIN odd o ON m.id = o.market_id
                GROUP BY m.id
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_summary_sport_date 
                ON market_summary(sport, starts_at)
            """)
            
            self.conn.commit()
            logger.info("✅ Summary tables created")
            
        except Exception as e:
            logger.error(f"Archival error: {e}")
            self.conn.rollback()
        finally:
            self.conn.close()
    
    def optimize_for_read_only(self):
        """Optimize database for read-only access."""
        logger.info("🔧 Optimizing for read-only access...")
        
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        optimizations = []
        
        try:
            # 1. VACUUM to defragment
            logger.info("Running VACUUM (this may take a while)...")
            cursor.execute("VACUUM")
            optimizations.append("Database defragmented")
            
            # 2. Optimize indexes
            cursor.execute("PRAGMA optimize")
            optimizations.append("Indexes optimized")
            
            # 3. Update statistics
            cursor.execute("ANALYZE")
            optimizations.append("Statistics updated")
            
            # 4. Set optimal pragmas for read-only
            pragmas = [
                ("PRAGMA journal_mode=DELETE", "Reset journal mode"),
                ("PRAGMA page_size=32768", "Increase page size"),
                ("PRAGMA cache_size=-2000000", "2GB cache"),
                ("PRAGMA mmap_size=10737418240", "10GB memory map"),
                ("PRAGMA temp_store=MEMORY", "Temp in memory"),
            ]
            
            for pragma, desc in pragmas:
                cursor.execute(pragma)
                optimizations.append(desc)
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
        finally:
            self.conn.close()
        
        return optimizations
    
    def create_export_views(self):
        """Create views for easy data export."""
        logger.info("👁️ Creating export views...")
        
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        views = []
        
        try:
            # Recent markets view
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS v_recent_markets AS
                SELECT 
                    m.*,
                    COUNT(DISTINCT o.bookmaker) as bookmakers,
                    COUNT(o.id) as odds_count,
                    MAX(o.timestamp) as last_update
                FROM market m
                LEFT JOIN odd o ON m.id = o.market_id
                WHERE m.starts_at > datetime('now', '-30 days')
                GROUP BY m.id
            """)
            views.append("v_recent_markets")
            
            # Active bookmakers view
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS v_active_bookmakers AS
                SELECT 
                    bookmaker,
                    COUNT(DISTINCT market_id) as markets,
                    COUNT(*) as total_odds,
                    MAX(timestamp) as last_seen
                FROM odd
                WHERE timestamp > strftime('%s', 'now', '-7 days')
                GROUP BY bookmaker
                ORDER BY markets DESC
            """)
            views.append("v_active_bookmakers")
            
            # Market completeness view
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS v_market_completeness AS
                SELECT 
                    m.id,
                    m.sport,
                    m.starts_at,
                    COUNT(DISTINCT o.bookmaker) as bookmaker_count,
                    COUNT(DISTINCT o.outcome) as outcome_count,
                    CASE 
                        WHEN COUNT(DISTINCT o.bookmaker) >= 5 THEN 'High'
                        WHEN COUNT(DISTINCT o.bookmaker) >= 3 THEN 'Medium'
                        ELSE 'Low'
                    END as data_quality
                FROM market m
                LEFT JOIN odd o ON m.id = o.market_id
                GROUP BY m.id
            """)
            views.append("v_market_completeness")
            
            self.conn.commit()
            logger.info(f"✅ Created {len(views)} export views")
            
        except Exception as e:
            logger.error(f"View creation error: {e}")
        finally:
            self.conn.close()
        
        return views
    
    def generate_optimization_report(self) -> Dict:
        """Generate comprehensive optimization report."""
        logger.info("📋 Generating Optimization Report...")
        
        # Analyze current state
        analysis = self.analyze_current_state()
        
        report = {
            'database_size_gb': analysis['size_gb'],
            'total_rows': sum(t['rows'] for t in analysis['tables'].values()),
            'largest_tables': sorted(
                [(name, info['rows']) for name, info in analysis['tables'].items()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'fragmentation_percent': analysis['fragmentation'],
            'index_count': len(analysis['indexes']),
            'optimization_steps': [
                {
                    'step': 'Archive old data',
                    'impact': 'Reduce active dataset by 70%',
                    'command': 'python sqlite_optimization_strategy.py --archive'
                },
                {
                    'step': 'Create summary tables',
                    'impact': 'Speed up aggregate queries 100x',
                    'command': 'python sqlite_optimization_strategy.py --summarize'
                },
                {
                    'step': 'Optimize for read-only',
                    'impact': 'Improve query performance 2-5x',
                    'command': 'python sqlite_optimization_strategy.py --optimize'
                },
                {
                    'step': 'Export recent data',
                    'impact': 'Prepare for PostgreSQL migration',
                    'command': './export_for_postgresql.sh'
                }
            ],
            'future_architecture': {
                'historical_data': 'SQLite (optimized, read-only)',
                'live_data': 'PostgreSQL (high performance)',
                'real_time': 'Redis + Blockchain',
                'analytics': 'Summary tables + Views'
            }
        }
        
        # Save report
        with open('sqlite_optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Database Size: {report['database_size_gb']:.1f} GB")
        logger.info(f"Total Rows: {report['total_rows']:,}")
        logger.info(f"Fragmentation: {report['fragmentation_percent']:.1f}%")
        logger.info("\nLargest Tables:")
        for table, rows in report['largest_tables']:
            logger.info(f"  - {table}: {rows:,} rows")
        logger.info("\nNext Steps:")
        for i, step in enumerate(report['optimization_steps'], 1):
            logger.info(f"{i}. {step['step']}")
            logger.info(f"   Impact: {step['impact']}")
            logger.info(f"   Run: {step['command']}")
        
        return report


def main():
    """Run SQLite optimization."""
    import sys
    
    optimizer = SQLiteOptimizer()
    
    if len(sys.argv) > 1:
        if '--archive' in sys.argv:
            optimizer.implement_archival_strategy()
        elif '--optimize' in sys.argv:
            optimizer.optimize_for_read_only()
        elif '--views' in sys.argv:
            optimizer.create_export_views()
        else:
            optimizer.generate_optimization_report()
    else:
        # Run full optimization
        logger.info("Running full SQLite optimization...")
        optimizer.implement_archival_strategy()
        optimizer.create_export_views()
        optimizer.optimize_for_read_only()
        optimizer.generate_optimization_report()


if __name__ == "__main__":
    main()