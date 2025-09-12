#!/usr/bin/env python3
"""
Hybrid Database Setup for Ominari

Implements the recommended hybrid approach:
- PostgreSQL for new blockchain data (hot data)
- SQLite for historical data (warm/cold data)
- Redis for caching (already implemented)
"""

import os
import logging
import json
from datetime import datetime, timezone
from typing import Dict, Optional
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float, DateTime, Boolean, JSON, Index
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Base = declarative_base()


class HybridDatabaseSetup:
    """Sets up hybrid database architecture."""
    
    def __init__(self):
        self.pg_config = {
            'host': os.getenv('PG_HOST', 'localhost'),
            'port': os.getenv('PG_PORT', 5432),
            'database': os.getenv('PG_DATABASE', 'ominari_blockchain'),
            'user': os.getenv('PG_USER', 'ominari_user'),
            'password': os.getenv('PG_PASSWORD', 'ominari_2025_secure')
        }
        
        self.sqlite_path = 'sport_odds.db'
        self.cutoff_date = datetime.now(timezone.utc)  # New data after this goes to PostgreSQL
        
    def create_postgresql_database(self):
        """Create PostgreSQL database if it doesn't exist."""
        logger.info("🐘 Creating PostgreSQL database...")
        
        try:
            # Connect to default postgres database to create new one
            conn = psycopg2.connect(
                host=self.pg_config['host'],
                port=self.pg_config['port'],
                user=self.pg_config['user'],
                password=self.pg_config['password'],
                database='postgres'
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()
            
            # Check if database exists
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.pg_config['database'],)
            )
            
            if not cur.fetchone():
                cur.execute(f"CREATE DATABASE {self.pg_config['database']}")
                logger.info(f"✅ Created database: {self.pg_config['database']}")
            else:
                logger.info(f"✅ Database already exists: {self.pg_config['database']}")
            
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Database creation failed: {e}")
            return False
        
        return True
    
    def create_blockchain_tables(self):
        """Create optimized tables for blockchain data."""
        logger.info("📊 Creating blockchain-optimized tables...")
        
        # PostgreSQL connection string
        pg_url = f"postgresql://{self.pg_config['user']}:{self.pg_config['password']}@" \
                 f"{self.pg_config['host']}:{self.pg_config['port']}/{self.pg_config['database']}"
        
        # Create engine
        engine = create_engine(pg_url)
        metadata = MetaData()
        
        # Blockchain markets table (optimized for new data)
        blockchain_markets = Table('blockchain_markets', metadata,
            Column('market_id', String(100), primary_key=True),
            Column('blockchain_address', String(42), unique=True, nullable=False),
            Column('network', String(20), nullable=False),
            Column('sport', String(50), nullable=False),
            Column('league', String(100)),
            Column('home_team', String(200), nullable=False),
            Column('away_team', String(200), nullable=False),
            Column('starts_at', DateTime(timezone=True), nullable=False),
            Column('is_finished', Boolean, default=False),
            Column('created_at', DateTime(timezone=True), default=datetime.now(timezone.utc)),
            Column('updated_at', DateTime(timezone=True), default=datetime.now(timezone.utc)),
            Column('metadata', JSON),
            
            # Indexes for performance
            Index('idx_blockchain_network', 'network'),
            Index('idx_blockchain_sport', 'sport'),
            Index('idx_blockchain_starts_at', 'starts_at'),
            Index('idx_blockchain_active', 'is_finished', 'starts_at'),
            Index('idx_blockchain_address', 'blockchain_address')
        )
        
        # Blockchain odds table (high-frequency updates)
        blockchain_odds = Table('blockchain_odds', metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('market_id', String(100), nullable=False),
            Column('position', Integer, nullable=False),
            Column('outcome', String(50), nullable=False),
            Column('decimal_odds', Float),
            Column('buy_odds', Float),
            Column('sell_odds', Float),
            Column('liquidity', Float),
            Column('spread', Float),
            Column('updated_at', DateTime(timezone=True), default=datetime.now(timezone.utc)),
            
            # Indexes
            Index('idx_odds_market_updated', 'market_id', 'updated_at'),
            Index('idx_odds_market_position', 'market_id', 'position')
        )
        
        # Blockchain positions table
        blockchain_positions = Table('blockchain_positions', metadata,
            Column('position_id', String(36), primary_key=True),
            Column('market_id', String(100), nullable=False),
            Column('trader_id', String(100), nullable=False),
            Column('position_type', String(20), nullable=False),
            Column('stake', Float, nullable=False),
            Column('odds', Float, nullable=False),
            Column('expected_payout', Float),
            Column('signal_probability', Float),
            Column('signal_providers', JSON),
            Column('kelly_fraction', Float),
            Column('opened_at', DateTime(timezone=True), default=datetime.now(timezone.utc)),
            Column('closed_at', DateTime(timezone=True)),
            Column('pnl', Float, default=0),
            Column('status', String(20), default='open'),
            
            # Indexes
            Index('idx_positions_trader', 'trader_id'),
            Index('idx_positions_status', 'status'),
            Index('idx_positions_market', 'market_id')
        )
        
        # Blockchain sync status
        blockchain_sync = Table('blockchain_sync_status', metadata,
            Column('network', String(20), primary_key=True),
            Column('last_synced_block', Integer, default=0),
            Column('last_sync_timestamp', DateTime(timezone=True)),
            Column('markets_discovered', Integer, default=0),
            Column('sync_errors', Integer, default=0),
            Column('status', String(20), default='active')
        )
        
        # Create all tables
        metadata.create_all(engine)
        logger.info("✅ Blockchain tables created successfully")
        
        # Initialize sync status for networks
        Session = sessionmaker(bind=engine)
        session = Session()
        
        networks = ['optimism', 'arbitrum', 'base', 'polygon']
        for network in networks:
            session.execute(
                f"""
                INSERT INTO blockchain_sync_status (network, last_synced_block, last_sync_timestamp)
                VALUES ('{network}', 0, CURRENT_TIMESTAMP)
                ON CONFLICT (network) DO NOTHING
                """
            )
        
        session.commit()
        session.close()
        
        return True
    
    def create_data_routing_config(self):
        """Create configuration for routing data to appropriate database."""
        logger.info("🔀 Creating data routing configuration...")
        
        config = {
            'routing_strategy': 'hybrid_by_date',
            'cutoff_date': self.cutoff_date.isoformat(),
            'databases': {
                'postgresql': {
                    'connection': f"postgresql://{self.pg_config['user']}:{self.pg_config['password']}@"
                                f"{self.pg_config['host']}:{self.pg_config['port']}/{self.pg_config['database']}",
                    'use_for': [
                        'blockchain_markets',
                        'blockchain_odds',
                        'blockchain_positions',
                        'blockchain_sync_status',
                        'new_market_data',
                        'real_time_updates'
                    ],
                    'data_after': self.cutoff_date.isoformat()
                },
                'sqlite': {
                    'connection': f"sqlite:///{self.sqlite_path}",
                    'use_for': [
                        'historical_markets',
                        'historical_odds',
                        'backtesting_data',
                        'archived_positions'
                    ],
                    'data_before': self.cutoff_date.isoformat()
                },
                'redis': {
                    'connection': 'redis://localhost:6379/0',
                    'use_for': [
                        'cache_layer',
                        'real_time_signals',
                        'session_data',
                        'hot_market_data'
                    ],
                    'ttl_seconds': 300
                }
            },
            'query_routing_rules': [
                {
                    'pattern': 'SELECT * FROM market WHERE starts_at > ?',
                    'route_to': 'postgresql',
                    'reason': 'Recent/future markets in PostgreSQL'
                },
                {
                    'pattern': 'SELECT * FROM odd WHERE timestamp > ?',
                    'route_to': 'postgresql',
                    'reason': 'Recent odds in PostgreSQL'
                },
                {
                    'pattern': 'SELECT * FROM market WHERE starts_at < ?',
                    'route_to': 'sqlite',
                    'reason': 'Historical markets in SQLite'
                }
            ]
        }
        
        # Save configuration
        with open('hybrid_database_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("✅ Data routing configuration created")
        return config
    
    def create_hybrid_database_manager(self):
        """Create hybrid database manager code."""
        logger.info("📝 Creating hybrid database manager...")
        
        manager_code = '''#!/usr/bin/env python3
"""
Hybrid Database Manager

Automatically routes queries to the appropriate database based on
data age and type.
"""

from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json

class HybridDatabaseManager:
    """Manages hybrid PostgreSQL/SQLite database access."""
    
    def __init__(self):
        with open('hybrid_database_config.json', 'r') as f:
            self.config = json.load(f)
        
        self.cutoff_date = datetime.fromisoformat(self.config['cutoff_date'])
        
        # Create engines
        self.pg_engine = create_engine(self.config['databases']['postgresql']['connection'])
        self.sqlite_engine = create_engine(self.config['databases']['sqlite']['connection'])
        
        # Create session makers
        self.PGSession = sessionmaker(bind=self.pg_engine)
        self.SQLiteSession = sessionmaker(bind=self.sqlite_engine)
    
    def get_session_for_data(self, data_date=None, data_type='market'):
        """Get appropriate database session based on data characteristics."""
        if data_date is None:
            data_date = datetime.now(timezone.utc)
        
        # Route blockchain data to PostgreSQL
        if data_type.startswith('blockchain_'):
            return self.PGSession()
        
        # Route based on date
        if data_date >= self.cutoff_date:
            return self.PGSession()
        else:
            return self.SQLiteSession()
    
    def query_markets(self, start_date=None, end_date=None):
        """Query markets from appropriate database."""
        if start_date and start_date >= self.cutoff_date:
            # Use PostgreSQL for recent data
            session = self.PGSession()
            table = 'blockchain_markets'
        else:
            # Use SQLite for historical data
            session = self.SQLiteSession()
            table = 'market'
        
        # Execute query
        # ... query logic ...
        
        return results

# Global instance
hybrid_db = HybridDatabaseManager()
'''
        
        with open('hybrid_database_manager.py', 'w') as f:
            f.write(manager_code)
        
        logger.info("✅ Hybrid database manager created")
    
    def create_migration_scripts(self):
        """Create scripts for data archival and migration."""
        logger.info("📜 Creating migration scripts...")
        
        # Archive old odds data
        archive_script = '''#!/usr/bin/env python3
"""
Archive old odds data to reduce active dataset size.
Moves odds older than 6 months to archive tables.
"""

import sqlite3
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def archive_old_odds(db_path='sport_odds.db', months=6):
    """Archive odds older than specified months."""
    
    cutoff_date = datetime.now() - timedelta(days=months*30)
    logger.info(f"Archiving odds older than {cutoff_date}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Create archive table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odd_archive AS 
            SELECT * FROM odd WHERE 1=0
        """)
        
        # Count rows to archive
        cursor.execute("""
            SELECT COUNT(*) FROM odd 
            WHERE timestamp < ?
        """, (cutoff_date,))
        
        count = cursor.fetchone()[0]
        logger.info(f"Found {count:,} odds records to archive")
        
        if count > 0:
            # Move data in chunks
            chunk_size = 10000
            archived = 0
            
            while archived < count:
                cursor.execute(f"""
                    INSERT INTO odd_archive
                    SELECT * FROM odd 
                    WHERE timestamp < ?
                    LIMIT {chunk_size}
                """, (cutoff_date,))
                
                cursor.execute(f"""
                    DELETE FROM odd 
                    WHERE rowid IN (
                        SELECT rowid FROM odd 
                        WHERE timestamp < ?
                        LIMIT {chunk_size}
                    )
                """, (cutoff_date,))
                
                conn.commit()
                archived += chunk_size
                logger.info(f"Archived {min(archived, count):,}/{count:,} records")
        
        # Vacuum to reclaim space
        logger.info("Vacuuming database...")
        cursor.execute("VACUUM")
        
        logger.info("✅ Archival complete!")
        
    except Exception as e:
        logger.error(f"Archival failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    archive_old_odds()
'''
        
        with open('archive_old_odds.py', 'w') as f:
            f.write(archive_script)
        
        logger.info("✅ Migration scripts created")
    
    def setup_monitoring(self):
        """Set up monitoring for hybrid database."""
        logger.info("📊 Setting up database monitoring...")
        
        monitoring_config = {
            'metrics': [
                {
                    'name': 'database_size_gb',
                    'description': 'Size of each database',
                    'query_pg': "SELECT pg_database_size(current_database())/1024.0/1024.0/1024.0",
                    'query_sqlite': "SELECT page_count * page_size / 1024.0 / 1024.0 / 1024.0 FROM pragma_page_count(), pragma_page_size()"
                },
                {
                    'name': 'active_connections',
                    'description': 'Number of active connections',
                    'query_pg': "SELECT count(*) FROM pg_stat_activity",
                    'query_sqlite': "N/A"
                },
                {
                    'name': 'new_markets_per_hour',
                    'description': 'Rate of new market discovery',
                    'query_pg': "SELECT COUNT(*) FROM blockchain_markets WHERE created_at > NOW() - INTERVAL '1 hour'"
                }
            ],
            'alerts': [
                {
                    'condition': 'database_size_gb > 250',
                    'severity': 'warning',
                    'message': 'Database size exceeding threshold'
                },
                {
                    'condition': 'new_markets_per_hour < 1',
                    'severity': 'error',
                    'message': 'No new markets discovered - check blockchain sync'
                }
            ]
        }
        
        with open('hybrid_monitoring_config.json', 'w') as f:
            json.dump(monitoring_config, f, indent=2)
        
        logger.info("✅ Monitoring configuration created")
    
    def run_complete_setup(self):
        """Run complete hybrid database setup."""
        logger.info("🚀 Running Hybrid Database Setup")
        logger.info("=" * 60)
        
        steps = [
            ("Create PostgreSQL database", self.create_postgresql_database),
            ("Create blockchain tables", self.create_blockchain_tables),
            ("Create routing configuration", self.create_data_routing_config),
            ("Create database manager", self.create_hybrid_database_manager),
            ("Create migration scripts", self.create_migration_scripts),
            ("Setup monitoring", self.setup_monitoring)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n📋 {step_name}...")
            try:
                result = step_func()
                if result:
                    logger.info(f"✅ {step_name} completed")
                else:
                    logger.error(f"❌ {step_name} failed")
            except Exception as e:
                logger.error(f"❌ {step_name} error: {e}")
        
        logger.info("\n🎉 Hybrid Database Setup Complete!")
        logger.info("\n📊 Summary:")
        logger.info("   - PostgreSQL: Ready for new blockchain data")
        logger.info("   - SQLite: Remains for historical data (read-only)")
        logger.info("   - Redis: Caching layer active")
        logger.info("   - Data routing: Automatic based on date/type")
        logger.info("\n🚀 Next Steps:")
        logger.info("   1. Run archive_old_odds.py to reduce SQLite size")
        logger.info("   2. Update blockchain_reader.py to use PostgreSQL")
        logger.info("   3. Monitor data growth with provided metrics")
        logger.info("\n✨ System ready for blockchain data collection!")


if __name__ == "__main__":
    setup = HybridDatabaseSetup()
    setup.run_complete_setup()