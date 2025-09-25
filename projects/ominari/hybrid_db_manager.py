#!/usr/bin/env python3
"""
Hybrid Database Manager
Falls back to SQLite if PostgreSQL is not available.
"""

import os
import logging
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class HybridDatabaseManager:
    """Database manager that can use either PostgreSQL or SQLite."""
    
    def __init__(self):
        self._engine = None
        self._session_factory = None
        self.db_type = None
        self._initialize()
    
    def _initialize(self):
        """Initialize database connection with fallback."""
        # First try PostgreSQL
        pg_config = {
            'host': os.getenv('PG_HOST', 'localhost'),
            'port': os.getenv('PG_PORT', '5432'),
            'user': os.getenv('PG_USER', 'ominari_user'),
            'password': os.getenv('PG_PASSWORD', 'ominari_2025_secure'),
            'database': os.getenv('PG_DB', 'ominari_production')
        }
        
        pg_url = f"postgresql://{pg_config['user']}:{pg_config['password']}@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
        
        try:
            # Try PostgreSQL first
            logger.info("Attempting to connect to PostgreSQL...")
            self._engine = create_engine(pg_url, pool_pre_ping=True, pool_size=10)
            # Test connection
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.db_type = 'postgresql'
            logger.info("✅ Connected to PostgreSQL")
            
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")
            logger.info("Falling back to SQLite...")
            
            # Fall back to SQLite
            db_path = 'sport_odds.db'
            sqlite_url = f'sqlite:///{db_path}'
            
            self._engine = create_engine(
                sqlite_url,
                connect_args={
                    'timeout': 30,
                    'check_same_thread': False
                },
                pool_size=20,
                max_overflow=30
            )
            
            # Enable WAL mode for better concurrency
            @event.listens_for(self._engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=10000")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=-64000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()
            
            self.db_type = 'sqlite'
            logger.info("✅ Connected to SQLite")
        
        # Create session factory
        self._session_factory = sessionmaker(bind=self._engine)
    
    @contextmanager
    def get_db_session(self):
        """Get a database session."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_engine(self):
        """Get the database engine."""
        return self._engine
    
    def get_db_type(self):
        """Get the type of database being used."""
        return self.db_type
    
    def is_postgresql(self):
        """Check if using PostgreSQL."""
        return self.db_type == 'postgresql'
    
    def is_sqlite(self):
        """Check if using SQLite."""
        return self.db_type == 'sqlite'

# Create global instance
hybrid_db_manager = HybridDatabaseManager()

# For compatibility with existing code
get_db_session = hybrid_db_manager.get_db_session
get_engine = hybrid_db_manager.get_engine