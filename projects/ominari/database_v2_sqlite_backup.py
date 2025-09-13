#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced database connection configuration with better locking prevention.
"""

from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import OperationalError
from contextlib import contextmanager
import sqlite3
import time
import logging
import os
from typing import Generator
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

DB_NAME = "sport_odds.db"
DB_URL = f"sqlite:///{DB_NAME}"

# Enhanced SQLite configuration for large databases
SQLITE_PRAGMAS = {
    "journal_mode": "WAL",
    "cache_size": -64000,  # 64MB cache
    "synchronous": "NORMAL",  # Balance between safety and speed
    "temp_store": "MEMORY",
    "mmap_size": 268435456,  # 256MB memory-mapped I/O
    "busy_timeout": 30000,  # 30 seconds
    "wal_autocheckpoint": 1000,  # Checkpoint every 1000 pages
    "optimize": None,  # Run ANALYZE periodically
}

# Connection pool configuration
POOL_CONFIG = {
    "pool_size": 5,  # Number of connections to maintain
    "max_overflow": 10,  # Maximum overflow connections
    "pool_timeout": 30,  # Timeout for getting connection from pool
    "pool_recycle": 3600,  # Recycle connections after 1 hour
    "pool_pre_ping": True,  # Check connections before using
}


class DatabaseManager:
    """Enhanced database manager with better locking prevention."""
    
    def __init__(self, db_url: str = DB_URL):
        self.db_url = db_url
        self._engine = None
        self._session_factory = None
        self._scoped_session = None
        self.last_optimize = None
        
    @property
    def engine(self):
        if self._engine is None:
            if "sqlite" in self.db_url:
                # SQLite with StaticPool doesn't support pool parameters
                self._engine = create_engine(
                    self.db_url,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": 30,
                        "isolation_level": None,  # Use SQLite's default autocommit
                    },
                    poolclass=pool.StaticPool,
                    pool_pre_ping=True,
                    future=True,
                )
            else:
                # Other databases support full pool configuration
                self._engine = create_engine(
                    self.db_url,
                    poolclass=pool.QueuePool,
                    **POOL_CONFIG,
                    future=True,
                )
            
            # Configure SQLite on each connection
            event.listen(self._engine, "connect", self._configure_sqlite)
            
        return self._engine
    
    @property
    def session_factory(self):
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autoflush=False,
                autocommit=False,
                expire_on_commit=False,  # Don't expire objects after commit
            )
        return self._session_factory
    
    @property
    def scoped_session(self):
        if self._scoped_session is None:
            self._scoped_session = scoped_session(self.session_factory)
        return self._scoped_session
    
    def _configure_sqlite(self, dbapi_con, connection_record):
        """Configure SQLite connection with optimal settings."""
        cursor = dbapi_con.cursor()
        
        for pragma, value in SQLITE_PRAGMAS.items():
            if value is not None:
                try:
                    cursor.execute(f"PRAGMA {pragma}={value};")
                except sqlite3.OperationalError as e:
                    logger.warning(f"Failed to set PRAGMA {pragma}: {e}")
            elif pragma == "optimize":
                # Run ANALYZE periodically
                if self.last_optimize is None or \
                   datetime.now() - self.last_optimize > timedelta(hours=1):
                    try:
                        cursor.execute("PRAGMA optimize;")
                        self.last_optimize = datetime.now()
                    except sqlite3.OperationalError:
                        pass
        
        cursor.close()
    
    @contextmanager
    def get_db_session(self, retries: int = 3, retry_delay: float = 0.5) -> Generator:
        """Get a database session with retry logic."""
        session = None
        last_error = None
        
        for attempt in range(retries):
            try:
                session = self.session_factory()
                yield session
                session.commit()
                return
            except OperationalError as e:
                last_error = e
                if session:
                    session.rollback()
                if "database is locked" in str(e):
                    logger.warning(f"Database locked, attempt {attempt + 1}/{retries}")
                    if attempt < retries - 1:
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise
            except Exception:
                if session:
                    session.rollback()
                raise
            finally:
                if session:
                    session.close()
        
        # If we get here, all retries failed
        raise OperationalError(f"Database locked after {retries} attempts", None, None) from last_error
    
    def execute_with_retry(self, func, *args, retries: int = 3, **kwargs):
        """Execute a function with database retry logic."""
        last_error = None
        
        for attempt in range(retries):
            try:
                with self.get_db_session() as session:
                    return func(session, *args, **kwargs)
            except OperationalError as e:
                last_error = e
                if "database is locked" not in str(e) or attempt == retries - 1:
                    raise
                logger.warning(f"Retrying after database lock, attempt {attempt + 1}/{retries}")
                time.sleep(0.5 * (2 ** attempt))
        
        raise last_error
    
    def vacuum_wal(self):
        """Force a WAL checkpoint to reduce WAL file size."""
        try:
            with self.engine.connect() as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                logger.info("WAL checkpoint completed")
        except Exception as e:
            logger.error(f"Failed to checkpoint WAL: {e}")
    
    def optimize_database(self):
        """Run database optimization."""
        try:
            with self.engine.connect() as conn:
                conn.execute("PRAGMA optimize;")
                conn.execute("ANALYZE;")
                logger.info("Database optimization completed")
        except Exception as e:
            logger.error(f"Failed to optimize database: {e}")
    
    def get_database_stats(self) -> dict:
        """Get database statistics."""
        stats = {}
        try:
            with self.engine.connect() as conn:
                # Database size
                result = conn.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size();")
                stats['size_bytes'] = result.scalar()
                
                # WAL size
                wal_path = f"{DB_NAME}-wal"
                if os.path.exists(wal_path):
                    stats['wal_size_bytes'] = os.path.getsize(wal_path)
                else:
                    stats['wal_size_bytes'] = 0
                
                # Cache stats
                result = conn.execute("SELECT * FROM pragma_cache_stats;")
                cache_stats = result.fetchone()
                if cache_stats:
                    stats['cache_hits'] = cache_stats[0]
                    stats['cache_misses'] = cache_stats[1]
                    stats['cache_hit_rate'] = cache_stats[0] / (cache_stats[0] + cache_stats[1]) if cache_stats[1] > 0 else 1.0
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
        
        return stats


# Global database manager instance
db_manager = DatabaseManager()

# Backward compatibility
engine = db_manager.engine
SessionLocal = db_manager.session_factory
get_db = db_manager.get_db_session


# Connection pool monitoring
@event.listens_for(engine, "connect")
def receive_connect(dbapi_connection, connection_record):
    """Log when a new connection is created."""
    logger.debug("New database connection created")


@event.listens_for(engine, "close")
def receive_close(dbapi_connection, connection_record):
    """Log when a connection is closed."""
    logger.debug("Database connection closed")


# Helper functions for common patterns
def with_db_session(func):
    """Decorator to automatically handle database sessions."""
    def wrapper(*args, **kwargs):
        with db_manager.get_db_session() as session:
            return func(session, *args, **kwargs)
    return wrapper


def bulk_insert_with_retry(model_class, records, chunk_size=1000):
    """Bulk insert records with chunking and retry logic."""
    total_inserted = 0
    
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        
        def insert_chunk(session):
            session.bulk_insert_mappings(model_class, chunk)
            return len(chunk)
        
        inserted = db_manager.execute_with_retry(insert_chunk)
        total_inserted += inserted
        logger.info(f"Inserted {inserted} records ({total_inserted}/{len(records)})")
    
    return total_inserted