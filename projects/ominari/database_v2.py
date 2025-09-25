#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQL Database Manager V2
Updated from SQLite to PostgreSQL with enhanced connection management.
"""

from sqlalchemy import create_engine, event, pool, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import OperationalError
from contextlib import contextmanager
import time
import logging
import os
from typing import Generator
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# PostgreSQL Configuration
PG_CONFIG = {
    'host': os.getenv('PG_HOST', 'localhost'),
    'port': os.getenv('PG_PORT', '5432'),
    'user': os.getenv('PG_USER', 'ominari_user'),
    'password': os.getenv('PG_PASSWORD', 'ominari_2025_secure'),
    'database': os.getenv('PG_DB', 'ominari_production')
}

DB_URL = f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['database']}"

# Connection pool configuration optimized for PostgreSQL
POOL_CONFIG = {
    "pool_size": 20,  # Number of connections to maintain
    "max_overflow": 30,  # Maximum overflow connections
    "pool_timeout": 30,  # Timeout for getting connection from pool
    "pool_recycle": 3600,  # Recycle connections after 1 hour
    "pool_pre_ping": True,  # Check connections before using
}


class DatabaseManager:
    """Enhanced PostgreSQL database manager with connection pooling."""

    def __init__(self, db_url: str = DB_URL):
        self.db_url = db_url
        self._engine = None
        self._session_factory = None
        self._scoped_session = None
        logger.info("DatabaseManager initialized for PostgreSQL")

    @property
    def engine(self):
        """Lazy initialization of database engine."""
        if self._engine is None:
            self._engine = create_engine(
                self.db_url,
                **POOL_CONFIG,
                echo=False,  # Set to True for SQL debugging
                future=True
            )

            # Configure PostgreSQL session settings
            @event.listens_for(self._engine, "connect")
            def set_postgresql_search_path(dbapi_connection, connection_record):
                with dbapi_connection.cursor() as cursor:
                    # Set search path to include ominari schema
                    cursor.execute("SET search_path TO ominari, public")
                    cursor.execute("SET statement_timeout = '300s'")
                    cursor.execute("SET work_mem = '32MB'")

            logger.info("PostgreSQL engine created with connection pooling")

        return self._engine
    
    @property
    def session_factory(self):
        """Get session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autoflush=False,
                autocommit=False,
                expire_on_commit=False
            )
        return self._session_factory

    @property
    def scoped_session(self):
        """Get scoped session for thread-safe operations."""
        if self._scoped_session is None:
            self._scoped_session = scoped_session(self.session_factory)
        return self._scoped_session
    
    @contextmanager
    def get_db_session(self, retries: int = 3, retry_delay: float = 1.0) -> Generator:
        """
        Get database session with automatic retry and proper error handling.

        Args:
            retries: Number of retry attempts
            retry_delay: Delay between retries in seconds

        Yields:
            SQLAlchemy session object
        """
        session = None
        last_exception = None

        for attempt in range(retries + 1):
            try:
                session = self.session_factory()

                # Test the connection with a simple query
                session.execute(text("SELECT 1"))

                yield session
                return

            except OperationalError as e:
                last_exception = e
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")

                if session:
                    try:
                        session.rollback()
                        session.close()
                    except Exception:
                        pass
                    session = None

                if attempt < retries:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    logger.error(f"All database connection attempts failed")
                    raise last_exception

            except Exception as e:
                logger.error(f"Unexpected database error: {e}")
                if session:
                    try:
                        session.rollback()
                        session.close()
                    except Exception:
                        pass
                raise e

            finally:
                if session:
                    try:
                        session.close()
                    except Exception as e:
                        logger.warning(f"Error closing session: {e}")
    
    def test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            with self.get_db_session() as session:
                result = session.execute(text("SELECT COUNT(*) FROM ominari.markets_normalized LIMIT 1"))
                count = result.scalar()
                logger.info(f"Connection test successful - found {count:,} markets")
                return True

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_database_stats(self) -> dict:
        """Get comprehensive database statistics."""
        try:
            with self.get_db_session() as session:
                # Database size
                size_result = session.execute(text("""
                    SELECT pg_size_pretty(pg_database_size('ominari_production'))
                """))
                db_size = size_result.scalar()

                # Table counts
                tables = ['markets_normalized', 'lu_teams', 'lu_sports', 'lu_sources']
                counts = {}

                for table in tables:
                    count_result = session.execute(text(f"""
                        SELECT COUNT(*) FROM ominari.{table}
                    """))
                    counts[table] = count_result.scalar()

                return {
                    'database_size': db_size,
                    'table_counts': counts,
                    'connection_pool': {
                        'pool_size': POOL_CONFIG['pool_size'],
                        'max_overflow': POOL_CONFIG['max_overflow']
                    }
                }

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

    def close_all_connections(self):
        """Close all database connections."""
        try:
            if self._scoped_session:
                self._scoped_session.remove()

            if self._engine:
                self._engine.dispose()

            logger.info("All database connections closed")

        except Exception as e:
            logger.error(f"Error closing connections: {e}")


# Global database manager instance
db_manager = DatabaseManager()


def get_db():
    """Compatibility function for getting database session."""
    return db_manager.get_db_session()


# Compatibility aliases for existing code
SessionLocal = db_manager.session_factory
engine = db_manager.engine


if __name__ == "__main__":
    """Test the PostgreSQL database manager."""
    logging.basicConfig(level=logging.INFO)

    print("Testing PostgreSQL Database Manager V2...")

    if db_manager.test_connection():
        print("✅ Database connection successful!")

        stats = db_manager.get_database_stats()
        if stats:
            print(f"📊 Database size: {stats['database_size']}")
            print("📋 Table counts:")
            for table, count in stats['table_counts'].items():
                print(f"  {table}: {count:,}")
    else:
        print("❌ Database connection failed!")