#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQL-only Database Configuration
Updated from SQLite to use normalized PostgreSQL schema with performance optimizations.
"""

import os
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import Engine
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# PostgreSQL Configuration
PG_CONFIG = {
    'host': os.getenv('PG_HOST', 'localhost'),
    'port': os.getenv('PG_PORT', '5435'),
    'user': os.getenv('PG_USER', 'ominari_user'),
    'password': os.getenv('PG_PASSWORD', 'ominari_2025_secure'),
    'database': os.getenv('PG_DB', 'ominari_production')
}

# Build PostgreSQL connection URL
DB_URL = f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['database']}"

# Create optimized PostgreSQL engine
engine = create_engine(
    DB_URL,
    echo=False,  # Set to True for SQL query debugging
    future=True,
    pool_size=20,  # Connection pool for concurrent access
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=3600,  # Recycle connections every hour
    # No special connect args needed for basic operation
)


@event.listens_for(Engine, "connect")
def _configure_postgresql(dbapi_connection, connection_record):
    """Configure PostgreSQL connection for optimal performance."""
    try:
        with dbapi_connection.cursor() as cursor:
            # Set PostgreSQL parameters for read performance
            cursor.execute("SET statement_timeout = '300s'")  # 5 minute timeout
            cursor.execute("SET work_mem = '32MB'")  # Increase work memory for queries

            # Set schema search path
            cursor.execute("SET search_path TO ominari, public")

        logger.debug("PostgreSQL connection configured successfully")
    except Exception as e:
        logger.warning(f"Could not configure PostgreSQL connection: {e}")


# Session factory with optimized settings
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False  # Keep objects accessible after commit
)


def get_db():
    """Dependency to get PostgreSQL database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection():
    """Test PostgreSQL connection and verify schema."""
    try:
        with engine.connect() as conn:
            # Test basic connectivity
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            logger.info(f"Connected to PostgreSQL: {version}")

            # Check schema exists
            result = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = 'ominari'
            """))
            table_count = result.scalar()
            logger.info(f"Found {table_count} tables in ominari schema")

            # Check key tables
            key_tables = ['markets_normalized', 'odds_normalized', 'lu_teams', 'lu_sports']
            for table in key_tables:
                result = conn.execute(text(f"""
                    SELECT COUNT(*) FROM ominari.{table}
                """))
                count = result.scalar()
                logger.info(f"Table {table}: {count:,} records")

            return True

    except Exception as e:
        logger.error(f"PostgreSQL connection test failed: {e}")
        return False


def get_database_stats():
    """Get comprehensive database statistics."""
    try:
        with engine.connect() as conn:
            # Database size
            result = conn.execute(text("""
                SELECT pg_size_pretty(pg_database_size('ominari_production')) as db_size
            """))
            db_size = result.scalar()

            # Table statistics
            result = conn.execute(text("""
                SELECT
                    schemaname,
                    relname as tablename,
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes,
                    n_live_tup as live_tuples,
                    n_dead_tup as dead_tuples
                FROM pg_stat_user_tables
                WHERE schemaname = 'ominari'
                ORDER BY n_live_tup DESC
            """))

            stats = {
                'database_size': db_size,
                'tables': []
            }

            for row in result:
                stats['tables'].append({
                    'schema': row.schemaname,
                    'table': row.tablename,
                    'live_tuples': row.live_tuples,
                    'dead_tuples': row.dead_tuples,
                    'inserts': row.inserts,
                    'updates': row.updates,
                    'deletes': row.deletes
                })

            return stats

    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return None


# Performance monitoring
def log_slow_queries():
    """Enable logging of slow queries for performance monitoring."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SET log_min_duration_statement = 1000"))  # Log queries > 1 second
            logger.info("Enabled slow query logging")
    except Exception as e:
        logger.warning(f"Could not enable slow query logging: {e}")


if __name__ == "__main__":
    """Test the PostgreSQL connection."""
    logging.basicConfig(level=logging.INFO)

    print("Testing PostgreSQL connection...")
    if test_connection():
        print("✅ PostgreSQL connection successful!")

        stats = get_database_stats()
        if stats:
            print(f"\n📊 Database Statistics:")
            print(f"Database Size: {stats['database_size']}")
            print(f"\nTable Statistics:")
            for table in stats['tables'][:10]:  # Top 10 tables
                print(f"  {table['table']}: {table['live_tuples']:,} records")
    else:
        print("❌ PostgreSQL connection failed!")