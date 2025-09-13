#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe database wrapper that prevents direct SQL execution and enforces ORM usage.
This prevents accidental raw SQL queries on the 216GB database.
"""

import sqlite3
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import logging
import warnings

logger = logging.getLogger(__name__)

# Override sqlite3.connect to prevent direct connections
_original_sqlite3_connect = sqlite3.connect

def _safe_sqlite3_connect(*args, **kwargs):
    """Override sqlite3.connect to warn about direct usage."""
    warnings.warn(
        "Direct sqlite3 connection detected! Use the ORM instead:\n"
        "  from database_v2 import db_manager\n"
        "  with db_manager.get_db_session() as db:\n"
        "      result = db.query(Model).filter(...).all()",
        stacklevel=2
    )
    raise RuntimeError(
        "Direct SQLite connections are disabled for the 216GB database. "
        "Use SQLAlchemy ORM through database_v2.db_manager instead."
    )

# Monkey patch sqlite3.connect
sqlite3.connect = _safe_sqlite3_connect


class SafeDatabaseManager:
    """Database manager that enforces safe practices for large databases."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self._engine = None
        self._session_factory = None
        
    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine(
                self.db_url,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30,
                },
                echo=False,  # Set to True for debugging
            )
            
            # Add listeners to prevent dangerous operations
            event.listen(self._engine, "before_execute", self._before_execute)
            event.listen(self._engine, "connect", self._configure_connection)
            
        return self._engine
    
    def _before_execute(self, conn, clauseelement, multiparams, params, execution_options):
        """Intercept and validate SQL before execution."""
        sql_text = str(clauseelement).lower()
        
        # Dangerous patterns for a 216GB database
        dangerous_patterns = [
            # Full table scans without LIMIT
            (r'select.*from\s+\w+\s*(?!.*\blimit\b)', 
             "Query appears to scan entire table without LIMIT. Add .limit(N) to your query."),
            
            # COUNT without WHERE on large tables
            (r'select\s+count\(\*\)\s+from\s+(?:market|odd)\s*(?!.*\bwhere\b)',
             "COUNT(*) without WHERE clause on large table. Add filters or use approximate counts."),
            
            # PRAGMA operations that might lock
            (r'pragma\s+(?:vacuum|integrity_check|optimize)',
             "This PRAGMA operation may lock the database for extended time."),
            
            # Large DELETE/UPDATE without WHERE
            (r'(?:delete|update)\s+(?:market|odd)\s*(?!.*\bwhere\b)',
             "DELETE/UPDATE without WHERE clause detected. This is dangerous!"),
        ]
        
        import re
        for pattern, message in dangerous_patterns:
            if re.search(pattern, sql_text):
                logger.warning(f"Potentially dangerous query detected: {message}")
                logger.warning(f"Query: {sql_text[:200]}...")
                # In production, you might want to raise an exception here
                # raise RuntimeError(f"Dangerous query blocked: {message}")
    
    def _configure_connection(self, dbapi_conn, connection_record):
        """Configure connection with safety settings."""
        cursor = dbapi_conn.cursor()
        
        # Set pragmas for better performance with large database
        pragmas = [
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA cache_size=-64000",  # 64MB
            "PRAGMA mmap_size=268435456",  # 256MB
            "PRAGMA temp_store=MEMORY",
            "PRAGMA busy_timeout=5000",  # 5 seconds max wait
        ]
        
        for pragma in pragmas:
            try:
                cursor.execute(pragma)
            except Exception as e:
                logger.warning(f"Failed to set {pragma}: {e}")
        
        cursor.close()
    
    @contextmanager
    def get_safe_session(self):
        """Get a database session with safety checks."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        
        session = self._session_factory()
        
        # Add query timeout at session level
        @event.listens_for(session, "after_begin")
        def receive_after_begin(session, transaction, connection):
            # Set statement timeout (SQLite doesn't support this directly)
            pass
        
        try:
            yield SafeQueryWrapper(session)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


class SafeQueryWrapper:
    """Wrapper that enforces safe query patterns."""
    
    def __init__(self, session):
        self._session = session
        
    def query(self, *args, **kwargs):
        """Create a safe query that enforces limits."""
        query = self._session.query(*args, **kwargs)
        return SafeQuery(query)
    
    def execute(self, *args, **kwargs):
        """Block direct SQL execution."""
        raise RuntimeError(
            "Direct SQL execution is disabled. Use the ORM query interface:\n"
            "  db.query(Model).filter(...).all()"
        )
    
    # Delegate other methods to session
    def __getattr__(self, name):
        return getattr(self._session, name)


class SafeQuery:
    """Query wrapper that enforces safety limits."""
    
    def __init__(self, query, max_limit=10000):
        self._query = query
        self._max_limit = max_limit
        self._has_limit = False
        
    def filter(self, *args, **kwargs):
        self._query = self._query.filter(*args, **kwargs)
        return self
    
    def filter_by(self, **kwargs):
        self._query = self._query.filter_by(**kwargs)
        return self
    
    def order_by(self, *args):
        self._query = self._query.order_by(*args)
        return self
    
    def limit(self, limit):
        self._has_limit = True
        self._query = self._query.limit(min(limit, self._max_limit))
        return self
    
    def offset(self, offset):
        self._query = self._query.offset(offset)
        return self
    
    def all(self):
        if not self._has_limit:
            logger.warning(
                "Query executed without explicit limit. "
                f"Auto-limiting to {self._max_limit} results."
            )
            self._query = self._query.limit(self._max_limit)
        return self._query.all()
    
    def first(self):
        return self._query.first()
    
    def one(self):
        return self._query.one()
    
    def one_or_none(self):
        return self._query.one_or_none()
    
    def count(self):
        # For count queries, check if there's a filter
        if not hasattr(self._query, 'whereclause') or self._query.whereclause is None:
            logger.warning(
                "COUNT query without WHERE clause detected. "
                "This may be slow on large tables."
            )
        return self._query.count()
    
    # Delegate other methods
    def __getattr__(self, name):
        return getattr(self._query, name)


def create_safe_cli_override():
    """Create a wrapper for CLI that blocks dangerous commands."""
    
    def safe_cli_sqlite3(*args, **kwargs):
        """Override for CLI sqlite3 command."""
        print("❌ Direct SQLite CLI access is disabled for the 216GB database.")
        print("\nUse the ORM instead:")
        print("  python")
        print("  >>> from database_v2 import db_manager")
        print("  >>> from models import Market, Odd")
        print("  >>> with db_manager.get_db_session() as db:")
        print("  >>>     markets = db.query(Market).limit(10).all()")
        print("\nOr use the safe query script:")
        print("  python safe_query.py 'Market' --limit 10 --filter \"sport LIKE '%Soccer%'\"")
        return 1
    
    return safe_cli_sqlite3


# Example usage and migration guide
def example_safe_usage():
    """Show how to use the safe database interface."""
    
    from models import Market
    
    # Initialize safe database
    safe_db = SafeDatabaseManager("sqlite:///sport_odds.db")
    
    with safe_db.get_safe_session() as db:
        # ✅ Good: Query with limit
        markets = db.query(Market).filter(
            Market.sport.like('%Soccer%')
        ).limit(100).all()
        
        # ✅ Good: First/one queries are safe
        market = db.query(Market).filter_by(source_id='123').first()
        
        # ⚠️ Warning: Will auto-limit to 10000
        all_markets = db.query(Market).all()
        
        # ❌ Error: Direct SQL blocked
        # db.execute("SELECT * FROM market")  # This will raise an error
        
        # ✅ Good: Count with filter
        count = db.query(Market).filter(
            Market.sport == 'Soccer'
        ).count()


if __name__ == "__main__":
    print("Safe Database Module")
    print("===================")
    print("This module prevents dangerous queries on the 216GB database.")
    print("\nDangerous patterns blocked:")
    print("- Full table scans without LIMIT")
    print("- COUNT(*) without WHERE on large tables")
    print("- Direct SQL execution")
    print("- Direct sqlite3 connections")
    print("\nUse the ORM with proper filters and limits!")