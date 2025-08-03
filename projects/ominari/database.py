#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database connection configuration with SQLite pragmas to avoid locking issues.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

DB_NAME = "sport_odds.db"
DB_URL = f"sqlite:///{DB_NAME}"

# Create engine with timeout and connection health checks
engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False, "timeout": 30},  # wait up to 30 seconds
    future=True,
)


import sqlite3


@event.listens_for(engine, "connect")
def _configure_sqlite(dbapi_con, connection_record):
    try:
        # enable WAL the *first* time, but don’t blow up if it’s locked
        dbapi_con.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.OperationalError:
        # if the DB is locked just move on — we’ll already be in WAL or it will be next time
        pass


# Listen for the `connect` event to apply pragmas
event.listen(engine, "connect", _configure_sqlite)

# Session factory
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
