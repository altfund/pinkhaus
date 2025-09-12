#!/usr/bin/env python3
"""Drop paper trading tables for fresh migration."""

from database_v2 import db_manager
from sqlalchemy import text

def drop_tables():
    """Drop all paper trading tables."""
    with db_manager.get_db_session() as db:
        tables = [
            'paper_trading_trades',
            'paper_trading_snapshots', 
            'paper_trading_positions',
            'paper_trading_sessions',
            'market_names'
        ]
        
        for table in tables:
            try:
                db.execute(text(f"DROP TABLE IF EXISTS {table}"))
                print(f"Dropped table: {table}")
            except Exception as e:
                print(f"Error dropping {table}: {e}")
        
        db.commit()
        print("All paper trading tables dropped successfully")

if __name__ == "__main__":
    drop_tables()