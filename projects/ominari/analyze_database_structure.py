#!/usr/bin/env python3
"""Analyze database structure and size optimization opportunities."""

import os
from sqlalchemy import inspect, text
from database_v2 import db_manager
from models import Base

def analyze_database():
    """Analyze database structure and storage usage."""
    print("DATABASE STRUCTURE ANALYSIS")
    print("=" * 80)
    
    # Get database file size
    db_file = "sport_odds.db"
    if os.path.exists(db_file):
        size_gb = os.path.getsize(db_file) / (1024**3)
        print(f"Database file size: {size_gb:.2f} GB")
    else:
        print("Database file not found")
    
    print("\nTABLE ANALYSIS:")
    print("-" * 80)
    
    with db_manager.get_db_session() as db:
        inspector = inspect(db.bind)
        
        # Analyze each table
        for table_name in inspector.get_table_names():
            print(f"\n{table_name}:")
            
            # Get columns
            columns = inspector.get_columns(table_name)
            print(f"  Columns: {len(columns)}")
            
            # Show column details
            for col in columns:
                nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
                print(f"    - {col['name']}: {col['type']} {nullable}")
            
            # Get indexes
            indexes = inspector.get_indexes(table_name)
            print(f"  Indexes: {len(indexes)}")
            for idx in indexes:
                print(f"    - {idx['name']}: {idx['column_names']}")
            
            # Try to get row count (safely)
            try:
                result = db.execute(text(f"SELECT COUNT(*) FROM {table_name} LIMIT 1"))
                count = result.scalar()
                print(f"  Row count: {count:,}")
            except:
                print(f"  Row count: Unable to determine")
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION OPPORTUNITIES:")
    print("-" * 80)
    
    print("\n1. DATA TYPE OPTIMIZATION:")
    print("   - Use INTEGER instead of FLOAT for whole numbers")
    print("   - Use SMALLINT for limited range values (0-65535)")
    print("   - Use BOOLEAN instead of String for true/false")
    print("   - Use proper length limits on VARCHAR columns")
    
    print("\n2. REDUNDANT DATA:")
    print("   - Normalize repeated strings (team names, leagues)")
    print("   - Remove duplicate columns across tables")
    print("   - Use foreign keys instead of copying data")
    
    print("\n3. COMPRESSION:")
    print("   - Enable compression for large text fields")
    print("   - Archive old data to separate tables")
    print("   - Use JSON columns for variable schema data")
    
    print("\n4. PAPER TRADING TABLES NEEDED:")
    print("   - paper_trading_sessions")
    print("   - paper_trading_positions") 
    print("   - paper_trading_trades")
    print("   - paper_trading_performance")

if __name__ == "__main__":
    analyze_database()