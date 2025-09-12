#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe query tool for the large database.
Provides a CLI interface for common queries with built-in safety limits.
"""

import argparse
import sys
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd, Bet, BettingSession


def get_model_class(model_name: str):
    """Get model class from string name."""
    models = {
        'market': Market,
        'odd': Odd,
        'bet': Bet,
        'bettingsession': BettingSession,
    }
    return models.get(model_name.lower())


def safe_count(model_name: str, filter_str: str = None):
    """Safely count records with optional filter."""
    model = get_model_class(model_name)
    if not model:
        print(f"Unknown model: {model_name}")
        return
    
    with db_manager.get_db_session() as db:
        query = db.query(model)
        
        if filter_str:
            # Parse simple filter (field=value or field LIKE '%value%')
            if ' LIKE ' in filter_str:
                field, pattern = filter_str.split(' LIKE ')
                field = field.strip()
                pattern = pattern.strip().strip("'\"")
                query = query.filter(getattr(model, field).like(pattern))
            elif '=' in filter_str:
                field, value = filter_str.split('=')
                field = field.strip()
                value = value.strip().strip("'\"")
                query = query.filter(getattr(model, field) == value)
        
        count = query.count()
        print(f"Count of {model_name}: {count:,}")
        if filter_str:
            print(f"Filter: {filter_str}")


def safe_sample(model_name: str, limit: int = 10, filter_str: str = None):
    """Get a sample of records safely."""
    model = get_model_class(model_name)
    if not model:
        print(f"Unknown model: {model_name}")
        return
    
    with db_manager.get_db_session() as db:
        query = db.query(model)
        
        if filter_str:
            if ' LIKE ' in filter_str:
                field, pattern = filter_str.split(' LIKE ')
                field = field.strip()
                pattern = pattern.strip().strip("'\"")
                query = query.filter(getattr(model, field).like(pattern))
            elif '=' in filter_str:
                field, value = filter_str.split('=')
                field = field.strip()
                value = value.strip().strip("'\"")
                query = query.filter(getattr(model, field) == value)
        
        # Always limit results
        results = query.limit(limit).all()
        
        print(f"\nSample of {model_name} (limit={limit}):")
        if filter_str:
            print(f"Filter: {filter_str}")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result}")
            # Print first few attributes
            for attr in dir(result):
                if not attr.startswith('_') and hasattr(result, attr):
                    value = getattr(result, attr)
                    if not callable(value):
                        print(f"   {attr}: {value}")
                        if i > 5:  # Limit attributes shown
                            break


def check_indexes():
    """Check what indexes exist (using ORM metadata)."""
    print("\nDatabase Indexes:")
    print("-" * 80)
    
    from sqlalchemy import inspect
    
    with db_manager.get_db_session() as db:
        inspector = inspect(db.bind)
        
        for table_name in ['market', 'odd', 'bet', 'betting_session']:
            indexes = inspector.get_indexes(table_name)
            print(f"\nTable: {table_name}")
            if indexes:
                for idx in indexes:
                    print(f"  - {idx['name']}: {idx['column_names']}")
            else:
                print("  No indexes found")


def recent_activity(hours: int = 24):
    """Show recent database activity."""
    print(f"\nRecent Activity (last {hours} hours):")
    print("-" * 80)
    
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    with db_manager.get_db_session() as db:
        # Recent markets
        recent_markets = db.query(Market).filter(
            Market.updated_at >= cutoff
        ).limit(10).all()
        
        print(f"\nRecent markets: {len(recent_markets)}")
        for market in recent_markets[:3]:
            print(f"  - {market.home_team} vs {market.away_team} ({market.sport})")
        
        # Recent odds
        recent_odds = db.query(Odd).filter(
            Odd.updated_at >= cutoff
        ).limit(10).all()
        
        print(f"\nRecent odds: {len(recent_odds)}")
        for odd in recent_odds[:3]:
            print(f"  - {odd.source_id}: {odd.decimal_odds} ({odd.outcome})")


def database_summary():
    """Show database summary statistics."""
    print("\nDatabase Summary:")
    print("-" * 80)
    
    stats = db_manager.get_database_stats()
    
    print(f"Database size: {stats.get('size_bytes', 0) / 1e9:.2f} GB")
    print(f"WAL size: {stats.get('wal_size_bytes', 0) / 1e6:.2f} MB")
    
    if 'cache_hit_rate' in stats:
        print(f"Cache hit rate: {stats['cache_hit_rate'] * 100:.1f}%")
    
    with db_manager.get_db_session() as db:
        # Get approximate counts using a subquery with limit
        print("\nApproximate record counts (sampled):")
        
        for model_name, model in [('Markets', Market), ('Odds', Odd)]:
            # Instead of full count, estimate from a sample
            sample_size = 1000
            sample_count = db.query(model).limit(sample_size + 1).count()
            if sample_count > sample_size:
                print(f"  {model_name}: > {sample_size:,} (large table)")
            else:
                print(f"  {model_name}: {sample_count:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Safe database query tool for large SQLite database"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Count command
    count_parser = subparsers.add_parser('count', help='Count records')
    count_parser.add_argument('model', help='Model name (Market, Odd, etc.)')
    count_parser.add_argument('--filter', help='Filter condition (e.g., "sport=Soccer")')
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Get sample records')
    sample_parser.add_argument('model', help='Model name')
    sample_parser.add_argument('--limit', type=int, default=10, help='Number of records')
    sample_parser.add_argument('--filter', help='Filter condition')
    
    # Other commands
    subparsers.add_parser('indexes', help='Show database indexes')
    subparsers.add_parser('summary', help='Show database summary')
    
    recent_parser = subparsers.add_parser('recent', help='Show recent activity')
    recent_parser.add_argument('--hours', type=int, default=24, help='Hours to look back')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'count':
            safe_count(args.model, args.filter)
        elif args.command == 'sample':
            safe_sample(args.model, args.limit, args.filter)
        elif args.command == 'indexes':
            check_indexes()
        elif args.command == 'summary':
            database_summary()
        elif args.command == 'recent':
            recent_activity(args.hours)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()