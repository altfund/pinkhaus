#!/usr/bin/env python3
"""
Debug migration issues
"""

from sqlalchemy import create_engine, text

# Test both connections
sqlite_engine = create_engine('sqlite:///sport_odds.db')
postgres_engine = create_engine('postgresql://ominari_user:ominari_2025_secure@localhost:5435/ominari_production')

print("Testing SQLite connection...")
with sqlite_engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM market WHERE source LIKE 'blockchain_%'")).scalar()
    print(f"SQLite blockchain markets: {result}")
    
    # Show sample data
    sample = conn.execute(text("SELECT source_id, source, sport, home_team, away_team FROM market WHERE source LIKE 'blockchain_%' LIMIT 5")).fetchall()
    print("Sample SQLite data:")
    for row in sample:
        print(f"  {row}")

print("\nTesting PostgreSQL connection...")
with postgres_engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM market")).scalar()
    print(f"PostgreSQL total markets: {result}")
    
    # Show table structure
    tables = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")).fetchall()
    print("PostgreSQL tables:")
    for table in tables:
        print(f"  {table[0]}")

# Try a simple insert
print("\nTesting simple insert...")
try:
    with postgres_engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO market (source_id, source, sport, league_name, home_team, away_team, market_type, is_finished)
            VALUES ('test_123', 'test', 'Soccer', 'Test League', 'Home Team', 'Away Team', 'winner', false)
        """))
        conn.commit()
        
        count = conn.execute(text("SELECT COUNT(*) FROM market WHERE source_id = 'test_123'")).scalar()
        print(f"Test insert successful: {count} row inserted")
        
        # Clean up
        conn.execute(text("DELETE FROM market WHERE source_id = 'test_123'"))
        conn.commit()
except Exception as e:
    print(f"Insert failed: {e}")