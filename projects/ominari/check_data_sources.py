#!/usr/bin/env python3
"""Check data sources and activity levels."""

from database_v2 import db_manager
from models import Market, Odd
from datetime import datetime, timezone, timedelta
from sqlalchemy import func, distinct

def analyze_data_sources():
    """Analyze market data by source."""
    
    with db_manager.get_db_session() as db:
        # Check markets by source
        print("=== MARKET DATA BY SOURCE ===")
        sources = db.query(
            Market.source,
            func.count(Market.source_id).label('total'),
            func.sum(func.cast(Market.is_finished == False, int)).label('active')
        ).group_by(Market.source).all()
        
        for source, total, active in sources:
            print(f"\n{source}:")
            print(f"  Total markets: {total}")
            print(f"  Active markets: {active}")
        
        # Check recent market activity
        print("\n=== RECENT MARKET ACTIVITY (Last 24h) ===")
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        
        recent_markets = db.query(
            Market.source,
            func.count(Market.source_id).label('count')
        ).filter(
            Market.last_update > yesterday
        ).group_by(Market.source).all()
        
        for source, count in recent_markets:
            print(f"{source}: {count} markets updated")
        
        # Check odds volume
        print("\n=== ODDS DATA VOLUME ===")
        odds_stats = db.query(
            func.count(Odd.source_id).label('total_odds'),
            func.count(distinct(Odd.source_id)).label('unique_markets')
        ).first()
        
        print(f"Total odds records: {odds_stats.total_odds:,}")
        print(f"Unique markets with odds: {odds_stats.unique_markets}")
        
        # Check blockchain-specific data
        print("\n=== BLOCKCHAIN DATA ===")
        blockchain_markets = db.query(Market).filter(
            Market.source == 'blockchain'
        ).limit(5).all()
        
        print("Sample blockchain markets:")
        for market in blockchain_markets:
            print(f"  - {market.home_team} vs {market.away_team} ({market.sport})")
            print(f"    ID: {market.source_id}")
            print(f"    Status: {'Active' if not market.is_finished else 'Finished'}")

def check_paper_trading_data():
    """Check what data paper trading has access to."""
    
    with db_manager.get_db_session() as db:
        print("\n=== PAPER TRADING DATA ACCESS ===")
        
        # Get active soccer markets with odds
        active_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).limit(3).all()
        
        for market in active_markets:
            print(f"\nMarket: {market.home_team} vs {market.away_team}")
            print(f"  Source: {market.source}")
            print(f"  Kick-off: {market.maturity_date}")
            
            # Get latest odds
            odds = db.query(Odd).filter(
                Odd.source_id == market.source_id
            ).order_by(Odd.updated_at.desc()).limit(3).all()
            
            print(f"  Latest odds ({len(odds)} found):")
            for odd in odds[:3]:
                print(f"    - {odd.outcome}: {odd.decimal_odds} ({odd.bookmaker})")

def check_blockchain_tables():
    """Check blockchain-specific tables."""
    print("\n=== BLOCKCHAIN TABLES ===")
    
    # Check if blockchain_markets table exists
    from sqlalchemy import inspect, text
    
    with db_manager.get_db_session() as db:
        inspector = inspect(db.bind)
        tables = inspector.get_table_names()
        
        if 'blockchain_markets' in tables:
            # Check blockchain_markets table
            result = db.execute(text("SELECT COUNT(*) FROM blockchain_markets")).scalar()
            print(f"blockchain_markets table: {result} records")
            
            # Get sample
            sample = db.execute(text("""
                SELECT game_id, home_team, away_team, created_at 
                FROM blockchain_markets 
                ORDER BY created_at DESC 
                LIMIT 3
            """)).fetchall()
            
            print("Recent blockchain markets:")
            for row in sample:
                print(f"  - {row[1]} vs {row[2]} (ID: {row[0][:16]}...)")

if __name__ == "__main__":
    analyze_data_sources()
    check_paper_trading_data()
    check_blockchain_tables()