#!/usr/bin/env python3
"""Simple check of data sources."""

from database_v2 import db_manager
from models import Market
from datetime import datetime, timezone
from sqlalchemy import text

def check_data():
    with db_manager.get_db_session() as db:
        # Check total markets
        total = db.query(Market).count()
        active = db.query(Market).filter(Market.is_finished == False).count()
        
        print(f"Total markets: {total}")
        print(f"Active markets: {active}")
        
        # Check by source
        print("\nMarkets by source:")
        sources = db.execute(text("""
            SELECT source, COUNT(*) as total, 
                   SUM(CASE WHEN is_finished = 0 THEN 1 ELSE 0 END) as active
            FROM market
            GROUP BY source
        """)).fetchall()
        
        for source, total, active in sources:
            print(f"  {source}: {total} total, {active} active")
        
        # Check recent activity
        print("\nRecent market updates (last 24h):")
        result = db.execute(text("""
            SELECT source, COUNT(*) as count
            FROM market
            WHERE last_update > datetime('now', '-1 day')
            GROUP BY source
        """)).fetchall()
        
        for source, count in result:
            print(f"  {source}: {count} markets")
        
        # Check blockchain_markets table
        print("\nBlockchain markets table:")
        try:
            bc_count = db.execute(text("SELECT COUNT(*) FROM blockchain_markets")).scalar()
            print(f"  Total records: {bc_count}")
            
            # Recent blockchain markets
            recent = db.execute(text("""
                SELECT home_team, away_team, sport, created_at
                FROM blockchain_markets
                ORDER BY created_at DESC
                LIMIT 5
            """)).fetchall()
            
            print("  Recent blockchain markets:")
            for home, away, sport, created in recent:
                print(f"    - {home} vs {away} ({sport}) - {created}")
                
        except Exception as e:
            print(f"  Error: {e}")
        
        # Check active soccer markets for paper trading
        print("\nActive Soccer markets for paper trading:")
        soccer = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).limit(5).all()
        
        for market in soccer:
            print(f"  - {market.home_team} vs {market.away_team}")
            print(f"    Source: {market.source}")
            print(f"    Kick-off: {market.maturity_date}")

if __name__ == "__main__":
    check_data()