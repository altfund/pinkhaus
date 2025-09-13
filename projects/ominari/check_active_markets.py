#!/usr/bin/env python3
"""Check active markets in the database."""

from database_v2 import db_manager
from models import Market
from datetime import datetime, timezone
from sqlalchemy import func

with db_manager.get_db_session() as db:
    # Total markets
    total = db.query(Market).count()
    
    # Active markets (not finished)
    active = db.query(Market).filter(
        Market.is_finished == False
    ).count()
    
    # Active with future dates
    future = db.query(Market).filter(
        Market.is_finished == False,
        Market.maturity_date > datetime.now(timezone.utc)
    ).count()
    
    # Active soccer
    soccer = db.query(Market).filter(
        Market.sport == 'Soccer',
        Market.is_finished == False,
        Market.maturity_date > datetime.now(timezone.utc)
    ).count()
    
    # By source
    by_source = db.query(
        Market.source,
        func.count(Market.source_id)
    ).filter(
        Market.is_finished == False
    ).group_by(Market.source).all()
    
    print(f"Total markets in DB: {total}")
    print(f"Active markets (not finished): {active}")
    print(f"Active with future dates: {future}")
    print(f"Active soccer matches: {soccer}")
    print("\nBy source:")
    for source, count in by_source:
        print(f"  {source}: {count}")