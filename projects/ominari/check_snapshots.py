#!/usr/bin/env python3
"""Check snapshot data."""

from database_v2 import db_manager
from paper_trading_models_v2 import PaperTradingSnapshot
from sqlalchemy import func

with db_manager.get_db_session() as db:
    count = db.query(func.count(PaperTradingSnapshot.snapshot_id)).scalar()
    print(f'Total snapshots: {count}')
    
    if count > 0:
        latest = db.query(PaperTradingSnapshot).order_by(
            PaperTradingSnapshot.snapshot_time.desc()
        ).first()
        print(f'Latest snapshot: {latest.snapshot_time}')
        print(f'Portfolio value: ${latest.portfolio_value}')