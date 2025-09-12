#!/usr/bin/env python3
"""Verify paper trading migration was successful."""

from database_v2 import db_manager
from paper_trading_models_v2 import *
from sqlalchemy import func

def verify_migration():
    """Verify the migration data."""
    with db_manager.get_db_session() as db:
        # Check session
        session_count = db.query(func.count(PaperTradingSession.session_id)).scalar()
        print(f"Sessions migrated: {session_count}")
        
        # Check positions
        total_positions = db.query(func.count(PaperTradingPosition.position_id)).scalar()
        open_positions = db.query(func.count(PaperTradingPosition.position_id)).filter(
            PaperTradingPosition.status == PositionStatus.OPEN
        ).scalar()
        closed_positions = db.query(func.count(PaperTradingPosition.position_id)).filter(
            PaperTradingPosition.status == PositionStatus.CLOSED
        ).scalar()
        
        print(f"\nPositions:")
        print(f"  Total: {total_positions}")
        print(f"  Open: {open_positions}")
        print(f"  Closed: {closed_positions}")
        
        # Check trades
        trade_count = db.query(func.count(PaperTradingTrade.trade_id)).scalar()
        print(f"\nTrades: {trade_count}")
        
        # Check snapshots
        snapshot_count = db.query(func.count(PaperTradingSnapshot.snapshot_id)).scalar()
        print(f"Snapshots: {snapshot_count}")
        
        # Check market names
        market_count = db.query(func.count(MarketName.market_id)).scalar()
        print(f"Market names: {market_count}")
        
        # Sample some data
        print("\nSample closed positions:")
        positions = db.query(PaperTradingPosition).filter(
            PaperTradingPosition.status == PositionStatus.CLOSED
        ).limit(5).all()
        
        for pos in positions:
            market = db.query(MarketName).filter_by(market_id=pos.market_id).first()
            print(f"  - {market.market_name if market else 'Unknown'}: "
                  f"{pos.outcome.value} @ {pos.avg_odds:.3f}, "
                  f"P&L: ${pos.pnl:.2f}")

if __name__ == "__main__":
    verify_migration()