#!/usr/bin/env python3
"""Simple daily change calculation using snapshots."""

from database_v2 import db_manager
from paper_trading_models_v2 import PaperTradingSession, PaperTradingSnapshot, SessionStatus
from datetime import datetime, timedelta, timezone
from sqlalchemy import func, and_, desc

def get_daily_change(session_id: str) -> dict:
    """
    Calculate actual daily change using snapshot history.
    Falls back to total change if no history available.
    """
    with db_manager.get_db_session() as db:
        # Get most recent snapshot
        latest = db.query(PaperTradingSnapshot).filter(
            PaperTradingSnapshot.session_id == session_id
        ).order_by(desc(PaperTradingSnapshot.snapshot_time)).first()
        
        if not latest:
            # No snapshots, return zero change
            return {'daily_change': 0.0, 'daily_change_pct': 0.0}
        
        current_value = float(latest.portfolio_value)
        
        # Get snapshot from ~24 hours ago
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        yesterday_snapshot = db.query(PaperTradingSnapshot).filter(
            and_(
                PaperTradingSnapshot.session_id == session_id,
                PaperTradingSnapshot.snapshot_time <= yesterday
            )
        ).order_by(desc(PaperTradingSnapshot.snapshot_time)).first()
        
        if yesterday_snapshot:
            # Calculate 24h change
            yesterday_value = float(yesterday_snapshot.portfolio_value)
            daily_change = current_value - yesterday_value
            daily_change_pct = (daily_change / yesterday_value * 100) if yesterday_value > 0 else 0
            
            return {
                'daily_change': daily_change,
                'daily_change_pct': daily_change_pct,
                'since_time': yesterday_snapshot.snapshot_time
            }
        else:
            # No 24h data, try start of today
            today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            today_first = db.query(PaperTradingSnapshot).filter(
                and_(
                    PaperTradingSnapshot.session_id == session_id,
                    PaperTradingSnapshot.snapshot_time >= today_start
                )
            ).order_by(PaperTradingSnapshot.snapshot_time).first()
            
            if today_first and today_first.snapshot_id != latest.snapshot_id:
                # Calculate intraday change
                start_value = float(today_first.portfolio_value)
                daily_change = current_value - start_value
                daily_change_pct = (daily_change / start_value * 100) if start_value > 0 else 0
                
                return {
                    'daily_change': daily_change,
                    'daily_change_pct': daily_change_pct,
                    'since_time': today_first.snapshot_time
                }
            else:
                # Use daily_pnl from snapshot if available
                if latest.daily_pnl:
                    daily_change = float(latest.daily_pnl)
                    # Calculate percentage based on value 24h ago estimate
                    prev_value = current_value - daily_change
                    daily_change_pct = (daily_change / prev_value * 100) if prev_value > 0 else 0
                else:
                    # Fall back to total change
                    daily_change = float(latest.total_pnl) if latest.total_pnl else 0.0
                    session = db.query(PaperTradingSession).filter_by(session_id=session_id).first()
                    if session:
                        initial = float(session.initial_bankroll)
                        daily_change_pct = (daily_change / initial * 100) if initial > 0 else 0
                    else:
                        daily_change_pct = 0.0
                
                return {
                    'daily_change': daily_change,
                    'daily_change_pct': daily_change_pct
                }


if __name__ == "__main__":
    # Test the function
    with db_manager.get_db_session() as db:
        # Get active session
        session = db.query(PaperTradingSession).filter_by(
            status=SessionStatus.ACTIVE
        ).first()
        
        if session:
            print(f"Session {session.session_id}:")
            
            # Get latest snapshot
            latest = db.query(PaperTradingSnapshot).filter(
                PaperTradingSnapshot.session_id == session.session_id
            ).order_by(desc(PaperTradingSnapshot.snapshot_time)).first()
            
            if latest:
                print(f"Latest portfolio value: ${latest.portfolio_value:.2f}")
                print(f"Total P&L: ${latest.total_pnl:.2f}")
            
            daily = get_daily_change(session.session_id)
            print(f"\nDaily change: ${daily['daily_change']:.2f} ({daily['daily_change_pct']:.2f}%)")
            if 'since_time' in daily:
                print(f"Since: {daily['since_time']}")
        else:
            print("No active session found")