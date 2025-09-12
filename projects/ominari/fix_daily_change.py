#!/usr/bin/env python3
"""Fix to calculate actual daily P&L instead of total P&L from inception."""

from database_v2 import db_manager
from paper_trading_models_v2 import PaperTradingSession, PaperTradingSnapshot, PaperTradingPosition, PositionStatus
from datetime import datetime, timedelta, timezone
from sqlalchemy import func, and_

def get_daily_change(session_id: int) -> dict:
    """
    Calculate actual daily change by comparing current portfolio value 
    with value 24 hours ago or start of current trading day.
    """
    with db_manager.get_db_session() as db:
        # Get current session
        session = db.query(PaperTradingSession).filter_by(
            session_id=session_id
        ).first()
        
        if not session:
            return {'daily_change': 0.0, 'daily_change_pct': 0.0}
        
        # Calculate current portfolio value
        current_bankroll = float(db.query(
            func.coalesce(
                session.initial_bankroll + 
                func.sum(PaperTradingPosition.realized_pnl),
                session.initial_bankroll
            )
        ).filter(
            PaperTradingPosition.session_id == session.session_id
        ).scalar() or session.initial_bankroll)
        
        # Get active positions value
        active_positions_value = float(db.query(
            func.coalesce(func.sum(PaperTradingPosition.current_value), 0)
        ).filter(
            and_(
                PaperTradingPosition.session_id == session.session_id,
                PaperTradingPosition.status == PositionStatus.ACTIVE
            )
        ).scalar() or 0)
        
        current_value = current_bankroll + active_positions_value
        
        # Get value from 24 hours ago
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        
        # Try to get snapshot from ~24 hours ago
        yesterday_snapshot = db.query(PaperTradingSnapshot).filter(
            and_(
                PaperTradingSnapshot.session_id == session_id,
                PaperTradingSnapshot.snapshot_time <= yesterday
            )
        ).order_by(PaperTradingSnapshot.snapshot_time.desc()).first()
        
        if yesterday_snapshot:
            # We have historical data
            yesterday_value = float(yesterday_snapshot.portfolio_value)
            daily_change = current_value - yesterday_value
            daily_change_pct = (daily_change / yesterday_value * 100) if yesterday_value > 0 else 0
            
            return {
                'daily_change': daily_change,
                'daily_change_pct': daily_change_pct,
                'since_time': yesterday_snapshot.snapshot_time
            }
        else:
            # No historical data, use start of current day
            today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Get first snapshot of today
            today_first = db.query(PaperTradingSnapshot).filter(
                and_(
                    PaperTradingSnapshot.session_id == session_id,
                    PaperTradingSnapshot.snapshot_time >= today_start
                )
            ).order_by(PaperTradingSnapshot.snapshot_time.asc()).first()
            
            if today_first:
                start_value = float(today_first.portfolio_value)
                daily_change = current_value - start_value
                daily_change_pct = (daily_change / start_value * 100) if start_value > 0 else 0
                
                return {
                    'daily_change': daily_change,
                    'daily_change_pct': daily_change_pct,
                    'since_time': today_first.snapshot_time
                }
            else:
                # No data for today, show change from initial
                initial_value = float(session.initial_bankroll)
                total_change = current_value - initial_value
                change_pct = (total_change / initial_value * 100) if initial_value > 0 else 0
                
                return {
                    'daily_change': total_change,  # Fallback to total
                    'daily_change_pct': change_pct,
                    'since_time': session.created_at
                }


def create_snapshot(session_id: int):
    """Create a portfolio snapshot for tracking daily changes."""
    with db_manager.get_db_session() as db:
        session = db.query(PaperTradingSession).filter_by(
            session_id=session_id
        ).first()
        
        if not session:
            return None
        
        # Calculate daily P&L
        daily_data = get_daily_change(session_id)
        
        # Calculate current values
        current_bankroll = float(db.query(
            func.coalesce(
                session.initial_bankroll + 
                func.sum(PaperTradingPosition.realized_pnl),
                session.initial_bankroll
            )
        ).filter(
            PaperTradingPosition.session_id == session.session_id
        ).scalar() or session.initial_bankroll)
        
        active_positions_value = float(db.query(
            func.coalesce(func.sum(PaperTradingPosition.current_value), 0)
        ).filter(
            and_(
                PaperTradingPosition.session_id == session.session_id,
                PaperTradingPosition.status == PositionStatus.ACTIVE
            )
        ).scalar() or 0)
        
        portfolio_value = current_bankroll + active_positions_value
        
        # Get P&L values
        realized_pnl = float(db.query(
            func.coalesce(func.sum(PaperTradingPosition.realized_pnl), 0)
        ).filter(
            PaperTradingPosition.session_id == session.session_id
        ).scalar() or 0)
        
        unrealized_pnl = float(db.query(
            func.coalesce(func.sum(PaperTradingPosition.unrealized_pnl), 0)
        ).filter(
            and_(
                PaperTradingPosition.session_id == session.session_id,
                PaperTradingPosition.status == PositionStatus.ACTIVE
            )
        ).scalar() or 0)
        
        # Get bet stats
        winning_bets = db.query(func.count(PaperTradingPosition.position_id)).filter(
            and_(
                PaperTradingPosition.session_id == session.session_id,
                PaperTradingPosition.realized_pnl > 0
            )
        ).scalar() or 0
        
        losing_bets = db.query(func.count(PaperTradingPosition.position_id)).filter(
            and_(
                PaperTradingPosition.session_id == session.session_id,
                PaperTradingPosition.realized_pnl < 0
            )
        ).scalar() or 0
        
        total_bets = db.query(func.count(PaperTradingPosition.position_id)).filter(
            PaperTradingPosition.session_id == session.session_id
        ).scalar() or 0
        
        active_positions = db.query(func.count(PaperTradingPosition.position_id)).filter(
            and_(
                PaperTradingPosition.session_id == session.session_id,
                PaperTradingPosition.status == PositionStatus.ACTIVE
            )
        ).scalar() or 0
        
        total_volume = float(db.query(
            func.coalesce(func.sum(PaperTradingPosition.total_stake), 0)
        ).filter(
            PaperTradingPosition.session_id == session.session_id
        ).scalar() or 0)
        
        total_fees = float(db.query(
            func.coalesce(func.sum(PaperTradingPosition.safebox_fee_paid), 0)
        ).filter(
            PaperTradingPosition.session_id == session.session_id
        ).scalar() or 0)
        
        snapshot = PaperTradingSnapshot(
            session_id=session_id,
            snapshot_time=datetime.now(timezone.utc),
            bankroll=current_bankroll,
            portfolio_value=portfolio_value,
            total_pnl=realized_pnl + unrealized_pnl,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            daily_pnl=daily_data['daily_change'],
            winning_bets=winning_bets,
            losing_bets=losing_bets,
            total_bets=total_bets,
            active_positions=active_positions,
            total_volume=total_volume,
            total_fees=total_fees
        )
        
        db.add(snapshot)
        db.commit()
        
        return snapshot


if __name__ == "__main__":
    # Test the function
    from paper_trading_models_v2 import SessionStatus
    
    with db_manager.get_db_session() as db:
        # Get active session
        session = db.query(PaperTradingSession).filter_by(
            status=SessionStatus.ACTIVE
        ).first()
        
        if session:
            print(f"Session {session.session_id}:")
            # Calculate portfolio value
            current_bankroll = float(db.query(
                func.coalesce(
                    session.initial_bankroll + 
                    func.sum(PaperTradingPosition.realized_pnl),
                    session.initial_bankroll
                )
            ).filter(
                PaperTradingPosition.session_id == session.session_id
            ).scalar() or session.initial_bankroll)
            
            active_value = float(db.query(
                func.coalesce(func.sum(PaperTradingPosition.current_value), 0)
            ).filter(
                and_(
                    PaperTradingPosition.session_id == session.session_id,
                    PaperTradingPosition.status == PositionStatus.ACTIVE
                )
            ).scalar() or 0)
            
            portfolio_value = current_bankroll + active_value
            print(f"Current portfolio value: ${portfolio_value:.2f}")
            
            daily = get_daily_change(session.session_id)
            print(f"Daily change: ${daily['daily_change']:.2f} ({daily['daily_change_pct']:.2f}%)")
            if 'since_time' in daily:
                print(f"Since: {daily['since_time']}")
            
            # Create a snapshot
            snapshot = create_snapshot(session.session_id)
            if snapshot:
                print(f"\nCreated snapshot {snapshot.snapshot_id}")
        else:
            print("No active session found")