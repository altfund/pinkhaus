#!/usr/bin/env python3
"""Calculate actual maximum drawdown from portfolio history."""

from database_v2 import db_manager
from paper_trading_models_v2 import PaperTradingSession, PaperTradingSnapshot, SessionStatus
from sqlalchemy import func, desc, and_
from datetime import datetime, timedelta, timezone
import numpy as np


def calculate_max_drawdown(session_id: str) -> dict:
    """
    Calculate maximum drawdown from portfolio snapshots.
    
    Returns:
        dict with 'max_drawdown' (decimal), 'max_drawdown_pct' (percentage),
        'peak_value', 'trough_value', 'peak_time', 'trough_time'
    """
    with db_manager.get_db_session() as db:
        # Get all snapshots for the session ordered by time
        snapshots = db.query(
            PaperTradingSnapshot.snapshot_time,
            PaperTradingSnapshot.portfolio_value
        ).filter(
            PaperTradingSnapshot.session_id == session_id
        ).order_by(
            PaperTradingSnapshot.snapshot_time
        ).all()
        
        if not snapshots or len(snapshots) < 2:
            # Not enough data
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'peak_value': None,
                'trough_value': None,
                'peak_time': None,
                'trough_time': None
            }
        
        # Convert to numpy arrays for efficient calculation
        times = [s[0] for s in snapshots]
        values = np.array([float(s[1]) for s in snapshots])
        
        # Calculate running maximum (peak values)
        peaks = np.maximum.accumulate(values)
        
        # Calculate drawdowns from peaks
        drawdowns = (peaks - values) / peaks
        
        # Find maximum drawdown
        max_dd_idx = np.argmax(drawdowns)
        max_dd = drawdowns[max_dd_idx]
        
        # Find the peak that led to this drawdown
        # Look backwards from max_dd_idx to find where the peak was
        peak_idx = max_dd_idx
        peak_value = peaks[max_dd_idx]
        for i in range(max_dd_idx, -1, -1):
            if values[i] == peak_value:
                peak_idx = i
                break
        
        return {
            'max_drawdown': float(max_dd),
            'max_drawdown_pct': float(max_dd * 100),
            'peak_value': float(values[peak_idx]),
            'trough_value': float(values[max_dd_idx]),
            'peak_time': times[peak_idx],
            'trough_time': times[max_dd_idx],
            'current_drawdown': float(drawdowns[-1]) if len(drawdowns) > 0 else 0.0,
            'current_drawdown_pct': float(drawdowns[-1] * 100) if len(drawdowns) > 0 else 0.0
        }


def calculate_rolling_metrics(session_id: str, window_days: int = 30) -> dict:
    """
    Calculate rolling performance metrics including drawdown.
    """
    with db_manager.get_db_session() as db:
        # Get snapshots for the window
        since = datetime.now(timezone.utc) - timedelta(days=window_days)
        
        snapshots = db.query(
            PaperTradingSnapshot.snapshot_time,
            PaperTradingSnapshot.portfolio_value,
            PaperTradingSnapshot.daily_pnl
        ).filter(
            and_(
                PaperTradingSnapshot.session_id == session_id,
                PaperTradingSnapshot.snapshot_time >= since
            )
        ).order_by(
            PaperTradingSnapshot.snapshot_time
        ).all()
        
        if not snapshots:
            return {
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'win_days': 0,
                'loss_days': 0
            }
        
        # Calculate daily returns
        values = np.array([float(s[1]) for s in snapshots])
        if len(values) > 1:
            returns = np.diff(values) / values[:-1]
            
            # Volatility (annualized)
            volatility = np.std(returns) * np.sqrt(252)  # Assuming 252 trading days
            
            # Sharpe ratio (assuming 0% risk-free rate)
            avg_return = np.mean(returns)
            sharpe = (avg_return * 252) / volatility if volatility > 0 else 0
            
            # Win/loss days
            daily_pnls = [float(s[2]) for s in snapshots if s[2] is not None]
            win_days = sum(1 for pnl in daily_pnls if pnl > 0)
            loss_days = sum(1 for pnl in daily_pnls if pnl < 0)
        else:
            volatility = 0.0
            sharpe = 0.0
            win_days = 0
            loss_days = 0
        
        # Calculate drawdown for this period
        dd_info = calculate_max_drawdown_from_values(values, [s[0] for s in snapshots])
        
        return {
            'max_drawdown': dd_info['max_drawdown'],
            'max_drawdown_pct': dd_info['max_drawdown_pct'],
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'win_days': win_days,
            'loss_days': loss_days,
            'total_days': len(snapshots)
        }


def calculate_max_drawdown_from_values(values, times=None):
    """Helper to calculate drawdown from value array."""
    if len(values) < 2:
        return {
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'peak_value': None,
            'trough_value': None,
            'peak_time': None,
            'trough_time': None
        }
    
    values = np.array(values)
    peaks = np.maximum.accumulate(values)
    drawdowns = (peaks - values) / peaks
    
    max_dd_idx = np.argmax(drawdowns)
    max_dd = drawdowns[max_dd_idx]
    
    # Find peak index
    peak_idx = max_dd_idx
    peak_value = peaks[max_dd_idx]
    for i in range(max_dd_idx, -1, -1):
        if values[i] == peak_value:
            peak_idx = i
            break
    
    return {
        'max_drawdown': float(max_dd),
        'max_drawdown_pct': float(max_dd * 100),
        'peak_value': float(values[peak_idx]),
        'trough_value': float(values[max_dd_idx]),
        'peak_time': times[peak_idx] if times else None,
        'trough_time': times[max_dd_idx] if times else None
    }


if __name__ == "__main__":
    # Test the calculation
    with db_manager.get_db_session() as db:
        # Get active session
        session = db.query(PaperTradingSession).filter_by(
            status=SessionStatus.ACTIVE
        ).first()
        
        if session:
            print(f"Session {session.session_id}:")
            print(f"Initial bankroll: ${session.initial_bankroll:.2f}")
            
            # Calculate max drawdown
            dd_info = calculate_max_drawdown(session.session_id)
            print(f"\nMax Drawdown: {dd_info['max_drawdown_pct']:.2f}%")
            if dd_info['peak_value']:
                print(f"Peak: ${dd_info['peak_value']:.2f} at {dd_info['peak_time']}")
                print(f"Trough: ${dd_info['trough_value']:.2f} at {dd_info['trough_time']}")
            print(f"Current drawdown: {dd_info['current_drawdown_pct']:.2f}%")
            
            # Calculate 30-day metrics
            print("\n30-Day Rolling Metrics:")
            metrics = calculate_rolling_metrics(session.session_id, 30)
            print(f"Max DD: {metrics['max_drawdown_pct']:.2f}%")
            print(f"Volatility: {metrics['volatility']*100:.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Win Days: {metrics['win_days']}/{metrics['total_days']}")
        else:
            print("No active session found")