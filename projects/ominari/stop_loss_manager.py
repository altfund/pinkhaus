#!/usr/bin/env python3
"""Robust Stop Loss System for Ominari Trading Platform

Provides:
1. Manual stop button functionality
2. Automatic valuation-based stop loss triggers
3. Position closing mechanisms
4. Performance tracking against expectations
"""

import os
import json
import logging
import psycopg2
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import threading
import time

logger = logging.getLogger(__name__)

class StopLossManager:
    """Manages stop loss operations and portfolio monitoring"""
    
    def __init__(self, session_manager, db_config: Optional[Dict] = None):
        self.session_manager = session_manager
        self.db_config = db_config or {
            'host': os.environ.get('PG_HOST', 'localhost'),
            'port': os.environ.get('PG_PORT', '5999'),
            'user': os.environ.get('PG_USER', 'ominari_user'),
            'password': os.environ.get('PG_PASSWORD', 'ominari_2025_secure'),
            'database': os.environ.get('PG_DB', 'ominari_production')
        }
        
        # Stop loss configuration
        self.stop_loss_config = {
            'drawdown_pct': 10,  # Stop if down 10% from peak
            'time_window_minutes': 30,  # Monitor over 30 minute windows
            'recovery_time_minutes': 60,  # Wait 60 minutes before reactivating
            'max_daily_loss_pct': 15,  # Maximum 15% daily loss
            'consecutive_losses': 5,  # Stop after 5 consecutive losses
            'volatility_threshold': 2.0,  # Stop if volatility exceeds 2x normal
        }
        
        # System state
        self.is_stopped = False
        self.stop_reason = None
        self.stop_time = None
        self.peak_value = 0
        self.daily_start_value = 0
        self.monitoring_thread = None
        self._monitoring_active = False
        
        # Performance tracking
        self.performance_history = []
        self.backtest_expectations = None
        
    def set_stop_loss_config(self, config: Dict[str, Any]):
        """Update stop loss configuration"""
        self.stop_loss_config.update(config)
        logger.info(f"Updated stop loss config: {self.stop_loss_config}")
        
    def set_backtest_expectations(self, expectations: Dict[str, Any]):
        """Set expected performance based on backtests"""
        self.backtest_expectations = expectations
        logger.info(f"Set backtest expectations: {expectations}")
        
    def manual_stop(self, session_id: str, reason: str = "Manual stop triggered") -> Dict[str, Any]:
        """Execute manual stop - close all positions immediately"""
        logger.warning(f"MANUAL STOP TRIGGERED: {reason}")
        
        try:
            # Set stop flag immediately
            self.is_stopped = True
            self.stop_reason = reason
            self.stop_time = datetime.now(timezone.utc)
            
            # Get all open positions
            positions = self._get_open_positions(session_id)
            closed_positions = []
            total_pnl = 0
            
            # Close each position
            for position in positions:
                result = self._close_position(session_id, position)
                if result['success']:
                    closed_positions.append(result)
                    total_pnl += result.get('pnl', 0)
                else:
                    logger.error(f"Failed to close position {position['bet_id']}: {result.get('error')}")
            
            # Update session status
            self._update_session_status(session_id, 'stopped', {
                'stop_time': self.stop_time.isoformat(),
                'stop_reason': reason,
                'positions_closed': len(closed_positions),
                'total_pnl': total_pnl
            })
            
            # Log stop event
            self._log_stop_event(session_id, reason, closed_positions)
            
            return {
                'success': True,
                'stop_time': self.stop_time.isoformat(),
                'reason': reason,
                'positions_closed': len(closed_positions),
                'total_positions': len(positions),
                'total_pnl': total_pnl,
                'closed_positions': closed_positions
            }
            
        except Exception as e:
            logger.error(f"Error in manual stop: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'stop_time': datetime.now(timezone.utc).isoformat()
            }
    
    def start_monitoring(self, session_id: str):
        """Start automatic monitoring for stop conditions"""
        if self._monitoring_active:
            logger.info("Monitoring already active")
            return
            
        self._monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_portfolio,
            args=(session_id,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Started portfolio monitoring for stop conditions")
        
    def stop_monitoring(self):
        """Stop automatic monitoring"""
        self._monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped portfolio monitoring")
        
    def _monitor_portfolio(self, session_id: str):
        """Background thread to monitor portfolio for stop conditions"""
        logger.info(f"Portfolio monitoring thread started for session {session_id}")
        
        check_interval = 10  # Check every 10 seconds
        history_window = []  # Store recent portfolio values
        
        while self._monitoring_active and not self.is_stopped:
            try:
                # Get current portfolio value
                session = self.session_manager.get_session(session_id)
                if not session:
                    logger.error(f"Session {session_id} not found")
                    break
                    
                current_value = session['portfolio_value']
                current_time = datetime.now(timezone.utc)
                
                # Update peak value
                if current_value > self.peak_value:
                    self.peak_value = current_value
                    
                # Add to history
                history_window.append({
                    'time': current_time,
                    'value': current_value,
                    'positions': session.get('open_positions', 0)
                })
                
                # Keep only recent history
                cutoff_time = current_time - timedelta(minutes=self.stop_loss_config['time_window_minutes'])
                history_window = [h for h in history_window if h['time'] > cutoff_time]
                
                # Check stop conditions
                stop_triggered, stop_reason = self._check_stop_conditions(
                    session, current_value, history_window
                )
                
                if stop_triggered:
                    logger.warning(f"AUTO STOP TRIGGERED: {stop_reason}")
                    self.manual_stop(session_id, f"Automatic stop: {stop_reason}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
                
            time.sleep(check_interval)
            
        logger.info("Portfolio monitoring thread ended")
        
    def _check_stop_conditions(self, session: Dict, current_value: float, 
                              history: List[Dict]) -> Tuple[bool, Optional[str]]:
        """Check if any stop conditions are met"""
        
        # 1. Drawdown from peak
        if self.peak_value > 0:
            drawdown_pct = ((self.peak_value - current_value) / self.peak_value) * 100
            if drawdown_pct >= self.stop_loss_config['drawdown_pct']:
                return True, f"Drawdown {drawdown_pct:.1f}% from peak"
                
        # 2. Daily loss limit
        if self.daily_start_value > 0:
            daily_loss_pct = ((self.daily_start_value - current_value) / self.daily_start_value) * 100
            if daily_loss_pct >= self.stop_loss_config['max_daily_loss_pct']:
                return True, f"Daily loss {daily_loss_pct:.1f}% exceeds limit"
                
        # 3. Rapid value decline in time window
        if len(history) >= 2:
            window_start_value = history[0]['value']
            window_loss_pct = ((window_start_value - current_value) / window_start_value) * 100
            if window_loss_pct >= self.stop_loss_config['drawdown_pct']:
                return True, f"Rapid {window_loss_pct:.1f}% loss in {self.stop_loss_config['time_window_minutes']} minutes"
                
        # 4. Consecutive losses
        recent_bets = self._get_recent_closed_bets(session['session_id'], limit=10)
        if self._count_consecutive_losses(recent_bets) >= self.stop_loss_config['consecutive_losses']:
            return True, f"Exceeded {self.stop_loss_config['consecutive_losses']} consecutive losses"
            
        # 5. Volatility check (if we have enough history)
        if len(history) >= 10:
            volatility = self._calculate_volatility(history)
            if volatility > self.stop_loss_config['volatility_threshold']:
                return True, f"Portfolio volatility {volatility:.2f}x exceeds threshold"
                
        # 6. Performance vs backtest expectations
        if self.backtest_expectations:
            deviation = self._check_performance_deviation(session, current_value)
            if deviation:
                return True, f"Performance deviation: {deviation}"
                
        return False, None
        
    def _calculate_volatility(self, history: List[Dict]) -> float:
        """Calculate portfolio volatility as multiple of normal"""
        if len(history) < 2:
            return 0
            
        values = [h['value'] for h in history]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        if not returns:
            return 0
            
        import numpy as np
        volatility = np.std(returns)
        normal_volatility = 0.01  # 1% normal volatility
        
        return volatility / normal_volatility
        
    def _check_performance_deviation(self, session: Dict, current_value: float) -> Optional[str]:
        """Check if performance deviates significantly from backtest expectations"""
        if not self.backtest_expectations:
            return None
            
        session_age_hours = (datetime.now(timezone.utc) - 
                           datetime.fromisoformat(session['created_at'].replace('+00:00', '+00:00'))).total_seconds() / 3600
                           
        if session_age_hours < 1:  # Need at least 1 hour of data
            return None
            
        # Calculate expected return based on backtest
        expected_hourly_return = self.backtest_expectations.get('hourly_return', 0)
        expected_value = session['initial_bankroll'] * (1 + expected_hourly_return * session_age_hours)
        
        # Check if significantly underperforming
        performance_ratio = current_value / expected_value
        if performance_ratio < 0.7:  # 30% below expected
            return f"Performance {(1-performance_ratio)*100:.1f}% below backtest expectations"
            
        return None
        
    def _get_open_positions(self, session_id: str) -> List[Dict]:
        """Get all open positions for a session"""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT bet_id, match_id, bet_type, bet_on, odds, stake, 
                           placed_at, status
                    FROM paper_trading_positions
                    WHERE session_id = %s AND status IN ('pending', 'open')
                    ORDER BY placed_at DESC
                """, (session_id,))
                
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
                
    def _close_position(self, session_id: str, position: Dict) -> Dict[str, Any]:
        """Close a single position"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    # Mark position as closed
                    cur.execute("""
                        UPDATE paper_trading_positions
                        SET status = 'closed',
                            result = 'stopped',
                            pnl = -stake,  -- Assume loss when stopped
                            settled_at = CURRENT_TIMESTAMP
                        WHERE bet_id = %s
                        RETURNING pnl
                    """, (position['bet_id'],))
                    
                    pnl = cur.fetchone()[0] if cur.rowcount > 0 else 0
                    
                    # Update session balance
                    cur.execute("""
                        UPDATE paper_trading_snapshots
                        SET cash_balance = cash_balance + %s,
                            portfolio_value = portfolio_value + %s
                        WHERE session_id = %s AND snapshot_time = (
                            SELECT MAX(snapshot_time) 
                            FROM paper_trading_snapshots 
                            WHERE session_id = %s
                        )
                    """, (position['stake'], pnl, session_id, session_id))
                    
                    conn.commit()
                    
                    return {
                        'success': True,
                        'bet_id': position['bet_id'],
                        'match_id': position['match_id'],
                        'stake': float(position['stake']),
                        'pnl': float(pnl),
                        'closed_at': datetime.now(timezone.utc).isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'success': False, 'error': str(e)}
            
    def _update_session_status(self, session_id: str, status: str, metadata: Dict):
        """Update session status in database"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    # Update session status
                    cur.execute("""
                        UPDATE paper_trading_sessions
                        SET status = %s,
                            strategy_config = strategy_config || %s
                        WHERE session_id = %s
                    """, (status, json.dumps({'stop_metadata': metadata}), session_id))
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error updating session status: {e}")
            
    def _log_stop_event(self, session_id: str, reason: str, closed_positions: List[Dict]):
        """Log stop event for analysis"""
        try:
            event_data = {
                'session_id': session_id,
                'stop_time': self.stop_time.isoformat(),
                'reason': reason,
                'positions_closed': len(closed_positions),
                'total_stake_closed': sum(p.get('stake', 0) for p in closed_positions),
                'total_pnl': sum(p.get('pnl', 0) for p in closed_positions),
                'peak_value': self.peak_value,
                'stop_loss_config': self.stop_loss_config
            }
            
            # Write to log file
            log_file = f"stop_loss_events_{datetime.now().strftime('%Y%m%d')}.json"
            with open(log_file, 'a') as f:
                f.write(json.dumps(event_data) + '\n')
                
            logger.info(f"Logged stop event to {log_file}")
            
        except Exception as e:
            logger.error(f"Error logging stop event: {e}")
            
    def _get_recent_closed_bets(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get recent closed bets for analysis"""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT bet_id, result, pnl
                    FROM paper_trading_positions
                    WHERE session_id = %s AND status = 'settled'
                    ORDER BY settled_at DESC
                    LIMIT %s
                """, (session_id, limit))
                
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
                
    def _count_consecutive_losses(self, bets: List[Dict]) -> int:
        """Count consecutive losses from recent bets"""
        consecutive = 0
        for bet in bets:
            if bet.get('result') == 'lost' or bet.get('pnl', 0) < 0:
                consecutive += 1
            else:
                break
        return consecutive
        
    def reset_daily_tracking(self, session_id: str):
        """Reset daily tracking values (call at start of trading day)"""
        session = self.session_manager.get_session(session_id)
        if session:
            self.daily_start_value = session['portfolio_value']
            logger.info(f"Reset daily tracking. Start value: {self.daily_start_value}")
            
    def get_stop_status(self) -> Dict[str, Any]:
        """Get current stop loss status"""
        return {
            'is_stopped': self.is_stopped,
            'stop_reason': self.stop_reason,
            'stop_time': self.stop_time.isoformat() if self.stop_time else None,
            'monitoring_active': self._monitoring_active,
            'peak_value': self.peak_value,
            'daily_start_value': self.daily_start_value,
            'config': self.stop_loss_config
        }
        
    def can_resume_trading(self) -> Tuple[bool, Optional[str]]:
        """Check if trading can be resumed after a stop"""
        if not self.is_stopped:
            return True, None
            
        if not self.stop_time:
            return False, "No stop time recorded"
            
        # Check recovery time
        time_since_stop = (datetime.now(timezone.utc) - self.stop_time).total_seconds() / 60
        recovery_minutes = self.stop_loss_config['recovery_time_minutes']
        
        if time_since_stop < recovery_minutes:
            remaining = recovery_minutes - time_since_stop
            return False, f"Recovery period: {remaining:.0f} minutes remaining"
            
        return True, None
        
    def resume_trading(self):
        """Resume trading after stop conditions are cleared"""
        can_resume, reason = self.can_resume_trading()
        if not can_resume:
            logger.warning(f"Cannot resume trading: {reason}")
            return False
            
        self.is_stopped = False
        self.stop_reason = None
        logger.info("Trading resumed")
        return True