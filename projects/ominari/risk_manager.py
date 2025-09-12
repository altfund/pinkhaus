#!/usr/bin/env python3
"""
Risk Management System for Production Trading.
Enforces comprehensive risk limits and monitoring.
"""

import logging
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from production_risk_config import (
    ProductionRiskConfig, RiskLevel, get_preset_config,
    validate_bet_against_limits
)
from database_v2 import db_manager
from models import Bet, BettingSession, PaperTradingPosition, PaperTradingSnapshot
from sqlalchemy import and_, func, desc

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Current risk metrics for monitoring."""
    current_exposure: float = 0.0
    daily_pnl: float = 0.0
    current_drawdown: float = 0.0
    open_positions: int = 0
    positions_by_sport: Dict[str, int] = field(default_factory=dict)
    positions_by_market: Dict[str, int] = field(default_factory=dict)
    recent_win_rate: float = 0.0
    consecutive_losses: int = 0
    session_duration_minutes: int = 0
    bets_this_hour: int = 0
    bets_today: int = 0
    max_correlation: float = 0.0
    concentration_ratio: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            'current_exposure': self.current_exposure,
            'daily_pnl': self.daily_pnl,
            'current_drawdown': self.current_drawdown,
            'open_positions': self.open_positions,
            'positions_by_sport': self.positions_by_sport,
            'positions_by_market': self.positions_by_market,
            'recent_win_rate': self.recent_win_rate,
            'consecutive_losses': self.consecutive_losses,
            'session_duration_minutes': self.session_duration_minutes,
            'bets_this_hour': self.bets_this_hour,
            'bets_today': self.bets_today,
            'max_correlation': self.max_correlation,
            'concentration_ratio': self.concentration_ratio
        }


class RiskManager:
    """Manages risk limits and monitoring for production trading."""
    
    def __init__(self, 
                 config: Optional[ProductionRiskConfig] = None,
                 session_id: Optional[str] = None):
        self.config = config or get_preset_config(RiskLevel.MODERATE)
        self.session_id = session_id
        self.alerts_sent = []
        self.kill_switch_activated = False
        self._last_metrics = None
        
    def calculate_current_metrics(self, bankroll: float) -> RiskMetrics:
        """Calculate current risk metrics from database."""
        metrics = RiskMetrics()
        
        with db_manager.get_db_session() as db:
            now = datetime.now(timezone.utc)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            hour_ago = now - timedelta(hours=1)
            
            # Get open positions
            if self.session_id:
                open_positions = db.query(PaperTradingPosition).filter(
                    and_(
                        PaperTradingPosition.session_id == self.session_id,
                        PaperTradingPosition.is_open == True
                    )
                ).all()
                
                metrics.open_positions = len(open_positions)
                metrics.current_exposure = sum(p.amount for p in open_positions)
                
                # Positions by sport/market
                for pos in open_positions:
                    sport = pos.sport or "Unknown"
                    market = pos.market_type or "Unknown"
                    
                    metrics.positions_by_sport[sport] = metrics.positions_by_sport.get(sport, 0) + 1
                    metrics.positions_by_market[market] = metrics.positions_by_market.get(market, 0) + 1
                
                # Daily P&L
                daily_pnl = db.query(func.sum(PaperTradingPosition.pnl)).filter(
                    and_(
                        PaperTradingPosition.session_id == self.session_id,
                        PaperTradingPosition.created_at >= today_start,
                        PaperTradingPosition.is_open == False
                    )
                ).scalar() or 0.0
                
                metrics.daily_pnl = daily_pnl
                
                # Recent win rate (last 20 bets)
                recent_closed = db.query(PaperTradingPosition).filter(
                    and_(
                        PaperTradingPosition.session_id == self.session_id,
                        PaperTradingPosition.is_open == False
                    )
                ).order_by(desc(PaperTradingPosition.settled_at)).limit(20).all()
                
                if recent_closed:
                    wins = sum(1 for p in recent_closed if p.pnl > 0)
                    metrics.recent_win_rate = wins / len(recent_closed)
                    
                    # Consecutive losses
                    consecutive = 0
                    for pos in recent_closed:
                        if pos.pnl < 0:
                            consecutive += 1
                        else:
                            break
                    metrics.consecutive_losses = consecutive
                
                # Bets placed recently
                metrics.bets_this_hour = db.query(func.count(PaperTradingPosition.id)).filter(
                    and_(
                        PaperTradingPosition.session_id == self.session_id,
                        PaperTradingPosition.created_at >= hour_ago
                    )
                ).scalar()
                
                metrics.bets_today = db.query(func.count(PaperTradingPosition.id)).filter(
                    and_(
                        PaperTradingPosition.session_id == self.session_id,
                        PaperTradingPosition.created_at >= today_start
                    )
                ).scalar()
                
                # Session duration
                first_bet = db.query(PaperTradingPosition).filter_by(
                    session_id=self.session_id
                ).order_by(PaperTradingPosition.created_at).first()
                
                if first_bet:
                    metrics.session_duration_minutes = int(
                        (now - first_bet.created_at).total_seconds() / 60
                    )
                
                # Drawdown calculation
                snapshots = db.query(PaperTradingSnapshot).filter_by(
                    session_id=self.session_id
                ).order_by(PaperTradingSnapshot.snapshot_time).all()
                
                if snapshots:
                    portfolio_values = [s.total_value for s in snapshots]
                    peak = max(portfolio_values)
                    current = portfolio_values[-1]
                    metrics.current_drawdown = (peak - current) / peak if peak > 0 else 0
                
                # Concentration ratio
                if metrics.positions_by_sport:
                    max_sport_positions = max(metrics.positions_by_sport.values())
                    metrics.concentration_ratio = max_sport_positions / metrics.open_positions if metrics.open_positions > 0 else 0
        
        self._last_metrics = metrics
        return metrics
    
    def check_pre_bet_limits(self, 
                           bet_proposal: Dict,
                           bankroll: float) -> Tuple[bool, List[str]]:
        """Check if a proposed bet violates any limits."""
        violations = []
        
        # First check basic bet limits
        bet_valid, bet_violations = validate_bet_against_limits(
            bet_proposal,
            self.config,
            {'bankroll': bankroll, 'total_exposure': self._last_metrics.current_exposure if self._last_metrics else 0}
        )
        violations.extend(bet_violations)
        
        # Get current metrics
        metrics = self.calculate_current_metrics(bankroll)
        
        # Check kill switch
        if self.kill_switch_activated:
            violations.append("Kill switch is activated - no new bets allowed")
        
        # Check session timeout
        if metrics.session_duration_minutes > self.config.time_limits.session_timeout_minutes:
            violations.append(f"Session timeout - exceeds {self.config.time_limits.session_timeout_minutes} minutes")
        
        # Check consecutive losses
        if metrics.consecutive_losses >= self.config.time_limits.break_after_consecutive_losses:
            violations.append(f"Too many consecutive losses ({metrics.consecutive_losses})")
        
        # Check rate limits
        if metrics.bets_this_hour >= self.config.time_limits.max_bets_per_hour:
            violations.append(f"Hourly bet limit reached ({metrics.bets_this_hour})")
        
        if metrics.bets_today >= self.config.time_limits.max_bets_per_day:
            violations.append(f"Daily bet limit reached ({metrics.bets_today})")
        
        # Check drawdown
        if metrics.current_drawdown >= self.config.portfolio_limits.max_drawdown_pct:
            violations.append(f"Max drawdown exceeded ({metrics.current_drawdown:.1%})")
        
        # Check daily loss
        daily_loss_pct = abs(metrics.daily_pnl) / bankroll if metrics.daily_pnl < 0 else 0
        if daily_loss_pct >= self.config.portfolio_limits.max_daily_loss_pct:
            violations.append(f"Daily loss limit exceeded ({daily_loss_pct:.1%})")
        
        # Check concentration
        sport = bet_proposal.get('sport', 'Unknown')
        sport_exposure = metrics.positions_by_sport.get(sport, 0) / max(metrics.open_positions, 1)
        if sport_exposure >= self.config.portfolio_limits.concentration_limit_pct:
            violations.append(f"Concentration limit exceeded for {sport}")
        
        return len(violations) == 0, violations
    
    def check_portfolio_risk(self, bankroll: float) -> Dict[str, any]:
        """Check overall portfolio risk status."""
        metrics = self.calculate_current_metrics(bankroll)
        
        status = {
            'healthy': True,
            'warnings': [],
            'alerts': [],
            'metrics': metrics.to_dict()
        }
        
        # Check for warnings (approaching limits)
        exposure_pct = metrics.current_exposure / bankroll if bankroll > 0 else 0
        if exposure_pct > self.config.portfolio_limits.max_total_exposure_pct * 0.8:
            status['warnings'].append(f"Approaching exposure limit: {exposure_pct:.1%}")
        
        drawdown_warning = self.config.monitoring.alert_on_drawdown_pct
        if metrics.current_drawdown > drawdown_warning:
            status['warnings'].append(f"Drawdown warning: {metrics.current_drawdown:.1%}")
        
        # Check for alerts (limits breached)
        if exposure_pct > self.config.portfolio_limits.max_total_exposure_pct:
            status['alerts'].append(f"Exposure limit breached: {exposure_pct:.1%}")
            status['healthy'] = False
        
        daily_loss_pct = abs(metrics.daily_pnl) / bankroll if metrics.daily_pnl < 0 else 0
        if daily_loss_pct > self.config.monitoring.alert_on_daily_loss_pct:
            status['alerts'].append(f"Daily loss alert: {daily_loss_pct:.1%}")
        
        # Check kill switch
        if daily_loss_pct > self.config.monitoring.kill_switch_loss_pct:
            status['alerts'].append(f"KILL SWITCH TRIGGERED: {daily_loss_pct:.1%} loss")
            status['healthy'] = False
            self.activate_kill_switch()
        
        # Send alerts if configured
        if status['alerts'] and self.config.monitoring.webhook_url:
            self._send_alerts(status['alerts'])
        
        return status
    
    def activate_kill_switch(self):
        """Activate the emergency kill switch."""
        self.kill_switch_activated = True
        logger.critical("KILL SWITCH ACTIVATED - All trading halted")
        
        # Log to database
        with db_manager.get_db_session() as db:
            # Could create a risk_events table to track this
            pass
    
    def _send_alerts(self, alerts: List[str]):
        """Send risk alerts via webhook."""
        if not self.config.monitoring.webhook_url:
            return
        
        try:
            import requests
            
            payload = {
                'session_id': self.session_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'alerts': alerts,
                'metrics': self._last_metrics.to_dict() if self._last_metrics else {}
            }
            
            response = requests.post(
                self.config.monitoring.webhook_url,
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info(f"Sent {len(alerts)} alerts successfully")
            else:
                logger.error(f"Failed to send alerts: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending alerts: {e}")
    
    def get_risk_report(self, bankroll: float) -> str:
        """Generate a comprehensive risk report."""
        metrics = self.calculate_current_metrics(bankroll)
        portfolio_status = self.check_portfolio_risk(bankroll)
        
        report = f"""
RISK MANAGEMENT REPORT
======================
Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
Session ID: {self.session_id or 'N/A'}
Risk Level: {self.config.risk_level.value.upper()}

PORTFOLIO METRICS
-----------------
Current Exposure: ${metrics.current_exposure:,.2f} ({metrics.current_exposure/bankroll*100:.1f}% of bankroll)
Open Positions: {metrics.open_positions}
Daily P&L: ${metrics.daily_pnl:,.2f} ({metrics.daily_pnl/bankroll*100:.1f}%)
Current Drawdown: {metrics.current_drawdown:.1%}
Session Duration: {metrics.session_duration_minutes} minutes

POSITION BREAKDOWN
------------------"""
        
        # Add position breakdown
        if metrics.positions_by_sport:
            report += "\nBy Sport:\n"
            for sport, count in sorted(metrics.positions_by_sport.items()):
                report += f"  {sport}: {count} positions\n"
        
        if metrics.positions_by_market:
            report += "\nBy Market Type:\n"
            for market, count in sorted(metrics.positions_by_market.items()):
                report += f"  {market}: {count} positions\n"
        
        report += f"""
ACTIVITY METRICS
----------------
Bets This Hour: {metrics.bets_this_hour} / {self.config.time_limits.max_bets_per_hour}
Bets Today: {metrics.bets_today} / {self.config.time_limits.max_bets_per_day}
Recent Win Rate: {metrics.recent_win_rate:.1%}
Consecutive Losses: {metrics.consecutive_losses}

RISK STATUS
-----------
Portfolio Health: {'✅ HEALTHY' if portfolio_status['healthy'] else '❌ UNHEALTHY'}
Kill Switch: {'🚨 ACTIVATED' if self.kill_switch_activated else '✅ Inactive'}"""
        
        if portfolio_status['warnings']:
            report += "\n\nWARNINGS:\n"
            for warning in portfolio_status['warnings']:
                report += f"  ⚠️  {warning}\n"
        
        if portfolio_status['alerts']:
            report += "\n\nALERTS:\n"
            for alert in portfolio_status['alerts']:
                report += f"  🚨 {alert}\n"
        
        report += f"""
LIMIT SUMMARY
-------------
Position Limits:
  Max per bet: {self.config.position_limits.max_single_bet_pct:.1%} / ${self.config.position_limits.max_single_bet_abs}
  Min edge required: {self.config.position_limits.min_edge:.1%}
  
Portfolio Limits:
  Max exposure: {self.config.portfolio_limits.max_total_exposure_pct:.1%}
  Max daily loss: {self.config.portfolio_limits.max_daily_loss_pct:.1%}
  Max drawdown: {self.config.portfolio_limits.max_drawdown_pct:.1%}
  
Kelly Settings:
  Kelly fraction: {self.config.kelly_limits.kelly_fraction:.1%}
  Per game cap: {self.config.kelly_limits.kelly_cap_per_game:.1%}
"""
        
        return report


def demonstrate_risk_manager():
    """Demonstrate risk management features."""
    # Create manager with moderate risk
    manager = RiskManager(
        config=get_preset_config(RiskLevel.MODERATE),
        session_id="test_session"
    )
    
    # Test pre-bet validation
    test_bet = {
        'stake': 100,
        'stake_pct': 0.01,
        'odds': 2.5,
        'edge': 0.02,
        'event_time': datetime.now(timezone.utc) + timedelta(hours=2),
        'sport': 'Soccer'
    }
    
    is_valid, violations = manager.check_pre_bet_limits(test_bet, bankroll=10000)
    
    print("=== Pre-Bet Validation ===")
    print(f"Bet proposal: {test_bet}")
    print(f"Valid: {is_valid}")
    if violations:
        print("Violations:")
        for v in violations:
            print(f"  - {v}")
    
    # Generate risk report
    print("\n" + manager.get_risk_report(bankroll=10000))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_risk_manager()