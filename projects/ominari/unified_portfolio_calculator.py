#!/usr/bin/env python3
"""
Unified Portfolio Calculator
Single source of truth for all portfolio calculations across the system.
"""

import sys
import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_trading_sessions import PaperTradingSessionManager

@dataclass
class PortfolioMetrics:
    """Standardized portfolio metrics."""
    # Core values
    initial_bankroll: float
    current_bankroll: float  # Available cash
    portfolio_value: float   # Total net worth (MTM - includes unrealized)

    # CORRECT BETTING ACCOUNTING (NEW)
    book_value: float        # Cash + Stakes at cost (no unrealized P&L)
    book_pnl: float          # Realized P&L only (from settled trades)
    book_roi_percentage: float  # ROI based on book value

    # P&L breakdown
    realized_pnl: float      # Closed position profits/losses
    unrealized_pnl: float    # Mark-to-market on open positions
    total_pnl: float         # realized + unrealized (MTM)

    # Cash-based metrics (DEPRECATED - use book_value instead)
    cash_portfolio_value: float  # Current cash only (missing stakes)
    cash_pnl: float             # Cash change from initial
    cash_roi_percentage: float   # ROI based on cash only

    # Position data
    active_positions: int
    total_stake: float       # Total deployed capital

    # Performance
    roi_percentage: float    # Total return on initial capital (MTM)
    win_rate: float         # Percentage of winning closed trades

    # Meta
    session_id: str
    last_updated: datetime

class UnifiedPortfolioCalculator:
    """Single source of truth for portfolio calculations."""
    
    def __init__(self):
        self.session_manager = PaperTradingSessionManager()
    
    def get_current_portfolio_metrics(self, force_reload: bool = True) -> PortfolioMetrics:
        """
        Get current portfolio metrics using the CORRECT calculation method.
        
        Args:
            force_reload: Whether to reload session data from disk
            
        Returns:
            PortfolioMetrics object with all standardized metrics
        """
        if force_reload:
            # Always reload from disk to get latest data
            self.session_manager.sessions = self.session_manager._load_sessions()
        
        # Get the current active session
        current_session = self._get_active_trading_session()
        
        if not current_session:
            # Return default metrics for empty session
            return PortfolioMetrics(
                initial_bankroll=10000.0,
                current_bankroll=10000.0,
                portfolio_value=10000.0,
                book_value=10000.0,
                book_pnl=0.0,
                book_roi_percentage=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                total_pnl=0.0,
                cash_portfolio_value=10000.0,
                cash_pnl=0.0,
                cash_roi_percentage=0.0,
                active_positions=0,
                total_stake=0.0,
                roi_percentage=0.0,
                win_rate=0.0,
                session_id="none",
                last_updated=datetime.now(timezone.utc)
            )
        
        # Extract core financial data
        session_id = current_session.get('session_id', 'unknown')
        initial_bankroll = current_session.get('initial_bankroll', 10000.0)
        current_bankroll = current_session.get('current_bankroll', initial_bankroll)
        
        # Get performance data
        performance = current_session.get('performance', {})
        realized_pnl = performance.get('total_pnl', 0.0)
        
        # Calculate position metrics
        positions = current_session.get('positions', {})
        active_positions = len(positions)
        total_stake = sum(pos.get('total_stake', 0) for pos in positions.values())
        unrealized_pnl = sum(pos.get('pnl', 0) for pos in positions.values())
        
        # MTM PORTFOLIO CALCULATION: Total net worth including unrealized gains
        total_pnl = realized_pnl + unrealized_pnl
        portfolio_value = initial_bankroll + total_pnl
        roi_percentage = (total_pnl / initial_bankroll * 100) if initial_bankroll > 0 else 0.0
        
        # CASH-BASED PORTFOLIO CALCULATION: Actual cash position only (DEPRECATED)
        cash_pnl = current_bankroll - initial_bankroll
        cash_portfolio_value = current_bankroll  # Just the actual cash
        cash_roi_percentage = (cash_pnl / initial_bankroll * 100) if initial_bankroll > 0 else 0.0

        # BOOK VALUE CALCULATION: Correct betting accounting (Cash + Stakes at cost)
        book_value = current_bankroll + total_stake  # Total capital (cash + deployed)
        book_pnl = realized_pnl  # Only realized P&L from settled trades
        book_roi_percentage = (book_pnl / initial_bankroll * 100) if initial_bankroll > 0 else 0.0

        # Calculate win rate from closed trades
        winning_trades = performance.get('winning_trades', 0)
        losing_trades = performance.get('losing_trades', 0)
        total_closed_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0.0
        
        return PortfolioMetrics(
            initial_bankroll=initial_bankroll,
            current_bankroll=current_bankroll,
            portfolio_value=portfolio_value,
            book_value=book_value,
            book_pnl=book_pnl,
            book_roi_percentage=book_roi_percentage,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            cash_portfolio_value=cash_portfolio_value,
            cash_pnl=cash_pnl,
            cash_roi_percentage=cash_roi_percentage,
            active_positions=active_positions,
            total_stake=total_stake,
            roi_percentage=roi_percentage,
            win_rate=win_rate,
            session_id=session_id,
            last_updated=datetime.now(timezone.utc)
        )
    
    def _get_active_trading_session(self) -> Optional[Dict[str, Any]]:
        """Find the active trading session - always use current session."""
        # Always use the current session, even if it has no positions/trades
        current_session = self.session_manager.get_current_session()

        if current_session:
            return current_session

        # Fallback: Look for any session with actual trading activity
        all_sessions = self.session_manager.sessions.get("sessions", {})

        # Find sessions with positions or trades, sorted by creation time
        active_sessions = []
        for session_id, session_data in all_sessions.items():
            if session_data.get('positions') or session_data.get('trades'):
                session_data['session_id'] = session_id  # Ensure session_id is set
                active_sessions.append((session_id, session_data))

        if active_sessions:
            # Sort by creation time and get most recent
            active_sessions.sort(key=lambda x: x[1].get('created_at', ''), reverse=True)
            return active_sessions[0][1]

        return None
    
    def validate_portfolio_calculation(self) -> Dict[str, Any]:
        """
        Validate that portfolio calculations are mathematically correct.
        Returns detailed breakdown for debugging.
        """
        metrics = self.get_current_portfolio_metrics()
        
        # Check the accounting equation
        accounting_check = abs(
            metrics.portfolio_value - 
            (metrics.initial_bankroll + metrics.total_pnl)
        ) < 0.01
        
        # Check P&L consistency
        pnl_check = abs(
            metrics.total_pnl - 
            (metrics.realized_pnl + metrics.unrealized_pnl)
        ) < 0.01
        
        # Check cash flow logic
        # Cash should equal: initial - total execution stakes + total payouts
        # (execution stakes include fees, payouts are gross winnings)
        current_session = self._get_active_trading_session()
        if current_session:
            # Get all positions (open + closed)
            all_positions = (
                list(current_session.get('positions', {}).values()) +
                current_session.get('closed_positions', [])
            )

            # Total execution stakes (nominal + fees)
            total_execution = sum(
                pos.get('execution_stake', pos.get('total_stake', 0))
                for pos in all_positions
            )

            # Total payouts from wins
            total_payouts = sum(
                pos.get('final_value', 0)
                for pos in current_session.get('closed_positions', [])
            )

            expected_cash = metrics.initial_bankroll - total_execution + total_payouts
            cash_check = abs(metrics.current_bankroll - expected_cash) < 1.0
        else:
            cash_check = True
            expected_cash = metrics.current_bankroll
        
        return {
            'metrics': metrics,
            'validation_checks': {
                'accounting_equation_valid': accounting_check,
                'pnl_calculation_valid': pnl_check,
                'cash_flow_valid': cash_check
            },
            'detailed_breakdown': {
                'portfolio_formula': f"${metrics.initial_bankroll:,.2f} + ${metrics.total_pnl:,.2f} = ${metrics.portfolio_value:,.2f}",
                'pnl_breakdown': f"${metrics.realized_pnl:,.2f} + ${metrics.unrealized_pnl:,.2f} = ${metrics.total_pnl:,.2f}",
                'expected_cash': f"${expected_cash:,.2f}",
                'actual_cash': f"${metrics.current_bankroll:,.2f}"
            }
        }

def main():
    """Test the unified portfolio calculator."""
    print("🧮 Testing Unified Portfolio Calculator")
    print("=" * 50)
    
    calculator = UnifiedPortfolioCalculator()
    
    # Get current metrics
    print("📊 CURRENT PORTFOLIO METRICS:")
    metrics = calculator.get_current_portfolio_metrics()
    
    print(f"Session ID: {metrics.session_id}")
    print(f"Initial Bankroll: ${metrics.initial_bankroll:,.2f}")
    print(f"Current Cash: ${metrics.current_bankroll:,.2f}")
    print(f"Portfolio Value: ${metrics.portfolio_value:,.2f}")
    print(f"Total P&L: ${metrics.total_pnl:,.2f} ({metrics.roi_percentage:+.1f}%)")
    print(f"  ↳ Realized: ${metrics.realized_pnl:,.2f}")
    print(f"  ↳ Unrealized: ${metrics.unrealized_pnl:,.2f}")
    print(f"Active Positions: {metrics.active_positions}")
    print(f"Total Stake: ${metrics.total_stake:,.2f}")
    print(f"Win Rate: {metrics.win_rate:.1f}%")
    
    print("\n🔍 VALIDATION CHECKS:")
    validation = calculator.validate_portfolio_calculation()
    
    checks = validation['validation_checks']
    for check_name, is_valid in checks.items():
        status = "✅" if is_valid else "❌"
        print(f"{status} {check_name.replace('_', ' ').title()}")
    
    print("\n📋 DETAILED BREAKDOWN:")
    breakdown = validation['detailed_breakdown']
    for key, value in breakdown.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    if all(checks.values()):
        print("\n✅ All portfolio calculations are mathematically correct!")
    else:
        print("\n❌ Portfolio calculation errors detected!")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()