#!/usr/bin/env python3
"""
Real-time Position Tracker with Notifications
Monitors open positions, P&L, and sends alerts for important events.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PositionTracker:
    """Tracks and monitors trading positions in real-time."""

    def __init__(self, session_file: str = None, check_interval: int = 60):
        """
        Initialize the position tracker.

        Args:
            session_file: Path to session JSON file (default: auto-detect)
            check_interval: How often to check positions (seconds)
        """
        self.check_interval = check_interval
        self.session_file = self._find_session_file(session_file)
        self.last_portfolio_value = None
        self.notification_log = []

        logger.info(f"Position Tracker initialized")
        logger.info(f"Monitoring: {self.session_file}")
        logger.info(f"Check interval: {check_interval}s")

    def _find_session_file(self, session_file: Optional[str]) -> str:
        """Find the appropriate session file to monitor."""
        if session_file and os.path.exists(session_file):
            return session_file

        # Check for environment-specific session
        env = os.getenv('OMINARI_ENV', 'dev')
        env_file = f"paper_trading_sessions_{env}.json"

        if os.path.exists(env_file):
            return env_file

        # Fallback to any available session file
        for pattern in ['paper_trading_sessions_*.json', 'paper_trading_session_*.json']:
            import glob
            files = glob.glob(pattern)
            if files:
                # Get most recently modified
                return max(files, key=os.path.getmtime)

        raise FileNotFoundError("No session file found")

    def load_session_data(self) -> Dict[str, Any]:
        """Load current session data."""
        try:
            with open(self.session_file, 'r') as f:
                data = json.load(f)

            # Handle different session file formats
            if 'sessions' in data:
                # Multi-session format
                sessions = data['sessions']
                if sessions:
                    # Get most recent active session
                    active = [s for s in sessions.values() if s.get('status') == 'active']
                    if active:
                        return max(active, key=lambda x: x.get('created_at', ''))

            # Single session format
            return data

        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return {}

    def get_open_positions(self, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract open positions from session."""
        positions = session.get('positions', {})
        open_positions = []

        for pos_id, pos_data in positions.items():
            if pos_data.get('status') == 'open':
                open_positions.append({
                    'id': pos_id,
                    **pos_data
                })

        return open_positions

    def format_position_summary(self, position: Dict[str, Any]) -> str:
        """Format a position for display."""
        market_name = position.get('market_name', 'Unknown')
        outcome = position.get('outcome', '?').upper()
        stake = position.get('total_stake', 0)
        odds = position.get('avg_odds', 0)
        pnl = position.get('pnl', 0)
        current_value = position.get('current_value', 0)

        # Calculate time to maturity
        maturity = position.get('maturity_date')
        if maturity:
            try:
                # Handle different datetime formats
                if isinstance(maturity, str):
                    # Try parsing with timezone info
                    maturity_str = maturity.replace('Z', '+00:00')
                    # Remove microseconds if present for cleaner parsing
                    if '.' in maturity_str and '+' in maturity_str:
                        parts = maturity_str.split('.')
                        maturity_str = parts[0] + parts[1][parts[1].find('+'):]

                    maturity_dt = datetime.fromisoformat(maturity_str)
                else:
                    maturity_dt = maturity

                # Ensure timezone aware
                if maturity_dt.tzinfo is None:
                    maturity_dt = maturity_dt.replace(tzinfo=timezone.utc)

                now = datetime.now(timezone.utc)
                time_left = maturity_dt - now

                if time_left.total_seconds() < 0:
                    time_str = "EXPIRED"
                elif time_left.total_seconds() < 3600:
                    time_str = f"{int(time_left.total_seconds() / 60)}m"
                elif time_left.total_seconds() < 86400:
                    time_str = f"{int(time_left.total_seconds() / 3600)}h"
                else:
                    time_str = f"{int(time_left.total_seconds() / 86400)}d"
            except Exception as e:
                # Debug: Log the error for troubleshooting
                # logger.debug(f"Error parsing maturity date '{maturity}': {e}")
                time_str = "?"
        else:
            time_str = "?"

        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        pnl_pct = (pnl / stake * 100) if stake > 0 else 0

        return (
            f"{market_name[:40]:<40} | {outcome:6} @ {odds:.2f} | "
            f"Stake: ${stake:.2f} | Value: ${current_value:.2f} | "
            f"P&L: {pnl_str:>10} ({pnl_pct:+.1f}%) | TTM: {time_str:>5}"
        )

    def check_for_alerts(self, session: Dict[str, Any], positions: List[Dict[str, Any]]):
        """Check for conditions that should trigger notifications."""
        alerts = []

        # Check portfolio value change
        current_value = session.get('portfolio_value', 0)
        if self.last_portfolio_value is not None:
            change = current_value - self.last_portfolio_value
            change_pct = (change / self.last_portfolio_value * 100) if self.last_portfolio_value > 0 else 0

            # Alert on significant changes (>2%)
            if abs(change_pct) > 2:
                alerts.append({
                    'type': 'portfolio_change',
                    'severity': 'high' if abs(change_pct) > 5 else 'medium',
                    'message': f"Portfolio value changed by {change_pct:+.2f}% (${change:+.2f})"
                })

        self.last_portfolio_value = current_value

        # Check positions expiring soon
        now = datetime.now(timezone.utc)
        for pos in positions:
            maturity = pos.get('maturity_date')
            if maturity:
                try:
                    maturity_dt = datetime.fromisoformat(maturity.replace('Z', '+00:00'))
                    time_left = maturity_dt - now

                    # Alert if expiring in next 30 minutes
                    if 0 < time_left.total_seconds() < 1800:
                        alerts.append({
                            'type': 'expiring_soon',
                            'severity': 'high',
                            'message': f"Position expiring in {int(time_left.total_seconds()/60)}m: {pos.get('market_name')}"
                        })

                    # Alert if already expired (should be settled)
                    elif time_left.total_seconds() < 0:
                        alerts.append({
                            'type': 'expired_unsettled',
                            'severity': 'high',
                            'message': f"Expired position not settled: {pos.get('market_name')}"
                        })
                except:
                    pass

        # Check for large unrealized P&L
        for pos in positions:
            pnl = pos.get('pnl', 0)
            stake = pos.get('total_stake', 1)
            pnl_pct = (pnl / stake * 100) if stake > 0 else 0

            # Alert on large unrealized gains/losses (>20%)
            if abs(pnl_pct) > 20:
                alerts.append({
                    'type': 'large_pnl',
                    'severity': 'medium',
                    'message': f"Large P&L ({pnl_pct:+.1f}%): {pos.get('market_name')} - ${pnl:+.2f}"
                })

        # Log alerts
        for alert in alerts:
            self._log_notification(alert)

    def _log_notification(self, alert: Dict[str, Any]):
        """Log a notification."""
        timestamp = datetime.now(timezone.utc).isoformat()
        notification = {
            'timestamp': timestamp,
            **alert
        }

        self.notification_log.append(notification)

        # Display alert
        severity_symbol = {
            'low': 'ℹ️',
            'medium': '⚠️',
            'high': '🚨'
        }
        symbol = severity_symbol.get(alert['severity'], 'ℹ️')

        logger.warning(f"{symbol} {alert['type'].upper()}: {alert['message']}")

    def display_dashboard(self):
        """Display current position dashboard."""
        session = self.load_session_data()
        if not session:
            logger.error("No session data available")
            return

        positions = self.get_open_positions(session)

        # Clear screen (optional, comment out if not desired)
        # os.system('clear' if os.name == 'posix' else 'cls')

        print("\n" + "="*120)
        print(f"POSITION TRACKER - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("="*120)

        # Portfolio summary
        initial = session.get('initial_bankroll', 0)
        current = session.get('current_bankroll', 0)
        portfolio = session.get('portfolio_value', 0)
        perf = session.get('performance', {})

        total_pnl = portfolio - initial
        total_pnl_pct = (total_pnl / initial * 100) if initial > 0 else 0

        print(f"\nPORTFOLIO SUMMARY:")
        print(f"  Initial Bankroll: ${initial:,.2f}")
        print(f"  Current Cash:     ${current:,.2f}")
        print(f"  Portfolio Value:  ${portfolio:,.2f}")
        print(f"  Total P&L:        ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")
        print(f"\n  Total Trades:     {perf.get('total_trades', 0)}")
        print(f"  Open Positions:   {len(positions)}")
        print(f"  Winning Trades:   {perf.get('winning_trades', 0)}")
        print(f"  Losing Trades:    {perf.get('losing_trades', 0)}")

        # Open positions
        if positions:
            print(f"\nOPEN POSITIONS ({len(positions)}):")
            print("-"*120)
            print(f"{'Market':<40} | {'Side':6} | {'Odds':6} | {'Stake':14} | {'Value':15} | {'P&L':22} | {'TTM':5}")
            print("-"*120)

            # Sort by maturity date
            positions_sorted = sorted(
                positions,
                key=lambda p: p.get('maturity_date', '9999'),
                reverse=False
            )

            for pos in positions_sorted:
                print(self.format_position_summary(pos))
        else:
            print("\nNo open positions")

        print("\n" + "="*120)

        # Check for alerts
        self.check_for_alerts(session, positions)

        # Show recent notifications
        if self.notification_log:
            recent = self.notification_log[-5:]  # Last 5
            print(f"\nRECENT ALERTS:")
            for notif in recent:
                ts = datetime.fromisoformat(notif['timestamp']).strftime('%H:%M:%S')
                print(f"  [{ts}] {notif['type']}: {notif['message']}")

        print()

    def run(self, continuous: bool = True):
        """Run the position tracker."""
        logger.info("Starting position tracker...")

        if not continuous:
            self.display_dashboard()
            return

        try:
            while True:
                self.display_dashboard()
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("\nPosition tracker stopped by user")
        except Exception as e:
            logger.error(f"Error in position tracker: {e}", exc_info=True)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Real-time Position Tracker')
    parser.add_argument('--session', type=str, help='Session file to monitor')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
    parser.add_argument('--once', action='store_true', help='Run once and exit (no continuous monitoring)')

    args = parser.parse_args()

    tracker = PositionTracker(
        session_file=args.session,
        check_interval=args.interval
    )

    tracker.run(continuous=not args.once)


if __name__ == "__main__":
    main()
