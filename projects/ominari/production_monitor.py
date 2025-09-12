#!/usr/bin/env python3
"""
Production Monitoring Dashboard for Ominari Trading System.
Real-time monitoring of risk metrics and system health.
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import curses
from dataclasses import dataclass

from database_v2 import db_manager
from models import PaperTradingPosition, PaperTradingSnapshot
from risk_manager import RiskManager, RiskMetrics
from production_risk_config import load_config
from sqlalchemy import and_, func, desc

logger = logging.getLogger(__name__)


@dataclass 
class SystemMetrics:
    """System-wide metrics for monitoring."""
    active_sessions: int = 0
    total_positions: int = 0
    total_volume_today: float = 0.0
    total_pnl_today: float = 0.0
    system_uptime_hours: float = 0.0
    last_bet_time: Optional[datetime] = None
    error_count_hour: int = 0
    api_latency_ms: float = 0.0


class ProductionMonitor:
    """Real-time monitoring for production trading."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.risk_manager = None
        self.start_time = datetime.now(timezone.utc)
        self.refresh_interval = 5  # seconds
        
        # Load risk config if session specified
        if session_id:
            try:
                # In production, would load from session metadata
                self.risk_manager = RiskManager(session_id=session_id)
            except:
                pass
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get system-wide metrics."""
        metrics = SystemMetrics()
        
        with db_manager.get_db_session() as db:
            now = datetime.now(timezone.utc)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            hour_ago = now - timedelta(hours=1)
            
            # Active sessions (positions opened in last 24h)
            active_sessions = db.query(
                func.count(func.distinct(PaperTradingPosition.session_id))
            ).filter(
                PaperTradingPosition.created_at >= now - timedelta(hours=24)
            ).scalar()
            metrics.active_sessions = active_sessions or 0
            
            # Total positions
            metrics.total_positions = db.query(
                func.count(PaperTradingPosition.id)
            ).filter(
                PaperTradingPosition.is_open == True
            ).scalar() or 0
            
            # Volume today
            volume = db.query(
                func.sum(PaperTradingPosition.amount)
            ).filter(
                PaperTradingPosition.created_at >= today_start
            ).scalar()
            metrics.total_volume_today = volume or 0.0
            
            # P&L today
            pnl = db.query(
                func.sum(PaperTradingPosition.pnl)
            ).filter(
                and_(
                    PaperTradingPosition.settled_at >= today_start,
                    PaperTradingPosition.is_open == False
                )
            ).scalar()
            metrics.total_pnl_today = pnl or 0.0
            
            # Last bet time
            last_bet = db.query(PaperTradingPosition).order_by(
                desc(PaperTradingPosition.created_at)
            ).first()
            if last_bet:
                metrics.last_bet_time = last_bet.created_at
            
            # System uptime
            metrics.system_uptime_hours = (now - self.start_time).total_seconds() / 3600
        
        return metrics
    
    def get_session_performance(self, limit: int = 10) -> List[Dict]:
        """Get top performing sessions."""
        with db_manager.get_db_session() as db:
            # Get latest snapshot for each session
            subq = db.query(
                PaperTradingSnapshot.session_id,
                func.max(PaperTradingSnapshot.snapshot_time).label('latest_time')
            ).group_by(
                PaperTradingSnapshot.session_id
            ).subquery()
            
            snapshots = db.query(PaperTradingSnapshot).join(
                subq,
                and_(
                    PaperTradingSnapshot.session_id == subq.c.session_id,
                    PaperTradingSnapshot.snapshot_time == subq.c.latest_time
                )
            ).order_by(
                desc(PaperTradingSnapshot.total_value)
            ).limit(limit).all()
            
            sessions = []
            for snap in snapshots:
                initial_value = 1000.0  # Would get from session metadata
                pnl = snap.total_value - initial_value
                pnl_pct = (pnl / initial_value * 100) if initial_value > 0 else 0
                
                sessions.append({
                    'session_id': snap.session_id,
                    'total_value': snap.total_value,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'open_positions': snap.open_positions,
                    'last_update': snap.snapshot_time
                })
            
            return sessions
    
    def format_dashboard(self) -> str:
        """Format monitoring data as text dashboard."""
        system_metrics = self.get_system_metrics()
        
        dashboard = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                      OMINARI PRODUCTION MONITORING                          ║
║                    {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}                         ║
╚════════════════════════════════════════════════════════════════════════════╝

SYSTEM STATUS
─────────────
Active Sessions:    {system_metrics.active_sessions}
Open Positions:     {system_metrics.total_positions}
Today's Volume:     ${system_metrics.total_volume_today:,.2f}
Today's P&L:        ${system_metrics.total_pnl_today:+,.2f}
System Uptime:      {system_metrics.system_uptime_hours:.1f} hours
Last Bet:           {system_metrics.last_bet_time.strftime('%H:%M:%S') if system_metrics.last_bet_time else 'N/A'}
"""
        
        # Add session-specific metrics if available
        if self.session_id and self.risk_manager:
            risk_metrics = self.risk_manager.calculate_current_metrics(10000)  # Would get actual bankroll
            portfolio_status = self.risk_manager.check_portfolio_risk(10000)
            
            dashboard += f"""
SESSION METRICS ({self.session_id[:8]}...)
────────────────
Exposure:           ${risk_metrics.current_exposure:,.2f}
Open Positions:     {risk_metrics.open_positions}
Daily P&L:          ${risk_metrics.daily_pnl:,.2f}
Drawdown:           {risk_metrics.current_drawdown:.1%}
Win Rate:           {risk_metrics.recent_win_rate:.1%}
Consecutive Losses: {risk_metrics.consecutive_losses}
Session Duration:   {risk_metrics.session_duration_minutes} min
Bets/Hour:          {risk_metrics.bets_this_hour}
Bets/Day:           {risk_metrics.bets_today}

RISK STATUS:        {'✅ HEALTHY' if portfolio_status['healthy'] else '❌ UNHEALTHY'}
"""
            
            if portfolio_status['warnings']:
                dashboard += "\nWARNINGS:\n"
                for warning in portfolio_status['warnings']:
                    dashboard += f"  ⚠️  {warning}\n"
            
            if portfolio_status['alerts']:
                dashboard += "\nALERTS:\n"
                for alert in portfolio_status['alerts']:
                    dashboard += f"  🚨 {alert}\n"
        
        # Top sessions
        top_sessions = self.get_session_performance(5)
        if top_sessions:
            dashboard += "\nTOP SESSIONS BY P&L\n────────────────────\n"
            dashboard += f"{'Session ID':<20} {'Value':>10} {'P&L':>10} {'P&L %':>8} {'Positions':>10}\n"
            dashboard += "─" * 60 + "\n"
            
            for sess in top_sessions:
                dashboard += f"{sess['session_id'][:18]:<20} "
                dashboard += f"${sess['total_value']:>9,.2f} "
                dashboard += f"${sess['pnl']:>9,.2f} "
                dashboard += f"{sess['pnl_pct']:>7.1f}% "
                dashboard += f"{sess['open_positions']:>10}\n"
        
        return dashboard
    
    def run_curses_dashboard(self, stdscr):
        """Run interactive curses dashboard."""
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        stdscr.timeout(100) # Refresh every 100ms
        
        while True:
            try:
                # Clear screen
                stdscr.clear()
                
                # Get dashboard text
                dashboard = self.format_dashboard()
                
                # Display line by line
                lines = dashboard.split('\n')
                max_y, max_x = stdscr.getmaxyx()
                
                for i, line in enumerate(lines[:max_y-1]):
                    try:
                        # Truncate line if too long
                        if len(line) > max_x - 1:
                            line = line[:max_x-1]
                        
                        # Color coding
                        if '✅' in line:
                            stdscr.addstr(i, 0, line, curses.color_pair(1))
                        elif '❌' in line or '🚨' in line:
                            stdscr.addstr(i, 0, line, curses.color_pair(2))
                        elif '⚠️' in line:
                            stdscr.addstr(i, 0, line, curses.color_pair(3))
                        else:
                            stdscr.addstr(i, 0, line)
                    except:
                        pass
                
                # Add footer
                footer = f"[q] Quit | [r] Refresh | Auto-refresh: {self.refresh_interval}s"
                try:
                    stdscr.addstr(max_y-1, 0, footer[:max_x-1])
                except:
                    pass
                
                stdscr.refresh()
                
                # Handle input
                key = stdscr.getch()
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    continue  # Force refresh
                
                # Auto refresh
                time.sleep(self.refresh_interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Dashboard error: {e}")
                break
    
    def run_text_mode(self):
        """Run simple text mode dashboard."""
        try:
            while True:
                os.system('clear' if os.name != 'nt' else 'cls')
                print(self.format_dashboard())
                print(f"\nRefreshing every {self.refresh_interval} seconds. Press Ctrl+C to exit.")
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")


def setup_colors():
    """Setup curses colors."""
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)   # Success
    curses.init_pair(2, curses.COLOR_RED, -1)     # Error/Alert
    curses.init_pair(3, curses.COLOR_YELLOW, -1)  # Warning


def main():
    """Main entry point for production monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ominari Production Monitor')
    parser.add_argument('--session', help='Monitor specific session ID')
    parser.add_argument('--text', action='store_true', help='Use text mode instead of curses')
    parser.add_argument('--interval', type=int, default=5, help='Refresh interval in seconds')
    
    args = parser.parse_args()
    
    monitor = ProductionMonitor(session_id=args.session)
    monitor.refresh_interval = args.interval
    
    if args.text:
        monitor.run_text_mode()
    else:
        try:
            curses.wrapper(lambda stdscr: (setup_colors(), monitor.run_curses_dashboard(stdscr)))
        except Exception as e:
            logger.error(f"Curses mode failed: {e}")
            print("Falling back to text mode...")
            monitor.run_text_mode()


if __name__ == "__main__":
    # Setup logging to file only (not console)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('monitor.log')]
    )
    
    main()