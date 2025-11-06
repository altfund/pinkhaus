#!/usr/bin/env python3
"""
Terminal-based Trading Monitor with Real-time Display
"""

import os
import sys
import time
import threading
import signal
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
import json

# Set PostgreSQL environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.columns import Columns
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import keyboard
from rich.prompt import Prompt, Confirm

from portfolio_trading_engine import PortfolioTradingEngine
from paper_trading_postgres_integrated import PaperTradingSessionManager
from edge_calculator import EdgeCalculator
from stop_loss_manager import StopLossManager
from database_v2 import db_manager
from models import Market, Odd
from trading_logger import trading_logger, log_trade, log_positions, log_performance, log_markets, log_stop_loss, log_cycle

console = Console()

class TerminalTradingMonitor:
    """Real-time terminal trading monitor"""
    
    def __init__(self):
        self.running = True
        self.paused = False
        self.session_manager = PaperTradingSessionManager()
        self.edge_calculator = EdgeCalculator()
        self.stop_loss_manager = StopLossManager(self.session_manager)
        
        # Initialize session
        self.session_id = self.session_manager.get_current_session()
        if not self.session_id:
            self.session_id = self.session_manager.create_session(initial_bankroll=10000)
            trading_logger.info(f"Created new session: {self.session_id}")
        else:
            trading_logger.info(f"Using existing session: {self.session_id}")
        
        # Get session data
        self.session = self.session_manager.get_session(self.session_id)
        
        # Trading configuration
        self.strategy_config = {
            'bankroll': float(self.session['current_bankroll']),
            'kelly_fraction': 0.25,
            'min_edge': 0.02,
            'cap_per_bet': 0.01,
            'cap_per_game': 0.02,
            'min_bet': 10,
            'max_positions': 20
        }
        
        # Initialize portfolio engine
        self.portfolio_engine = PortfolioTradingEngine(
            self.session_manager, 
            self.edge_calculator, 
            self.strategy_config
        )
        
        # Stop loss configuration
        stop_loss_config = {
            'drawdown_pct': 5,
            'time_window_minutes': 10,
            'max_daily_loss_pct': 10,
            'consecutive_losses': 3,
            'recovery_time_minutes': 60
        }
        self.stop_loss_manager.set_stop_loss_config(stop_loss_config)
        self.stop_loss_manager.start_monitoring(self.session_id)
        
        # Stats tracking
        self.cycle_count = 0
        self.total_trades = 0
        self.last_update = datetime.now()
        
        # Layout for display
        self.layout = self._create_layout()
    
    def _create_layout(self) -> Layout:
        """Create the terminal layout"""
        layout = Layout(name="root")
        
        # Split into header and body
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
        )
        
        # Split body into left and right
        layout["body"].split_row(
            Layout(name="main", ratio=2),
            Layout(name="sidebar", ratio=1),
        )
        
        # Split sidebar into sections
        layout["sidebar"].split(
            Layout(name="stats", size=10),
            Layout(name="positions", ratio=1),
        )
        
        # Split main into sections
        layout["main"].split(
            Layout(name="markets", ratio=1),
            Layout(name="logs", size=15),
        )
        
        return layout
    
    def _update_header(self):
        """Update header with system status"""
        status_color = "green" if not self.paused else "yellow"
        status_text = "RUNNING" if not self.paused else "PAUSED"
        
        header_text = f"[bold {status_color}]Ominari Trading System - {status_text}[/bold {status_color}]"
        header_text += f"  |  Session: {self.session_id[:8]}..."
        header_text += f"  |  Cycle: {self.cycle_count}"
        header_text += f"  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        self.layout["header"].update(Panel(header_text, style="blue"))
    
    def _update_stats(self):
        """Update statistics panel"""
        try:
            # Get performance data
            performance = self.session_manager.get_enhanced_performance_analytics(self.session_id)
            if not performance:
                performance = {'overview': {}, 'financial': {}}
            
            # Get current positions
            positions = self.session_manager.get_positions(self.session_id)
            open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
            current_exposure = sum(float(p['stake']) for p in open_positions)
            current_bankroll = float(self.session_manager.get_session(self.session_id)['current_bankroll'])
            
            # Create stats table
            stats_table = Table(show_header=False, box=None, padding=(0, 1))
            stats_table.add_column(style="cyan", width=15)
            stats_table.add_column(style="white", width=20)
            
            # Add rows
            stats_table.add_row("Bankroll:", f"${current_bankroll:,.2f}")
            stats_table.add_row("Exposure:", f"${current_exposure:.2f} ({current_exposure/current_bankroll*100:.1f}%)")
            stats_table.add_row("Positions:", f"{len(open_positions)}")
            stats_table.add_row("Total Trades:", f"{performance.get('overview', {}).get('total_trades', 0)}")
            
            win_rate = performance.get('overview', {}).get('win_rate', 0)
            win_color = "green" if win_rate > 0.5 else "red" if win_rate < 0.5 else "white"
            stats_table.add_row("Win Rate:", f"[{win_color}]{win_rate:.1%}[/{win_color}]")
            
            roi = performance.get('financial', {}).get('roi', 0)
            roi_color = "green" if roi > 0 else "red" if roi < 0 else "white"
            stats_table.add_row("ROI:", f"[{roi_color}]{roi:.2%}[/{roi_color}]")
            
            pnl = performance.get('financial', {}).get('total_pnl', 0)
            pnl_color = "green" if pnl > 0 else "red" if pnl < 0 else "white"
            stats_table.add_row("Total P&L:", f"[{pnl_color}]${pnl:,.2f}[/{pnl_color}]")
            
            self.layout["stats"].update(Panel(stats_table, title="📊 Statistics", border_style="blue"))
        except Exception as e:
            self.layout["stats"].update(Panel(f"Error loading stats: {e}", title="📊 Statistics", border_style="red"))
    
    def _update_positions(self):
        """Update positions panel"""
        try:
            positions = self.session_manager.get_positions(self.session_id)
            open_positions = [p for p in positions if p['status'] in ['pending', 'open']][:10]  # Show top 10
            
            if not open_positions:
                self.layout["positions"].update(Panel("No open positions", title="📈 Positions", border_style="green"))
                return
            
            # Create positions table
            pos_table = Table(show_header=True, header_style="bold cyan")
            pos_table.add_column("Match", width=25, overflow="fold")
            pos_table.add_column("Side", width=5)
            pos_table.add_column("Stake", justify="right", width=8)
            pos_table.add_column("Odds", justify="center", width=6)
            
            for pos in open_positions:
                match_name = f"{pos.get('home_team', 'Unknown')} vs {pos.get('away_team', 'Unknown')}"
                if len(match_name) > 25:
                    match_name = match_name[:22] + "..."
                
                pos_table.add_row(
                    match_name,
                    pos.get('bet_on', '').upper(),
                    f"${pos.get('stake', 0):.0f}",
                    f"{pos.get('odds', 0):.2f}"
                )
            
            panel_text = pos_table
            if len(positions) > 10:
                panel_text = f"{pos_table}\n[dim]... and {len(positions) - 10} more[/dim]"
            
            self.layout["positions"].update(Panel(panel_text, title="📈 Positions", border_style="green"))
        except Exception as e:
            self.layout["positions"].update(Panel(f"Error loading positions: {e}", title="📈 Positions", border_style="red"))
    
    def _update_markets(self):
        """Update markets panel"""
        try:
            # Get upcoming markets
            with db_manager.get_db_session() as db:
                markets = db.query(Market).filter(
                    Market.maturity_date > datetime.now(timezone.utc),
                    Market.maturity_date < datetime.now(timezone.utc) + timedelta(hours=24),
                    Market.is_finished == False,
                    Market.sport == 'Soccer'
                ).order_by(Market.maturity_date).limit(15).all()
            
            if not markets:
                self.layout["markets"].update(Panel("No upcoming markets", title="🎯 Markets", border_style="yellow"))
                return
            
            # Create markets table
            market_table = Table(show_header=True, header_style="bold yellow")
            market_table.add_column("Match", width=30, overflow="fold")
            market_table.add_column("Sport", width=10)
            market_table.add_column("Time", justify="center", width=8)
            market_table.add_column("Source", width=12)
            
            for market in markets[:10]:
                # Calculate time to match
                time_to_match = market.maturity_date - datetime.now(timezone.utc)
                hours = time_to_match.total_seconds() / 3600
                
                if hours < 0:
                    time_str = "LIVE"
                elif hours < 1:
                    time_str = f"{int(hours * 60)}m"
                elif hours < 24:
                    time_str = f"{hours:.1f}h"
                else:
                    time_str = f"{int(hours / 24)}d"
                
                match_name = f"{market.home_team} vs {market.away_team}"
                if len(match_name) > 30:
                    match_name = match_name[:27] + "..."
                
                market_table.add_row(
                    match_name,
                    market.sport[:10],
                    time_str,
                    market.source.replace('overtime_', '')[:12]
                )
            
            self.layout["markets"].update(Panel(market_table, title="🎯 Upcoming Markets", border_style="yellow"))
        except Exception as e:
            self.layout["markets"].update(Panel(f"Error loading markets: {e}", title="🎯 Markets", border_style="red"))
    
    def _update_logs(self):
        """Update logs panel with recent activity"""
        # This will be populated by the logger
        log_text = f"[dim]Last update: {self.last_update.strftime('%H:%M:%S')}[/dim]\n"
        log_text += "[dim]Press 'q' to quit, 'p' to pause/resume, 'h' for help[/dim]"
        self.layout["logs"].update(Panel(log_text, title="📋 Activity Log", border_style="dim"))
    
    def _update_display(self):
        """Update all display panels"""
        self._update_header()
        self._update_stats()
        self._update_positions()
        self._update_markets()
        self._update_logs()
    
    def _handle_keyboard(self):
        """Handle keyboard input in a separate thread"""
        while self.running:
            try:
                if keyboard.is_pressed('q'):
                    self.running = False
                    break
                elif keyboard.is_pressed('p'):
                    self.paused = not self.paused
                    trading_logger.info(f"Trading {'paused' if self.paused else 'resumed'}")
                    time.sleep(0.5)  # Debounce
                elif keyboard.is_pressed('h'):
                    self._show_help()
                    time.sleep(0.5)
                time.sleep(0.1)
            except:
                pass
    
    def _show_help(self):
        """Show help information"""
        help_text = """
[bold cyan]Keyboard Controls:[/bold cyan]
  q - Quit the application
  p - Pause/Resume trading
  h - Show this help

[bold yellow]Trading Parameters:[/bold yellow]
  Kelly Fraction: 25%
  Min Edge: 2%
  Max Per Bet: 1%
  Max Positions: 20
        """
        console.print(Panel(help_text, title="📚 Help", border_style="cyan"))
        input("Press Enter to continue...")
    
    def run_trading_cycle(self):
        """Run a single trading cycle"""
        if self.paused:
            return
        
        self.cycle_count += 1
        log_cycle(self.cycle_count, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        try:
            # Check stop loss status
            stop_status = self.stop_loss_manager.get_stop_status()
            if stop_status['is_stopped']:
                log_stop_loss(stop_status)
                return
            
            # Check if we can resume trading
            can_resume, reason = self.stop_loss_manager.can_resume_trading()
            if not can_resume:
                trading_logger.warning(f"Cannot resume: {reason}")
                return
            
            # Get markets and calculate edges
            with db_manager.get_db_session() as db:
                markets = db.query(Market).filter(
                    Market.maturity_date > datetime.now(timezone.utc),
                    Market.maturity_date < datetime.now(timezone.utc) + timedelta(hours=72),
                    Market.is_finished == False,
                    Market.sport == 'Soccer'
                ).order_by(Market.maturity_date).limit(100).all()
                
                if not markets:
                    trading_logger.info("No markets found")
                    return
                
                # Convert to market data format
                market_data = []
                for market in markets:
                    odds = db.query(Odd).filter(
                        Odd.source_id == market.source_id
                    ).order_by(Odd.updated_at.desc()).limit(3).all()
                    
                    if len(odds) >= 3:
                        home_odd = next((o for o in odds if 'home' in o.outcome.lower()), None)
                        draw_odd = next((o for o in odds if 'draw' in o.outcome.lower()), None)
                        away_odd = next((o for o in odds if 'away' in o.outcome.lower()), None)
                        
                        if home_odd and draw_odd and away_odd:
                            market_data.append({
                                'market_id': market.source_id,
                                'home_team': market.home_team,
                                'away_team': market.away_team,
                                'sport': market.sport,
                                'maturity_date': market.maturity_date,
                                'home_odds': float(home_odd.decimal_odds),
                                'draw_odds': float(draw_odd.decimal_odds),
                                'away_odds': float(away_odd.decimal_odds),
                                'source': market.source
                            })
                
                if not market_data:
                    trading_logger.info("No markets with complete odds")
                    return
                
                # Calculate edges
                raw_signals = self.edge_calculator.calculate_edges(market_data)
                
                # Transform signals
                signals = []
                for i, raw_signal in enumerate(raw_signals):
                    market = market_data[i]
                    edges = raw_signal.get('edge', {})
                    
                    signal = {
                        'market_id': raw_signal.get('market_id'),
                        'home_edge': edges.get('home', 0),
                        'draw_edge': edges.get('draw', 0),
                        'away_edge': edges.get('away', 0),
                        'home_odds': market.get('home_odds'),
                        'draw_odds': market.get('draw_odds'),
                        'away_odds': market.get('away_odds'),
                        'home_implied_prob': 1.0 / market.get('home_odds', 1),
                        'draw_implied_prob': 1.0 / market.get('draw_odds', 1),
                        'away_implied_prob': 1.0 / market.get('away_odds', 1),
                    }
                    signals.append(signal)
                
                # Log market overview
                log_markets(market_data, signals)
                
                # Get current bankroll
                current_bankroll = float(self.session_manager.get_session(self.session_id)['current_bankroll'])
                self.strategy_config['bankroll'] = current_bankroll
                self.portfolio_engine.strategy_config = self.strategy_config
                
                # Execute trades
                result = self.portfolio_engine.execute_portfolio_trades(
                    self.session_id, market_data, signals, current_bankroll
                )
                
                trades = result.get('trades', [])
                if trades:
                    self.total_trades += len(trades)
                    for trade in trades:
                        log_trade(trade)
                else:
                    trading_logger.info("No trades executed")
                
                # Update display
                self.last_update = datetime.now()
                
        except Exception as e:
            trading_logger.error(f"Error in trading cycle: {e}", exc_info=True)
    
    def run(self):
        """Main run loop"""
        # Start keyboard handler thread
        keyboard_thread = threading.Thread(target=self._handle_keyboard, daemon=True)
        keyboard_thread.start()
        
        trading_logger.info("🚀 Terminal Trading Monitor Started")
        trading_logger.info(f"Session: {self.session_id}")
        trading_logger.info(f"Initial bankroll: ${self.strategy_config['bankroll']:,.2f}")
        
        # Main loop with live display
        with Live(self.layout, refresh_per_second=1, screen=True) as live:
            while self.running:
                try:
                    # Update display
                    self._update_display()
                    
                    # Run trading cycle every 30 seconds
                    if not self.paused and (datetime.now() - self.last_update).total_seconds() >= 30:
                        self.run_trading_cycle()
                    
                    time.sleep(1)
                    
                except KeyboardInterrupt:
                    self.running = False
                    break
                except Exception as e:
                    trading_logger.error(f"Display error: {e}")
                    time.sleep(1)
        
        # Cleanup
        self.stop_loss_manager.stop_monitoring()
        
        # Final performance report
        try:
            performance = self.session_manager.get_enhanced_performance_analytics(self.session_id)
            if performance:
                log_performance(performance)
        except:
            pass
        
        trading_logger.info("✅ Terminal Trading Monitor Stopped")

def main():
    """Main entry point"""
    monitor = TerminalTradingMonitor()
    monitor.run()

if __name__ == "__main__":
    main()