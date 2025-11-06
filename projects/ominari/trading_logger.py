#!/usr/bin/env python3
"""
Enhanced Trading Logger with Color Support and Structured Output
"""

import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
import os

# Initialize rich console
console = Console()

class TradingLogFormatter(logging.Formatter):
    """Custom formatter for trading logs with emojis and structure"""
    
    EMOJIS = {
        'trade': '💰',
        'position': '📈',
        'market': '🎯',
        'performance': '📊',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️',
        'success': '✅',
        'cycle': '🔄',
        'stop_loss': '🛑',
        'portfolio': '💼',
        'signal': '📡',
        'edge': '🔍',
        'cash': '💵'
    }
    
    def format(self, record):
        # Add emoji based on message content
        emoji = self.EMOJIS.get('info')
        message = record.getMessage().lower()
        
        if 'error' in message:
            emoji = self.EMOJIS['error']
        elif 'trade' in message or 'executed' in message:
            emoji = self.EMOJIS['trade']
        elif 'position' in message:
            emoji = self.EMOJIS['position']
        elif 'market' in message:
            emoji = self.EMOJIS['market']
        elif 'performance' in message or 'roi' in message:
            emoji = self.EMOJIS['performance']
        elif 'warning' in message or 'stop' in message:
            emoji = self.EMOJIS['warning']
        elif 'cycle' in message:
            emoji = self.EMOJIS['cycle']
        elif 'portfolio' in message:
            emoji = self.EMOJIS['portfolio']
        elif 'signal' in message:
            emoji = self.EMOJIS['signal']
        elif 'edge' in message:
            emoji = self.EMOJIS['edge']
        
        # Format with emoji
        record.msg = f"{emoji} {record.msg}"
        return super().format(record)

class TradingLogger:
    """Enhanced trading logger with structured output"""
    
    def __init__(self, name: str = "TradingSystem", log_file: Optional[str] = None):
        self.name = name
        self.console = console
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Add rich handler for console output
        rich_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
            log_time_format="%H:%M:%S"
        )
        rich_handler.setLevel(logging.INFO)
        self.logger.addHandler(rich_handler)
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_formatter = TradingLogFormatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def log_trade(self, trade: Dict[str, Any]):
        """Log a trade execution with formatted output"""
        trade_text = f"[bold green]TRADE EXECUTED[/bold green]: {trade['bet_on'].upper()} on {trade['home_team']} vs {trade['away_team']}"
        
        # Create trade details table
        table = Table(show_header=False, box=None, padding=0)
        table.add_column(style="cyan", width=12)
        table.add_column(style="white")
        
        table.add_row("Stake:", f"${trade['stake']:.2f}")
        table.add_row("Odds:", f"{trade['odds']:.2f}")
        table.add_row("Edge:", f"{trade.get('edge', 0):.1f}%")
        table.add_row("Expected:", f"${trade['stake'] * trade['odds']:.2f}")
        
        self.console.print(trade_text)
        self.console.print(table)
        self.console.print()
    
    def log_position_update(self, positions: list):
        """Log current positions in a formatted table"""
        if not positions:
            return
        
        table = Table(title="📈 Open Positions", show_header=True, header_style="bold cyan")
        table.add_column("Match", style="white", width=30)
        table.add_column("Side", style="yellow", width=6)
        table.add_column("Stake", justify="right", style="green", width=10)
        table.add_column("Odds", justify="center", style="cyan", width=8)
        table.add_column("Time", justify="center", style="magenta", width=10)
        
        for pos in positions[:10]:  # Show top 10
            time_str = self._format_time_to_match(pos.get('kickoff_time'))
            table.add_row(
                f"{pos.get('home_team', '')} vs {pos.get('away_team', '')}",
                pos.get('bet_on', '').upper(),
                f"${pos.get('stake', 0):.2f}",
                f"{pos.get('odds', 0):.2f}",
                time_str
            )
        
        self.console.print(table)
        if len(positions) > 10:
            self.console.print(f"[dim]... and {len(positions) - 10} more positions[/dim]")
    
    def log_performance_stats(self, stats: Dict[str, Any]):
        """Log performance statistics in a formatted panel"""
        # Create stats text
        stats_lines = []
        
        # Overview stats
        overview = stats.get('overview', {})
        stats_lines.append(f"[bold]Total Trades:[/bold] {overview.get('total_trades', 0)}")
        stats_lines.append(f"[bold]Win Rate:[/bold] {overview.get('win_rate', 0):.1%}")
        stats_lines.append(f"[bold]Current Bankroll:[/bold] ${overview.get('ending_bankroll', 0):,.2f}")
        
        # Financial stats
        financial = stats.get('financial', {})
        roi = financial.get('roi', 0)
        roi_color = "green" if roi > 0 else "red" if roi < 0 else "white"
        stats_lines.append(f"[bold]ROI:[/bold] [{roi_color}]{roi:.2%}[/{roi_color}]")
        
        pnl = financial.get('total_pnl', 0)
        pnl_color = "green" if pnl > 0 else "red" if pnl < 0 else "white"
        stats_lines.append(f"[bold]Total P&L:[/bold] [{pnl_color}]${pnl:,.2f}[/{pnl_color}]")
        
        # Risk metrics
        stats_lines.append(f"[bold]Sharpe Ratio:[/bold] {financial.get('sharpe_ratio', 0):.2f}")
        stats_lines.append(f"[bold]Max Drawdown:[/bold] {financial.get('max_drawdown', 0):.1%}")
        
        # Create panel
        panel = Panel(
            "\n".join(stats_lines),
            title="📊 Performance Summary",
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def log_market_overview(self, markets: list, signals: list):
        """Log market overview with signals"""
        if not markets:
            return
        
        # Group by sport
        sport_counts = {}
        for market in markets:
            sport = market.get('sport', 'Unknown')
            sport_counts[sport] = sport_counts.get(sport, 0) + 1
        
        # Create overview text
        overview = f"[bold]Markets Found:[/bold] {len(markets)} | "
        overview += " | ".join([f"{sport}: {count}" for sport, count in sport_counts.items()])
        
        self.console.print(Panel(overview, title="🎯 Market Overview", border_style="yellow"))
        
        # Show top opportunities if signals exist
        if signals:
            top_signals = sorted(signals, key=lambda x: max(
                x.get('home_edge', -100),
                x.get('draw_edge', -100),
                x.get('away_edge', -100)
            ), reverse=True)[:5]
            
            if top_signals:
                table = Table(title="🔥 Top Opportunities", show_header=True, header_style="bold yellow")
                table.add_column("Match", style="white", width=30)
                table.add_column("Best Edge", justify="center", style="green", width=12)
                table.add_column("Side", style="cyan", width=6)
                table.add_column("Odds", justify="center", style="yellow", width=8)
                
                for signal in top_signals:
                    market = next((m for m in markets if m.get('market_id') == signal.get('market_id')), {})
                    
                    # Find best edge
                    edges = {
                        'home': signal.get('home_edge', -100),
                        'draw': signal.get('draw_edge', -100),
                        'away': signal.get('away_edge', -100)
                    }
                    best_side = max(edges, key=edges.get)
                    best_edge = edges[best_side]
                    
                    if best_edge > 0:
                        odds = signal.get(f'{best_side}_odds', 0)
                        table.add_row(
                            f"{market.get('home_team', '')} vs {market.get('away_team', '')}",
                            f"{best_edge:.1f}%",
                            best_side.upper(),
                            f"{odds:.2f}"
                        )
                
                self.console.print(table)
    
    def _format_time_to_match(self, kickoff_time):
        """Format time until match starts"""
        if not kickoff_time:
            return "Unknown"
        
        try:
            if isinstance(kickoff_time, str):
                kickoff = datetime.fromisoformat(kickoff_time.replace('Z', '+00:00'))
            else:
                kickoff = kickoff_time
            
            now = datetime.now(timezone.utc)
            delta = kickoff - now
            hours = delta.total_seconds() / 3600
            
            if hours < 0:
                return "LIVE"
            elif hours < 1:
                return f"{int(hours * 60)}m"
            elif hours < 24:
                return f"{hours:.1f}h"
            else:
                return f"{int(hours / 24)}d"
        except:
            return "Unknown"
    
    def log_stop_loss_status(self, status: Dict[str, Any]):
        """Log stop loss status with warning styling"""
        if status.get('is_stopped'):
            panel = Panel(
                f"[bold red]⛔ TRADING STOPPED[/bold red]\n\n"
                f"Reason: {status.get('reason', 'Unknown')}\n"
                f"Recovery Time: {status.get('recovery_time_remaining', 'N/A')}",
                border_style="red",
                padding=(1, 2)
            )
            self.console.print(panel)
    
    def log_cycle_start(self, cycle_num: int, timestamp: str):
        """Log trading cycle start with separator"""
        self.console.print()
        self.console.rule(f"[bold blue]🔄 Trading Cycle {cycle_num} - {timestamp}[/bold blue]")
        self.console.print()
    
    # Convenience methods that delegate to logger
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str, exc_info=False):
        self.logger.error(msg, exc_info=exc_info)
    
    def debug(self, msg: str):
        self.logger.debug(msg)

# Global logger instance
trading_logger = TradingLogger("OminariTrading", "trading_system.log")

# Export convenience functions
log_trade = trading_logger.log_trade
log_positions = trading_logger.log_position_update
log_performance = trading_logger.log_performance_stats
log_markets = trading_logger.log_market_overview
log_stop_loss = trading_logger.log_stop_loss_status
log_cycle = trading_logger.log_cycle_start