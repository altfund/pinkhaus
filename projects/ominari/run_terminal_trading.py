#!/usr/bin/env python3
"""
Terminal-based Portfolio Trading System with Streaming Logs
Simple version focused on clean log output
"""

import os
import time
import signal
import sys
from datetime import datetime, timezone, timedelta

# Set PostgreSQL environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import track

from portfolio_trading_engine import PortfolioTradingEngine
from paper_trading_postgres_integrated import PaperTradingSessionManager
from edge_calculator import EdgeCalculator
from stop_loss_manager import StopLossManager
from database_v2 import db_manager
from models import Market, Odd
from trading_logger import TradingLogger

# Initialize console and logger
console = Console()
logger = TradingLogger("TerminalTrading", "terminal_trading.log")

# Global flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running
    console.print("\n[bold red]⚠️  Received shutdown signal, stopping gracefully...[/bold red]")
    running = False

def print_header():
    """Print system header"""
    header = """
[bold cyan]╔══════════════════════════════════════════════════════════════╗
║               OMINARI TERMINAL TRADING SYSTEM                ║
║                    Real-time Log Stream                      ║
╚══════════════════════════════════════════════════════════════╝[/bold cyan]
"""
    console.print(header)

def print_session_info(session_id: str, bankroll: float):
    """Print session information"""
    info_table = Table(show_header=False, box=None)
    info_table.add_column(style="cyan", width=20)
    info_table.add_column(style="white")
    
    info_table.add_row("Session ID:", session_id[:12] + "...")
    info_table.add_row("Initial Bankroll:", f"${bankroll:,.2f}")
    info_table.add_row("Start Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    console.print(Panel(info_table, title="📋 Session Info", border_style="blue"))

def display_positions_summary(positions: list):
    """Display a summary of current positions"""
    if not positions:
        return
    
    open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
    if not open_positions:
        return
    
    total_stake = sum(float(p['stake']) for p in open_positions)
    
    # Create summary line
    console.print(f"\n[bold]📈 Positions:[/bold] {len(open_positions)} open | ${total_stake:.2f} staked")
    
    # Show top 5 positions in compact format
    for i, pos in enumerate(open_positions[:5]):
        bet_on_color = {"home": "green", "draw": "yellow", "away": "red"}.get(pos.get('bet_on', ''), "white")
        console.print(
            f"  • [{bet_on_color}]{pos.get('bet_on', '').upper()}[/{bet_on_color}] "
            f"{pos.get('home_team', 'Unknown')} vs {pos.get('away_team', 'Unknown')} "
            f"@ {pos.get('odds', 0):.2f} (${pos.get('stake', 0):.2f})"
        )
    
    if len(open_positions) > 5:
        console.print(f"  [dim]... and {len(open_positions) - 5} more[/dim]")

def display_market_summary(markets: list, signals: list):
    """Display market summary with top opportunities"""
    if not markets:
        return
    
    # Count sports
    sport_counts = {}
    for market in markets:
        sport = market.get('sport', 'Unknown')
        sport_counts[sport] = sport_counts.get(sport, 0) + 1
    
    # Display summary
    summary = f"[bold]🎯 Markets:[/bold] {len(markets)} found"
    for sport, count in sport_counts.items():
        summary += f" | {sport}: {count}"
    console.print(f"\n{summary}")
    
    # Show top 3 opportunities
    if signals:
        top_edges = []
        for i, signal in enumerate(signals):
            market = markets[i] if i < len(markets) else {}
            for outcome in ['home', 'draw', 'away']:
                edge = signal.get(f'{outcome}_edge', 0)
                if edge > 0:
                    top_edges.append({
                        'market': market,
                        'outcome': outcome,
                        'edge': edge,
                        'odds': signal.get(f'{outcome}_odds', 0)
                    })
        
        top_edges.sort(key=lambda x: x['edge'], reverse=True)
        
        if top_edges[:3]:
            console.print("\n[bold]🔥 Top Opportunities:[/bold]")
            for opp in top_edges[:3]:
                outcome_color = {"home": "green", "draw": "yellow", "away": "red"}.get(opp['outcome'], "white")
                console.print(
                    f"  • [{outcome_color}]{opp['outcome'].upper()}[/{outcome_color}] "
                    f"{opp['market'].get('home_team', '')} vs {opp['market'].get('away_team', '')} "
                    f"@ {opp['odds']:.2f} ([green]+{opp['edge']:.1f}%[/green])"
                )

def display_performance_update(session_manager, session_id: str, total_trades: int):
    """Display performance update"""
    try:
        performance = session_manager.get_enhanced_performance_analytics(session_id)
        session = session_manager.get_session(session_id)
        
        # Create performance line
        perf_parts = []
        
        # Current bankroll
        current_bankroll = session.get('current_bankroll', 0)
        perf_parts.append(f"💰 ${current_bankroll:,.2f}")
        
        # Win rate
        if performance and 'overview' in performance:
            win_rate = performance['overview'].get('win_rate', 0)
            win_color = "green" if win_rate > 0.5 else "red"
            perf_parts.append(f"[{win_color}]{win_rate:.0%} WR[/{win_color}]")
        
        # ROI
        if performance and 'financial' in performance:
            roi = performance['financial'].get('roi', 0)
            roi_color = "green" if roi > 0 else "red"
            perf_parts.append(f"[{roi_color}]{roi:+.1%} ROI[/{roi_color}]")
        
        # Total trades
        perf_parts.append(f"📊 {total_trades} trades")
        
        console.print(f"\n[bold]Performance:[/bold] {' | '.join(perf_parts)}")
        
    except Exception as e:
        console.print(f"\n[red]Error loading performance: {e}[/red]")

def run_terminal_trading():
    """Main trading function with terminal output"""
    global running
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Clear screen and print header
    console.clear()
    print_header()
    
    # Initialize components
    logger.info("Initializing trading system...")
    
    session_manager = PaperTradingSessionManager()
    edge_calculator = EdgeCalculator()
    
    # Get or create session
    session_id = session_manager.get_current_session()
    if not session_id:
        session_id = session_manager.create_session(initial_bankroll=10000)
        logger.info(f"Created new session: {session_id}")
    else:
        logger.info(f"Using existing session: {session_id}")
    
    # Get session data
    session = session_manager.get_session(session_id)
    print_session_info(session_id, float(session['initial_bankroll']))
    
    # Trading configuration
    strategy_config = {
        'bankroll': float(session['current_bankroll']),
        'kelly_fraction': 0.25,
        'min_edge': 0.02,
        'cap_per_bet': 0.01,
        'cap_per_game': 0.02,
        'min_bet': 10,
        'max_positions': 20
    }
    
    # Initialize portfolio engine
    portfolio_engine = PortfolioTradingEngine(
        session_manager, 
        edge_calculator, 
        strategy_config
    )
    
    # Initialize stop loss manager
    stop_loss_manager = StopLossManager(session_manager)
    stop_loss_config = {
        'drawdown_pct': 5,
        'time_window_minutes': 10,
        'max_daily_loss_pct': 10,
        'consecutive_losses': 3,
        'recovery_time_minutes': 60
    }
    stop_loss_manager.set_stop_loss_config(stop_loss_config)
    stop_loss_manager.start_monitoring(session_id)
    
    logger.info("✅ Trading system initialized")
    console.print(Panel("Press Ctrl+C to stop gracefully", style="dim"))
    
    cycle_count = 0
    total_trades = 0
    
    # Main trading loop
    while running:
        cycle_count += 1
        
        # Print cycle separator
        console.rule(f"[bold blue]Trading Cycle {cycle_count} - {datetime.now().strftime('%H:%M:%S')}[/bold blue]")
        
        try:
            # Check stop loss status
            stop_status = stop_loss_manager.get_stop_status()
            if stop_status['is_stopped']:
                console.print(f"[bold red]⛔ Trading stopped: {stop_status['reason']}[/bold red]")
                console.print(f"Recovery time: {stop_status.get('recovery_time_remaining', 'N/A')}")
                time.sleep(30)
                continue
            
            # Check if we can resume
            can_resume, reason = stop_loss_manager.can_resume_trading()
            if not can_resume:
                console.print(f"[yellow]⏸️  Cannot resume: {reason}[/yellow]")
                time.sleep(30)
                continue
            
            # Get current positions first
            positions = session_manager.get_positions(session_id)
            display_positions_summary(positions)
            
            # Get markets
            with db_manager.get_db_session() as db:
                markets = db.query(Market).filter(
                    Market.maturity_date > datetime.now(timezone.utc),
                    Market.maturity_date < datetime.now(timezone.utc) + timedelta(hours=72),
                    Market.is_finished == False,
                    Market.sport == 'Soccer'
                ).order_by(Market.maturity_date).limit(100).all()
                
                if not markets:
                    console.print("[dim]No markets found, waiting...[/dim]")
                    time.sleep(60)
                    continue
                
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
                    console.print("[dim]No markets with complete odds[/dim]")
                    time.sleep(60)
                    continue
                
                # Calculate edges
                console.print(f"\n[cyan]🔍 Analyzing {len(market_data)} markets...[/cyan]")
                raw_signals = edge_calculator.calculate_edges(market_data)
                
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
                
                # Display market summary
                display_market_summary(market_data, signals)
                
                # Update bankroll
                current_bankroll = float(session_manager.get_session(session_id)['current_bankroll'])
                strategy_config['bankroll'] = current_bankroll
                portfolio_engine.strategy_config = strategy_config
                
                # Execute trades
                console.print(f"\n[cyan]💰 Executing portfolio optimization...[/cyan]")
                result = portfolio_engine.execute_portfolio_trades(
                    session_id, market_data, signals, current_bankroll
                )
                
                trades = result.get('trades', [])
                if trades:
                    console.print(f"\n[bold green]✅ Executed {len(trades)} trades:[/bold green]")
                    total_trades += len(trades)
                    
                    for trade in trades:
                        bet_color = {"home": "green", "draw": "yellow", "away": "red"}.get(trade['bet_on'], "white")
                        console.print(
                            f"  • [{bet_color}]{trade['bet_on'].upper()}[/{bet_color}] "
                            f"{trade['home_team']} vs {trade['away_team']} "
                            f"@ {trade['odds']:.2f} | ${trade['stake']:.2f} "
                            f"(edge: [green]+{trade.get('edge', 0):.1f}%[/green])"
                        )
                else:
                    console.print("[yellow]No trades executed (no positive edge found)[/yellow]")
                
                # Performance update
                display_performance_update(session_manager, session_id, total_trades)
                
        except Exception as e:
            console.print(f"[red]❌ Error in cycle: {e}[/red]", style="bold red")
            logger.error(f"Cycle error: {e}", exc_info=True)
        
        # Wait before next cycle
        if running:
            console.print(f"\n[dim]Next cycle in 30 seconds...[/dim]\n")
            for _ in range(30):
                if not running:
                    break
                time.sleep(1)
    
    # Cleanup
    console.print("\n[bold yellow]Shutting down...[/bold yellow]")
    stop_loss_manager.stop_monitoring()
    
    # Final report
    try:
        console.rule("[bold]Final Performance Report[/bold]")
        performance = session_manager.get_enhanced_performance_analytics(session_id)
        
        if performance:
            final_table = Table(show_header=False, box=None)
            final_table.add_column(style="cyan", width=20)
            final_table.add_column(style="white")
            
            overview = performance.get('overview', {})
            financial = performance.get('financial', {})
            
            final_table.add_row("Total Trades:", str(overview.get('total_trades', 0)))
            final_table.add_row("Win Rate:", f"{overview.get('win_rate', 0):.1%}")
            final_table.add_row("ROI:", f"{financial.get('roi', 0):.2%}")
            final_table.add_row("Total P&L:", f"${financial.get('total_pnl', 0):,.2f}")
            final_table.add_row("Sharpe Ratio:", f"{financial.get('sharpe_ratio', 0):.2f}")
            final_table.add_row("Max Drawdown:", f"{financial.get('max_drawdown', 0):.1%}")
            final_table.add_row("Final Bankroll:", f"${overview.get('ending_bankroll', 0):,.2f}")
            
            console.print(Panel(final_table, title="📊 Final Performance", border_style="green"))
            
    except Exception as e:
        console.print(f"[red]Error generating final report: {e}[/red]")
    
    console.print("\n[bold green]✅ Terminal trading system stopped successfully[/bold green]")

if __name__ == "__main__":
    run_terminal_trading()