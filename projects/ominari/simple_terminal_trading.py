#!/usr/bin/env python3
"""
Simple Terminal-based Portfolio Trading System
No external dependencies - just clean streaming logs to terminal
"""

import os
import time
import signal as sig
import sys
from datetime import datetime, timezone, timedelta
import json

# Set PostgreSQL environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from portfolio_trading_engine import PortfolioTradingEngine
from paper_trading_postgres_integrated import PaperTradingSessionManager
from edge_calculator import EdgeCalculator
from stop_loss_manager import StopLossManager
from database_v2 import db_manager
from models import Market, Odd

# Global flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running
    print("\n🛑 Received shutdown signal, stopping gracefully...")
    running = False

def print_header():
    """Print system header"""
    print("=" * 80)
    print("                   OMINARI TERMINAL TRADING SYSTEM")
    print("                      Real-time Log Stream")
    print("=" * 80)
    print()

def print_session_info(session_id: str, bankroll: float):
    """Print session information"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"📋 Session Info:")
    print(f"   Session ID: {session_id[:12]}...")
    print(f"   Initial Bankroll: ${bankroll:,.2f}")
    print(f"   Start Time: {timestamp}")
    print()

def print_positions_summary(positions: list):
    """Print a summary of current positions"""
    if not positions:
        return
    
    open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
    if not open_positions:
        return
    
    total_stake = sum(float(p['stake']) for p in open_positions)
    
    print(f"📈 Open Positions: {len(open_positions)} | Total Staked: ${total_stake:.2f}")
    
    # Show top 5 positions
    for i, pos in enumerate(open_positions[:5]):
        bet_on = pos.get('bet_on', '').upper()
        home_team = pos.get('home_team', 'Unknown')
        away_team = pos.get('away_team', 'Unknown')
        odds = pos.get('odds', 0)
        stake = pos.get('stake', 0)
        
        print(f"   • {bet_on} {home_team} vs {away_team} @ {odds:.2f} (${stake:.2f})")
    
    if len(open_positions) > 5:
        print(f"   ... and {len(open_positions) - 5} more")
    print()

def print_market_summary(markets: list, signals: list):
    """Print market summary with top opportunities"""
    if not markets:
        return
    
    # Count sports
    sport_counts = {}
    for market in markets:
        sport = market.get('sport', 'Unknown')
        sport_counts[sport] = sport_counts.get(sport, 0) + 1
    
    # Display summary
    summary_parts = [f"{sport}: {count}" for sport, count in sport_counts.items()]
    print(f"🎯 Markets: {len(markets)} found | {' | '.join(summary_parts)}")
    
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
            print("\n🔥 Top Opportunities:")
            for opp in top_edges[:3]:
                outcome = opp['outcome'].upper()
                home_team = opp['market'].get('home_team', '')
                away_team = opp['market'].get('away_team', '')
                odds = opp['odds']
                edge = opp['edge']
                print(f"   • {outcome} {home_team} vs {away_team} @ {odds:.2f} (+{edge:.1f}%)")
    print()

def print_performance_update(session_manager, session_id: str, total_trades: int):
    """Print performance update"""
    try:
        session = session_manager.get_session(session_id)
        current_bankroll = session.get('current_bankroll', 0)
        
        performance_parts = [f"💰 ${current_bankroll:,.2f}"]
        
        try:
            performance = session_manager.get_enhanced_performance_analytics(session_id)
            if performance:
                overview = performance.get('overview', {})
                financial = performance.get('financial', {})
                
                win_rate = overview.get('win_rate', 0)
                performance_parts.append(f"WR: {win_rate:.0%}")
                
                roi = financial.get('roi', 0)
                roi_sign = "+" if roi >= 0 else ""
                performance_parts.append(f"ROI: {roi_sign}{roi:.1f}%")
        except:
            pass
        
        performance_parts.append(f"Trades: {total_trades}")
        
        print(f"📊 Performance: {' | '.join(performance_parts)}")
        
    except Exception as e:
        print(f"❌ Error loading performance: {e}")

def print_cycle_header(cycle_num: int):
    """Print trading cycle header"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print()
    print("-" * 60)
    print(f"🔄 Trading Cycle {cycle_num} - {timestamp}")
    print("-" * 60)

def run_simple_terminal_trading():
    """Main trading function with simple terminal output"""
    global running
    
    # Set up signal handlers
    sig.signal(sig.SIGINT, signal_handler)
    sig.signal(sig.SIGTERM, signal_handler)
    
    # Clear screen and print header
    os.system('clear' if os.name == 'posix' else 'cls')
    print_header()
    
    # Initialize components
    print("🚀 Initializing trading system...")
    
    session_manager = PaperTradingSessionManager()
    edge_calculator = EdgeCalculator()
    
    # Get or create session
    session_id = session_manager.get_current_session()
    if not session_id:
        session_id = session_manager.create_session(initial_bankroll=10000)
        print(f"✅ Created new session: {session_id}")
    else:
        print(f"✅ Using existing session: {session_id}")
    
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
    
    print("✅ Trading system initialized")
    print("💡 Press Ctrl+C to stop gracefully")
    print()
    
    cycle_count = 0
    total_trades = 0
    
    # Main trading loop
    while running:
        cycle_count += 1
        print_cycle_header(cycle_count)
        
        try:
            # Check stop loss status
            stop_status = stop_loss_manager.get_stop_status()
            if stop_status['is_stopped']:
                print(f"⛔ Trading stopped: {stop_status['reason']}")
                print(f"Recovery time: {stop_status.get('recovery_time_remaining', 'N/A')}")
                time.sleep(30)
                continue
            
            # Check if we can resume
            can_resume, reason = stop_loss_manager.can_resume_trading()
            if not can_resume:
                print(f"⏸️  Cannot resume: {reason}")
                time.sleep(30)
                continue
            
            # Get current positions first
            positions = session_manager.get_positions(session_id)
            print_positions_summary(positions)
            
            # Get markets
            with db_manager.get_db_session() as db:
                markets = db.query(Market).filter(
                    Market.maturity_date > datetime.now(timezone.utc),
                    Market.maturity_date < datetime.now(timezone.utc) + timedelta(hours=72),
                    Market.is_finished == False,
                    Market.sport == 'Soccer'
                ).order_by(Market.maturity_date).limit(100).all()
                
                if not markets:
                    print("❌ No markets found, waiting...")
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
                    print("❌ No markets with complete odds")
                    time.sleep(60)
                    continue
                
                # Calculate edges
                print(f"🔍 Analyzing {len(market_data)} markets...")
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
                print_market_summary(market_data, signals)
                
                # Update bankroll
                current_bankroll = float(session_manager.get_session(session_id)['current_bankroll'])
                strategy_config['bankroll'] = current_bankroll
                portfolio_engine.strategy_config = strategy_config
                
                # Execute trades
                print("💰 Executing portfolio optimization...")
                result = portfolio_engine.execute_portfolio_trades(
                    session_id, market_data, signals, current_bankroll
                )
                
                trades = result.get('trades', [])
                if trades:
                    print(f"✅ Executed {len(trades)} trades:")
                    total_trades += len(trades)
                    
                    for trade in trades:
                        bet_on = trade['bet_on'].upper()
                        home_team = trade['home_team']
                        away_team = trade['away_team']
                        odds = trade['odds']
                        stake = trade['stake']
                        edge = trade.get('edge', 0)
                        
                        print(f"   • {bet_on} {home_team} vs {away_team} @ {odds:.2f} | ${stake:.2f} (edge: +{edge:.1f}%)")
                else:
                    print("❌ No trades executed (no positive edge found)")
                
                # Performance update
                print()
                print_performance_update(session_manager, session_id, total_trades)
                
        except Exception as e:
            print(f"❌ Error in cycle: {e}")
            import traceback
            traceback.print_exc()
        
        # Wait before next cycle
        if running:
            print()
            print("⏱️  Next cycle in 30 seconds...")
            for _ in range(30):
                if not running:
                    break
                time.sleep(1)
    
    # Cleanup
    print()
    print("🛑 Shutting down...")
    stop_loss_manager.stop_monitoring()
    
    # Final report
    try:
        print()
        print("=" * 60)
        print("📊 FINAL PERFORMANCE REPORT")
        print("=" * 60)
        
        performance = session_manager.get_enhanced_performance_analytics(session_id)
        
        if performance:
            overview = performance.get('overview', {})
            financial = performance.get('financial', {})
            
            print(f"Total Trades: {overview.get('total_trades', 0)}")
            print(f"Win Rate: {overview.get('win_rate', 0):.1%}")
            print(f"ROI: {financial.get('roi', 0):.2%}")
            print(f"Total P&L: ${financial.get('total_pnl', 0):,.2f}")
            print(f"Sharpe Ratio: {financial.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {financial.get('max_drawdown', 0):.1%}")
            print(f"Final Bankroll: ${overview.get('ending_bankroll', 0):,.2f}")
            
    except Exception as e:
        print(f"Error generating final report: {e}")
    
    print()
    print("✅ Terminal trading system stopped successfully")

if __name__ == "__main__":
    run_simple_terminal_trading()