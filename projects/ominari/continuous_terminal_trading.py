#!/usr/bin/env python3
"""
Continuous Terminal Trading System with Time-Chunked Portfolio Optimization
"""

import os
import time
import signal as sig
import sys
from datetime import datetime, timezone, timedelta

# Set PostgreSQL environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from portfolio_trading_engine import PortfolioTradingEngine
from continuous_portfolio_optimizer import ContinuousPortfolioOptimizer
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
    print("           OMINARI CONTINUOUS PORTFOLIO TRADING SYSTEM")
    print("              Time-Chunked Optimization & Rebalancing")
    print("=" * 80)
    print()

def print_chunk_summary(chunks):
    """Print summary of time chunks"""
    if not chunks:
        return
    
    print("\n📊 Time Chunk Analysis:")
    total_markets = 0
    for chunk in chunks[:5]:  # Show first 5 chunks
        market_count = len(chunk.markets)
        total_markets += market_count
        print(f"   • {chunk.label}: {market_count} markets")
    
    if len(chunks) > 5:
        remaining = sum(len(c.markets) for c in chunks[5:])
        print(f"   • ... and {remaining} markets in {len(chunks)-5} more chunks")
    
    print(f"   Total: {total_markets} markets across {len(chunks)} time windows")

def print_rebalancing_summary(summary):
    """Print rebalancing operation summary"""
    if not summary.get('success'):
        return
    
    print("\n🔄 Portfolio Rebalancing:")
    print(f"   • Chunks analyzed: {summary.get('chunks_analyzed', 0)}")
    print(f"   • Markets evaluated: {summary.get('total_markets', 0)}")
    
    if summary.get('positions_closed', 0) > 0:
        print(f"   • Positions closed: {summary['positions_closed']} ❌")
    
    if summary.get('positions_opened', 0) > 0:
        print(f"   • New positions: {summary['positions_opened']} ✅")
    
    if summary.get('positions_adjusted', 0) > 0:
        print(f"   • Positions adjusted: {summary['positions_adjusted']} 🔄")
    
    current_exp = summary.get('current_exposure', 0)
    target_exp = summary.get('target_exposure', 0)
    
    if abs(target_exp - current_exp) > 1:
        arrow = "↑" if target_exp > current_exp else "↓"
        print(f"   • Exposure change: ${current_exp:.2f} → ${target_exp:.2f} {arrow}")

def run_continuous_trading():
    """Main continuous trading function with chunked optimization"""
    global running
    
    # Set up signal handlers
    sig.signal(sig.SIGINT, signal_handler)
    sig.signal(sig.SIGTERM, signal_handler)
    
    # Clear screen and print header
    os.system('clear' if os.name == 'posix' else 'cls')
    print_header()
    
    # Initialize components
    print("🚀 Initializing continuous trading system...")
    
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
    print(f"💰 Initial Bankroll: ${session['initial_bankroll']:,.2f}")
    print(f"💵 Current Bankroll: ${session['current_bankroll']:,.2f}")
    
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
    
    # Initialize continuous optimizer with 2-hour chunks
    continuous_optimizer = ContinuousPortfolioOptimizer(
        portfolio_engine, 
        chunk_hours=2.0
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
    
    print("✅ Continuous optimization system initialized")
    print("⏱️  Using 2-hour time chunks for portfolio grouping")
    print("💡 Press Ctrl+C to stop gracefully")
    print()
    
    cycle_count = 0
    total_trades = 0
    last_price_check = {}
    
    # Main trading loop
    while running:
        cycle_count += 1
        
        # Print cycle header
        print()
        print("-" * 60)
        print(f"🔄 Optimization Cycle {cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60)
        
        try:
            # Check stop loss status
            stop_status = stop_loss_manager.get_stop_status()
            if stop_status['is_stopped']:
                print(f"⛔ Trading stopped: {stop_status['reason']}")
                print(f"Recovery time: {stop_status.get('recovery_time_remaining', 'N/A')}")
                time.sleep(30)
                continue
            
            # Get current positions
            positions = session_manager.get_positions(session_id)
            open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
            current_exposure = sum(float(p['stake']) for p in open_positions)
            
            print(f"📈 Current Portfolio: {len(open_positions)} positions | ${current_exposure:.2f} staked")
            
            # Get markets
            with db_manager.get_db_session() as db:
                markets = db.query(Market).filter(
                    Market.maturity_date > datetime.now(timezone.utc),
                    Market.maturity_date < datetime.now(timezone.utc) + timedelta(hours=48),  # Look 48 hours ahead
                    Market.is_finished == False,
                    Market.sport == 'Soccer'
                ).order_by(Market.maturity_date).limit(200).all()  # Get more markets for chunking
                
                if not markets:
                    print("❌ No upcoming markets found")
                    time.sleep(60)
                    continue
                
                # Convert to market data format and collect current prices
                market_data = []
                current_prices = {}
                
                for market in markets:
                    odds = db.query(Odd).filter(
                        Odd.source_id == market.source_id
                    ).order_by(Odd.updated_at.desc()).limit(3).all()
                    
                    if len(odds) >= 3:
                        home_odd = next((o for o in odds if 'home' in o.outcome.lower()), None)
                        draw_odd = next((o for o in odds if 'draw' in o.outcome.lower()), None)
                        away_odd = next((o for o in odds if 'away' in o.outcome.lower()), None)
                        
                        if home_odd and draw_odd and away_odd:
                            market_dict = {
                                'market_id': market.source_id,
                                'home_team': market.home_team,
                                'away_team': market.away_team,
                                'sport': market.sport,
                                'maturity_date': market.maturity_date,
                                'home_odds': float(home_odd.decimal_odds),
                                'draw_odds': float(draw_odd.decimal_odds),
                                'away_odds': float(away_odd.decimal_odds),
                                'source': market.source
                            }
                            market_data.append(market_dict)
                            
                            # Store current prices
                            current_prices[f"{market.source_id}_home"] = market_dict['home_odds']
                            current_prices[f"{market.source_id}_draw"] = market_dict['draw_odds']
                            current_prices[f"{market.source_id}_away"] = market_dict['away_odds']
                
                if not market_data:
                    print("❌ No markets with complete odds")
                    time.sleep(60)
                    continue
                
                print(f"🎯 Found {len(market_data)} markets with complete odds")
                
                # Check if prices have changed significantly
                price_changed = False
                if last_price_check:
                    for key, new_price in current_prices.items():
                        old_price = last_price_check.get(key, new_price)
                        if abs(new_price - old_price) / old_price > 0.03:  # 3% change threshold
                            price_changed = True
                            break
                
                if price_changed:
                    print("📊 Significant price changes detected - triggering rebalance")
                
                # Calculate edges
                print(f"🔍 Calculating edges for all markets...")
                raw_signals = edge_calculator.calculate_edges(market_data)
                
                # Transform signals
                signals = []
                positive_edge_count = 0
                
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
                    
                    # Count positive edges
                    if max(edges.values()) > 0.02:  # Above minimum edge
                        positive_edge_count += 1
                
                print(f"✨ Found {positive_edge_count} markets with positive edge")
                
                # Update bankroll
                current_bankroll = float(session_manager.get_session(session_id)['current_bankroll'])
                strategy_config['bankroll'] = current_bankroll
                portfolio_engine.strategy_config = strategy_config
                continuous_optimizer.portfolio_engine.strategy_config = strategy_config
                
                # Execute continuous portfolio optimization
                print("\n💎 Executing continuous portfolio optimization...")
                result = continuous_optimizer.optimize_portfolio_continuously(
                    session_id, market_data, signals, current_bankroll
                )
                
                # Print chunk summary
                if hasattr(continuous_optimizer, 'last_portfolio_state'):
                    chunks = continuous_optimizer.last_portfolio_state.get('chunks', [])
                    print_chunk_summary(chunks)
                
                # Print rebalancing summary
                print_rebalancing_summary(result)
                
                # Print individual trades
                trades = result.get('trades', [])
                if trades:
                    print(f"\n✅ Executed {len(trades)} trades:")
                    total_trades += len(trades)
                    
                    for trade in trades[:5]:  # Show first 5
                        bet_on = trade.get('bet_on', '').upper()
                        home_team = trade.get('home_team', 'Unknown')
                        away_team = trade.get('away_team', 'Unknown')
                        odds = trade.get('odds', 0)
                        stake = trade.get('stake', 0)
                        edge = trade.get('edge', 0)
                        
                        print(f"   • {bet_on} {home_team} vs {away_team}")
                        print(f"     Odds: {odds:.2f} | Stake: ${stake:.2f} | Edge: +{edge*100:.1f}%")
                    
                    if len(trades) > 5:
                        print(f"   ... and {len(trades)-5} more trades")
                
                # Update price check
                last_price_check = current_prices
                
                # Performance summary
                session_data = session_manager.get_session(session_id)
                final_bankroll = session_data['current_bankroll']
                total_pnl = final_bankroll - float(session_data['initial_bankroll'])
                roi = (total_pnl / float(session_data['initial_bankroll'])) * 100 if session_data['initial_bankroll'] > 0 else 0
                
                print(f"\n📊 Performance: Bankroll: ${final_bankroll:,.2f} | P&L: ${total_pnl:+,.2f} | ROI: {roi:+.1f}%")
                
        except Exception as e:
            print(f"❌ Error in cycle: {e}")
            import traceback
            traceback.print_exc()
        
        # Wait before next cycle - shorter interval for continuous optimization
        if running:
            print("\n⏱️  Next optimization in 15 seconds...")
            for _ in range(15):
                if not running:
                    break
                time.sleep(1)
    
    # Cleanup
    print("\n🛑 Shutting down...")
    stop_loss_manager.stop_monitoring()
    
    # Final report
    try:
        print("\n" + "=" * 60)
        print("📊 FINAL PERFORMANCE REPORT")
        print("=" * 60)
        
        session_data = session_manager.get_session(session_id)
        final_bankroll = session_data['current_bankroll']
        initial_bankroll = float(session_data['initial_bankroll'])
        total_pnl = final_bankroll - initial_bankroll
        roi = (total_pnl / initial_bankroll) * 100 if initial_bankroll > 0 else 0
        
        print(f"Initial Bankroll: ${initial_bankroll:,.2f}")
        print(f"Final Bankroll: ${final_bankroll:,.2f}")
        print(f"Total P&L: ${total_pnl:+,.2f}")
        print(f"ROI: {roi:+.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Total Cycles: {cycle_count}")
        
    except Exception as e:
        print(f"Error generating final report: {e}")
    
    print("\n✅ Continuous trading system stopped successfully")

if __name__ == "__main__":
    run_continuous_trading()