#!/usr/bin/env python3
"""
Run Portfolio Trading System Continuously
Uses Kelly criterion optimization and stop loss protection
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta
import signal as sig
import sys

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
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running
    logger.info("\n⚠️  Received shutdown signal, stopping gracefully...")
    running = False

def run_continuous_trading():
    """Run the portfolio trading system continuously"""
    global running
    
    # Set up signal handlers
    sig.signal(sig.SIGINT, signal_handler)
    sig.signal(sig.SIGTERM, signal_handler)
    
    logger.info("🚀 Starting Portfolio Trading System")
    logger.info("=" * 80)
    
    # Initialize components
    session_manager = PaperTradingSessionManager()
    edge_calculator = EdgeCalculator()
    
    # Get or create active session
    session_id = session_manager.get_current_session()
    if not session_id:
        session_id = session_manager.create_session(initial_bankroll=10000)
        logger.info(f"Created new session: {session_id}")
    else:
        logger.info(f"Using existing session: {session_id}")
    
    # Get session data
    session = session_manager.get_session(session_id)
    logger.info(f"Current bankroll: ${session['current_bankroll']:,.2f}")
    
    # Trading configuration
    strategy_config = {
        'bankroll': float(session['current_bankroll']),
        'kelly_fraction': 0.25,      # Conservative 25% Kelly
        'min_edge': 0.02,            # Minimum 2% edge
        'cap_per_bet': 0.01,         # Max 1% per bet
        'cap_per_game': 0.02,        # Max 2% per game
        'min_bet': 10,               # Minimum $10 bet
        'max_positions': 20          # Increased to 20 positions
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
    
    logger.info(f"✅ Portfolio trading engine initialized")
    logger.info(f"✅ Stop loss protection active")
    logger.info(f"📊 Strategy Config: {strategy_config}")
    
    cycle_count = 0
    total_trades = 0
    
    # Main trading loop
    while running:
        cycle_count += 1
        logger.info(f"\n🔄 Trading Cycle {cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Check stop loss status
            stop_status = stop_loss_manager.get_stop_status()
            if stop_status['is_stopped']:
                logger.warning(f"⛔ Trading stopped: {stop_status['reason']}")
                logger.info(f"Recovery time remaining: {stop_status.get('recovery_time_remaining', 'N/A')}")
                time.sleep(30)  # Wait 30 seconds before checking again
                continue
            
            # Check if we can resume trading
            can_resume, reason = stop_loss_manager.can_resume_trading()
            if not can_resume:
                logger.info(f"⏸️  Cannot resume yet: {reason}")
                time.sleep(30)
                continue
            
            # Get upcoming markets - SOCCER ONLY
            with db_manager.get_db_session() as db:
                markets = db.query(Market).filter(
                    Market.maturity_date > datetime.now(timezone.utc),
                    Market.maturity_date < datetime.now(timezone.utc) + timedelta(hours=72),  # Look 72 hours ahead
                    Market.is_finished == False,
                    Market.sport == 'Soccer'  # Soccer matches only
                ).order_by(Market.maturity_date).limit(100).all()  # Increased limit
                
                logger.info(f"📊 Found {len(markets)} upcoming markets")
                
                if not markets:
                    logger.info("No markets found, waiting...")
                    time.sleep(60)  # Wait 1 minute
                    continue
                
                # Convert markets to dict format for portfolio engine
                market_data = []
                for market in markets:
                    # Get latest odds
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
                    logger.info("No markets with complete odds, waiting...")
                    time.sleep(60)
                    continue
                
                logger.info(f"🎯 Analyzing {len(market_data)} markets with complete odds")
                
                # Get current positions to check exposure
                positions = session_manager.get_positions(session_id)
                open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
                current_exposure = sum(float(p['stake']) for p in open_positions)
                current_bankroll = float(session_manager.get_session(session_id)['current_bankroll'])
                exposure_pct = (current_exposure / current_bankroll * 100) if current_bankroll > 0 else 0
                
                logger.info(f"💰 Current exposure: ${current_exposure:.2f} ({exposure_pct:.1f}% of bankroll)")
                logger.info(f"📈 Open positions: {len(open_positions)}")
                
                # Update strategy config with current bankroll
                strategy_config['bankroll'] = current_bankroll
                portfolio_engine.strategy_config = strategy_config
                
                # Calculate edges for markets
                raw_signals = edge_calculator.calculate_edges(market_data)
                
                # Transform signals to match portfolio engine format
                signals = []
                for i, raw_signal in enumerate(raw_signals):
                    market = market_data[i]
                    edges = raw_signal.get('edge', {})
                    
                    # Create signal in expected format
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
                        'probability': raw_signal.get('probability'),
                        'confidence': raw_signal.get('confidence')
                    }
                    signals.append(signal)
                
                # Debug log the signals
                if signals:
                    logger.info(f"🔍 Edge calculation results:")
                    for i, signal in enumerate(signals[:3]):  # Show first 3
                        logger.info(f"   Market {i+1}: home_edge={signal['home_edge']:.1f}%, draw_edge={signal['draw_edge']:.1f}%, away_edge={signal['away_edge']:.1f}%")
                else:
                    logger.info("🔍 No signals generated from edge calculator")
                
                # Run portfolio optimization
                result = portfolio_engine.execute_portfolio_trades(
                    session_id, market_data, signals, current_bankroll
                )
                trades = result.get('trades', [])
                
                if trades:
                    logger.info(f"✅ Executed {len(trades)} trades this cycle")
                    total_trades += len(trades)
                    
                    # Log each trade
                    for trade in trades:
                        logger.info(f"   • {trade['market_id']}: {trade['bet_type']} @ {trade['odds']:.2f} - ${trade['stake']:.2f}")
                else:
                    logger.info("❌ No trades executed this cycle")
                
                # Performance update
                try:
                    performance = session_manager.get_enhanced_performance_analytics(session_id)
                    if performance and 'overview' in performance:
                        logger.info(f"\n📊 Performance Update:")
                        logger.info(f"   Total Trades: {performance['overview']['total_trades']}")
                        logger.info(f"   Win Rate: {performance['overview']['win_rate']:.1%}")
                        logger.info(f"   ROI: {performance['financial']['roi']:.2%}")
                        logger.info(f"   Current Bankroll: ${current_bankroll:,.2f}")
                except:
                    # Simple performance log if analytics not available
                    logger.info(f"\n📊 Performance Update:")
                    logger.info(f"   Total Trades: {total_trades}")
                    logger.info(f"   Current Bankroll: ${current_bankroll:,.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {e}", exc_info=True)
        
        # Wait before next cycle (30 seconds)
        logger.info(f"\n⏱️  Waiting 30 seconds before next cycle...")
        time.sleep(30)
        
        # Check if we should stop
        if not running:
            break
    
    # Cleanup on shutdown
    logger.info("\n" + "=" * 80)
    logger.info("🛑 Shutting down Portfolio Trading System")
    
    # Stop monitoring
    stop_loss_manager.stop_monitoring()
    
    # Final performance report
    try:
        final_performance = session_manager.get_enhanced_performance_analytics(session_id)
        if final_performance and 'overview' in final_performance:
            logger.info(f"\n📊 Final Performance Report:")
            logger.info(f"   Total Trades: {final_performance['overview']['total_trades']}")
            logger.info(f"   Win Rate: {final_performance['overview']['win_rate']:.1%}")
            logger.info(f"   ROI: {final_performance['financial']['roi']:.2%}")
            logger.info(f"   Profit Factor: {final_performance['financial']['profit_factor']:.2f}")
            logger.info(f"   Sharpe Ratio: {final_performance['financial']['sharpe_ratio']:.2f}")
            logger.info(f"   Max Drawdown: {final_performance['financial']['max_drawdown']:.1%}")
            logger.info(f"   Final Bankroll: ${final_performance['overview']['ending_bankroll']:,.2f}")
        else:
            # Fallback to basic metrics
            metrics = session_manager.get_performance_metrics(session_id)
            if metrics:
                logger.info(f"\n📊 Final Performance Report:")
                logger.info(f"   Total Trades: {metrics.get('total_bets', 0)}")
                logger.info(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
                logger.info(f"   ROI: {metrics.get('roi', 0):.2%}")
    except Exception as e:
        logger.error(f"Error generating final report: {e}")
    
    logger.info("\n✅ Trading system stopped successfully")

if __name__ == "__main__":
    run_continuous_trading()