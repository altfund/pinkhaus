#!/usr/bin/env python3
"""
Simple edge-based trading system that actually executes trades
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta

# Set PostgreSQL environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from paper_trading_postgres_integrated import PaperTradingSessionManager
from edge_calculator import EdgeCalculator
from database_v2 import db_manager
from models import Market, Odd
import signal as sig
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running
    logger.info("\n⚠️ Received shutdown signal, stopping gracefully...")
    running = False

def run_simple_trading():
    """Run a simple edge-based trading system"""
    global running
    
    # Set up signal handlers
    sig.signal(sig.SIGINT, signal_handler)
    sig.signal(sig.SIGTERM, signal_handler)
    
    logger.info("🚀 Starting Simple Edge Trading System")
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
    
    # Trading parameters
    MIN_EDGE = 5.0  # 5% minimum edge
    MAX_STAKE_PCT = 1.0  # Max 1% of bankroll per bet
    MIN_STAKE = 10.0  # Minimum $10 bet
    
    cycle_count = 0
    total_trades = 0
    
    while running:
        cycle_count += 1
        logger.info(f"\n🔄 Trading Cycle {cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Get current session info
            session = session_manager.get_session(session_id)
            current_bankroll = float(session['current_bankroll'])
            
            # Get upcoming markets
            with db_manager.get_db_session() as db:
                markets = db.query(Market).filter(
                    Market.maturity_date > datetime.now(timezone.utc),
                    Market.maturity_date < datetime.now(timezone.utc) + timedelta(days=7),
                    Market.is_finished == False
                ).limit(10).all()
                
                logger.info(f"📊 Found {len(markets)} upcoming markets")
                
                if not markets:
                    logger.info("No markets found, waiting...")
                    time.sleep(30)
                    continue
                
                # Prepare market data for edge calculation
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
                                'away_odds': float(away_odd.decimal_odds)
                            })
                
                if not market_data:
                    logger.info("No markets with complete odds")
                    time.sleep(30)
                    continue
                
                logger.info(f"🎯 Analyzing {len(market_data)} markets")
                
                # Calculate edges
                signals = edge_calculator.calculate_edges(market_data)
                
                # Find best opportunities
                trades_this_cycle = 0
                for i, signal in enumerate(signals):
                    market = market_data[i]
                    edges = signal.get('edge', {})
                    
                    # Check each outcome for positive edge
                    for outcome in ['home', 'draw', 'away']:
                        edge = edges.get(outcome, 0)
                        
                        if edge >= MIN_EDGE:
                            # Calculate stake (simple fixed percentage)
                            stake = min(
                                current_bankroll * MAX_STAKE_PCT / 100,
                                100  # Max $100 per bet
                            )
                            
                            if stake < MIN_STAKE:
                                continue
                            
                            # Get odds for this outcome
                            odds = market.get(f'{outcome}_odds', 0)
                            
                            # Create trade
                            trade = {
                                'match_id': market['market_id'],
                                'match_name': f"{market['home_team']} vs {market['away_team']}",
                                'bet_type': outcome,
                                'stake': stake,
                                'odds': odds,
                                'probability': signal.get('probability', 0.5),
                                'kickoff_time': market['maturity_date'],
                                'signal_name': 'simple_edge',
                                'signal_value': edge / 100,
                                'edge': edge
                            }
                            
                            logger.info(f"\n💰 PLACING BET:")
                            logger.info(f"   Market: {trade['match_name']}")
                            logger.info(f"   Bet: {outcome.upper()} @ {odds:.2f}")
                            logger.info(f"   Edge: {edge:.1f}%")
                            logger.info(f"   Stake: ${stake:.2f}")
                            
                            # Execute trade
                            session_manager.record_trades(session_id, [trade])
                            trades_this_cycle += 1
                            total_trades += 1
                            
                            # Update bankroll for next bet
                            current_bankroll -= stake
                            
                            # Limit trades per cycle
                            if trades_this_cycle >= 3:
                                break
                    
                    if trades_this_cycle >= 3:
                        break
                
                if trades_this_cycle == 0:
                    logger.info("❌ No trades met edge criteria")
                else:
                    logger.info(f"✅ Placed {trades_this_cycle} trades")
                
                # Show current status
                positions = session_manager.get_positions(session_id)
                open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
                total_exposure = sum(float(p['stake']) for p in open_positions)
                
                logger.info(f"\n📊 Portfolio Status:")
                logger.info(f"   Open Positions: {len(open_positions)}")
                logger.info(f"   Total Exposure: ${total_exposure:.2f}")
                logger.info(f"   Available Bankroll: ${current_bankroll:.2f}")
                logger.info(f"   Total Trades Today: {total_trades}")
        
        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {e}")
        
        # Wait before next cycle
        logger.info(f"\n⏱️ Waiting 60 seconds...")
        for _ in range(60):
            if not running:
                break
            time.sleep(1)
    
    # Cleanup
    logger.info("\n" + "=" * 80)
    logger.info("🛑 Simple Edge Trading System Stopped")
    logger.info(f"Total trades executed: {total_trades}")
    logger.info("✅ Shutdown complete")

if __name__ == "__main__":
    run_simple_trading()