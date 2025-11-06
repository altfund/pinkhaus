#!/usr/bin/env python3
"""Settle test positions to demonstrate P&L system"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from paper_trading_postgres_integrated import PaperTradingSessionManager
from database_v2 import db_manager
from models import Market
from datetime import datetime, timezone
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session manager
sm = PaperTradingSessionManager()
session_id = sm.get_current_session()

# Get current positions
positions = sm.get_positions(session_id)
open_positions = [p for p in positions if p['status'] in ['pending', 'open']]

logger.info(f"Found {len(open_positions)} open positions to settle")

# Settle some positions as wins/losses
if open_positions:
    # Let's settle the first 3 positions
    positions_to_settle = open_positions[:3]
    
    for i, pos in enumerate(positions_to_settle):
        # Simulate 60% win rate
        is_win = random.random() < 0.6
        
        position_id = pos.get('position_id', pos.get('bet_id', pos.get('id')))
        
        logger.info(f"\nSettling position {i+1}:")
        logger.info(f"  Market: {pos.get('match_id', 'Unknown')}")
        logger.info(f"  Bet: {pos.get('bet_type', pos.get('bet_on', 'Unknown'))} @ {pos['odds']}")
        logger.info(f"  Stake: ${pos['stake']:.2f}")
        logger.info(f"  Result: {'WIN' if is_win else 'LOSS'}")
        
        # Settle the position
        sm.settle_position(
            position_id=position_id,
            is_winner=is_win,
            final_odds=pos['odds']
        )
        
        if is_win:
            profit = float(pos['stake']) * (float(pos['odds']) - 1)
            logger.info(f"  Profit: ${profit:.2f}")
        else:
            logger.info(f"  Loss: ${pos['stake']:.2f}")
    
    # Mark the corresponding markets as finished
    with db_manager.get_db_session() as db:
        for pos in positions_to_settle:
            market = db.query(Market).filter(
                Market.source_id == pos.get('match_id', pos.get('market_id'))
            ).first()
            
            if market:
                market.is_finished = True
                logger.info(f"Marked market {market.source_id} as finished")
        
        db.commit()
    
    logger.info("\n✅ Settlement complete!")
    
    # Show updated performance
    session = sm.get_session(session_id)
    logger.info(f"\n📊 UPDATED PERFORMANCE:")
    logger.info(f"Initial Bankroll: ${session['initial_bankroll']:,.2f}")
    logger.info(f"Current Bankroll: ${session['current_bankroll']:,.2f}")
    
    # Get performance metrics
    try:
        perf = sm.get_enhanced_performance_analytics(session_id)
        if perf and 'overview' in perf:
            logger.info(f"Total Trades: {perf['overview']['total_trades']}")
            logger.info(f"Wins: {perf['overview']['wins']}")
            logger.info(f"Losses: {perf['overview']['losses']}")
            logger.info(f"Win Rate: {perf['overview']['win_rate']:.1%}")
            logger.info(f"Total P&L: ${perf['financial']['total_pnl']:.2f}")
            logger.info(f"ROI: {perf['financial']['roi']:.2%}")
    except:
        # Fallback
        metrics = sm.get_performance_metrics(session_id)
        if metrics:
            logger.info(f"Win Rate: {metrics.get('win_rate', 0):.1%}")
            logger.info(f"ROI: {metrics.get('roi', 0):.2%}")
            
else:
    logger.info("No open positions to settle")
    
logger.info("\n💡 Note: This is a demonstration using synthetic data.")
logger.info("For real trading, we need:")
logger.info("  1. Overtime V2 API authentication")
logger.info("  2. Direct blockchain integration")  
logger.info("  3. Real-time match result feeds")