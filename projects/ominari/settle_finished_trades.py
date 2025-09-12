#!/usr/bin/env python3
"""Settle finished matches in paper trading sessions."""

from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market
from paper_trading_sessions import PaperTradingSessionManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def settle_finished_positions():
    """Check for finished markets and settle positions."""
    session_manager = PaperTradingSessionManager()
    current_session = session_manager.get_current_session()
    
    if not current_session:
        logger.info("No active session found")
        return
    
    session_id = current_session['session_id']
    logger.info(f"Checking positions for settlement in session {session_id}")
    
    # Get all open positions
    positions = current_session.get('positions', {})
    if not positions:
        logger.info("No open positions to settle")
        return
    
    # Collect market IDs from positions
    market_ids = set()
    for pos_key, pos in positions.items():
        market_id = pos.get('market_id')
        if market_id:
            market_ids.add(market_id)
    
    logger.info(f"Checking {len(market_ids)} markets for settlement")
    
    # Query database for finished markets
    market_results = {}
    with db_manager.get_db_session() as db:
        for market_id in market_ids:
            market = db.query(Market).filter(Market.source_id == market_id).first()
            if market and market.is_finished and market.resolved_outcome:
                market_results[market_id] = {
                    'is_finished': True,
                    'resolved_outcome': market.resolved_outcome,
                    'home_score': market.home_score,
                    'away_score': market.away_score
                }
                logger.info(f"Found finished market: {market.home_team} vs {market.away_team} - Result: {market.resolved_outcome} ({market.home_score}-{market.away_score})")
    
    if market_results:
        # Settle the finished markets
        settled_count = session_manager.settle_finished_markets(session_id, market_results)
        logger.info(f"Settled {settled_count} positions")
        
        # Get updated performance
        performance = session_manager.get_session_performance(session_id)
        logger.info(f"Updated performance - Portfolio: ${performance['current_value']:.2f}, Total P&L: ${performance['total_pnl']:.2f}, Win Rate: {performance['win_rate']:.1%}")
        
        # Show closed positions
        closed_positions = current_session.get('closed_positions', [])
        if closed_positions:
            logger.info(f"\nRecent closed positions:")
            for pos in closed_positions[-5:]:  # Last 5
                result = pos.get('result', 'unknown')
                pnl = pos.get('pnl', 0)
                logger.info(f"  {pos.get('market_name', 'Unknown')} - {pos.get('outcome')} - Result: {result} - P&L: ${pnl:.2f}")
    else:
        logger.info("No finished markets found to settle")

if __name__ == "__main__":
    settle_finished_positions()