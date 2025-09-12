#!/usr/bin/env python3
"""Check positions against finished matches and settle them."""

from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market
from paper_trading_sessions import PaperTradingSessionManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_settle_positions():
    """Check all positions against database for finished matches."""
    session_manager = PaperTradingSessionManager()
    current_session = session_manager.get_current_session()
    
    if not current_session:
        logger.info("No active session found")
        return
    
    session_id = current_session['session_id']
    positions = current_session.get('positions', {})
    
    logger.info(f"Checking {len(positions)} positions in session {session_id}")
    
    # Also check closed positions to see if we already settled some
    closed_positions = current_session.get('closed_positions', [])
    logger.info(f"Found {len(closed_positions)} already closed positions")
    
    # Get all market IDs from open positions
    market_ids_to_check = set()
    position_details = {}
    
    for pos_key, pos in positions.items():
        market_id = pos.get('market_id')
        if market_id:
            market_ids_to_check.add(market_id)
            position_details[market_id] = {
                'key': pos_key,
                'outcome': pos.get('outcome'),
                'stake': pos.get('total_stake', 0),
                'odds': pos.get('avg_odds', 0),
                'market_name': pos.get('market_name', 'Unknown')
            }
    
    # Query database for these specific markets
    settled_markets = {}
    matches_to_settle = []
    
    with db_manager.get_db_session() as db:
        for market_id in market_ids_to_check:
            market = db.query(Market).filter(Market.source_id == market_id).first()
            if market:
                logger.info(f"Checking {market.home_team} vs {market.away_team}:")
                logger.info(f"  - is_finished: {market.is_finished}")
                logger.info(f"  - resolved_outcome: {market.resolved_outcome}")
                logger.info(f"  - score: {market.home_score}-{market.away_score}")
                
                if market.is_finished and market.resolved_outcome:
                    settled_markets[market_id] = {
                        'is_finished': True,
                        'resolved_outcome': market.resolved_outcome,
                        'home_score': market.home_score,
                        'away_score': market.away_score
                    }
                    pos_detail = position_details.get(market_id)
                    if pos_detail:
                        matches_to_settle.append({
                            'market_name': f"{market.home_team} vs {market.away_team}",
                            'outcome_bet': pos_detail['outcome'],
                            'outcome_actual': market.resolved_outcome,
                            'stake': pos_detail['stake'],
                            'odds': pos_detail['odds'],
                            'won': pos_detail['outcome'] == market.resolved_outcome
                        })
    
    if matches_to_settle:
        logger.info(f"\nFound {len(matches_to_settle)} finished matches to settle:")
        for match in matches_to_settle:
            result = "WON" if match['won'] else "LOST"
            pnl = match['stake'] * (match['odds'] - 1) if match['won'] else -match['stake']
            logger.info(f"  {match['market_name']} - Bet: {match['outcome_bet']} - Result: {match['outcome_actual']} - {result} - P&L: ${pnl:.2f}")
        
        # Settle the positions
        settled_count = session_manager.settle_finished_markets(session_id, settled_markets)
        logger.info(f"\nSettled {settled_count} positions")
        
        # Get updated performance
        performance = session_manager.get_session_performance(session_id)
        logger.info(f"\nUpdated session performance:")
        logger.info(f"  Portfolio Value: ${performance['current_value']:.2f}")
        logger.info(f"  Total P&L: ${performance['total_pnl']:.2f}")
        logger.info(f"  Win Rate: {performance['win_rate']:.1%}")
        logger.info(f"  Winning Trades: {performance['winning_trades']}")
        logger.info(f"  Losing Trades: {performance['losing_trades']}")
        logger.info(f"  Pending Trades: {performance['pending_trades']}")
    else:
        logger.info("No finished matches found among current positions")

if __name__ == "__main__":
    check_and_settle_positions()