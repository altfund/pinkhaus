#!/usr/bin/env python3
"""Fix settlement for closed positions that should have been settled."""

from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market
from paper_trading_sessions import PaperTradingSessionManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_settlement():
    """Fix closed positions that weren't properly settled."""
    session_manager = PaperTradingSessionManager()
    current_session = session_manager.get_current_session()
    
    if not current_session:
        logger.info("No active session found")
        return
    
    session_id = current_session['session_id']
    closed_positions = current_session.get('closed_positions', [])
    
    logger.info(f"Checking {len(closed_positions)} closed positions for settlement issues")
    
    # Track which positions need fixing
    positions_to_fix = []
    market_ids_to_check = set()
    
    for i, pos in enumerate(closed_positions):
        if pos.get('result') == 'unknown' or pos.get('pnl', 0) == 0:
            market_id = pos.get('market_id')
            if market_id:
                market_ids_to_check.add(market_id)
                positions_to_fix.append((i, pos))
    
    logger.info(f"Found {len(positions_to_fix)} positions that need settlement fixing")
    
    # Get market results from database
    market_results = {}
    with db_manager.get_db_session() as db:
        for market_id in market_ids_to_check:
            market = db.query(Market).filter(Market.source_id == market_id).first()
            if market and market.is_finished and market.resolved_outcome:
                market_results[market_id] = {
                    'resolved_outcome': market.resolved_outcome,
                    'home_score': market.home_score,
                    'away_score': market.away_score,
                    'home_team': market.home_team,
                    'away_team': market.away_team
                }
    
    # Fix the settlements
    fixed_count = 0
    total_wins = 0
    total_losses = 0
    total_pnl = 0.0
    
    for idx, pos in positions_to_fix:
        market_id = pos.get('market_id')
        if market_id in market_results:
            market_info = market_results[market_id]
            winning_outcome = market_info['resolved_outcome']
            
            # Map market outcomes to position outcomes
            outcome_map = {
                'Home': 'Home',
                'Away': 'Away',
                'Draw': 'Draw'
            }
            
            pos_outcome = pos.get('outcome')
            stake = pos.get('total_stake', 0)
            odds = pos.get('avg_odds', 0)
            
            if pos_outcome == winning_outcome:
                # Win
                payout = stake * odds
                pnl = payout - stake
                pos['result'] = 'won'
                pos['pnl'] = pnl
                pos['final_value'] = payout
                total_wins += 1
                total_pnl += pnl
                logger.info(f"FIXED WIN: {market_info['home_team']} vs {market_info['away_team']} - {pos_outcome} - P&L: ${pnl:.2f}")
            else:
                # Loss
                pnl = -stake
                pos['result'] = 'lost'
                pos['pnl'] = pnl
                pos['final_value'] = 0
                total_losses += 1
                total_pnl += pnl
                logger.info(f"FIXED LOSS: {market_info['home_team']} vs {market_info['away_team']} - Bet: {pos_outcome}, Result: {winning_outcome} - P&L: ${pnl:.2f}")
            
            fixed_count += 1
    
    if fixed_count > 0:
        # Update performance metrics
        perf = current_session.get('performance', {})
        perf['winning_trades'] = perf.get('winning_trades', 0) + total_wins
        perf['losing_trades'] = perf.get('losing_trades', 0) + total_losses
        perf['total_pnl'] = perf.get('total_pnl', 0) + total_pnl
        
        # Save the updated session
        session_manager._save_sessions()
        
        logger.info(f"\nFixed {fixed_count} positions:")
        logger.info(f"  Wins: {total_wins}")
        logger.info(f"  Losses: {total_losses}")
        logger.info(f"  Total P&L adjustment: ${total_pnl:.2f}")
        
        # Recalculate performance
        performance = session_manager.get_session_performance(session_id)
        logger.info(f"\nUpdated Performance:")
        logger.info(f"  Portfolio Value: ${performance['current_value']:.2f}")
        logger.info(f"  Total P&L: ${performance['total_pnl']:.2f}")
        logger.info(f"  Win Rate: {performance['win_rate']:.1%}")
    else:
        logger.info("No positions could be fixed (markets not finished or missing data)")

if __name__ == "__main__":
    fix_settlement()