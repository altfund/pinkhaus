#!/usr/bin/env python3
"""Fix P&L calculations for paper trading positions that have results."""

from paper_trading_sessions import PaperTradingSessionManager
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_position_pnl(position):
    """Calculate P&L for a position based on market result."""
    market_result = position.get('market_result', {})
    
    # Check if market is finished
    if not market_result.get('is_finished'):
        return None, 'pending'
    
    # Get the resolved outcome
    resolved_outcome = market_result.get('resolved_outcome')
    position_outcome = position.get('outcome')
    
    # Determine if position won or lost
    if resolved_outcome and position_outcome:
        if resolved_outcome == position_outcome:
            # Position won
            execution_stake = position.get('execution_stake', position.get('total_stake', 0))
            avg_odds = position.get('avg_odds', 2.0)
            
            # Calculate winnings
            gross_return = execution_stake * avg_odds
            net_pnl = gross_return - execution_stake
            
            return net_pnl, 'won'
        else:
            # Position lost
            execution_stake = position.get('execution_stake', position.get('total_stake', 0))
            return -execution_stake, 'lost'
    
    return 0, 'unknown'

def fix_all_positions():
    """Fix P&L for all closed positions."""
    session_manager = PaperTradingSessionManager()
    current_session = session_manager.get_current_session()
    
    if not current_session:
        logger.error("No active session found")
        return
    
    closed_positions = current_session.get('closed_positions', [])
    logger.info(f"Processing {len(closed_positions)} closed positions")
    
    fixed_count = 0
    total_pnl = 0
    wins = 0
    losses = 0
    
    for i, pos in enumerate(closed_positions):
        # Calculate P&L if not already set
        if pos.get('pnl', 0) == 0 and pos.get('market_result'):
            pnl, result = calculate_position_pnl(pos)
            
            if pnl is not None:
                pos['pnl'] = pnl
                pos['result'] = result
                pos['final_pnl'] = pnl
                
                fixed_count += 1
                total_pnl += pnl
                
                if result == 'won':
                    wins += 1
                elif result == 'lost':
                    losses += 1
                
                # Log details
                if i < 5:  # Show first 5
                    logger.info(f"Fixed: {pos['market_name']} - {pos['outcome']}")
                    logger.info(f"  Result: {result} (match was {pos['market_result']['resolved_outcome']})")
                    logger.info(f"  P&L: ${pnl:.2f}")
    
    # Update session P&L totals
    if fixed_count > 0:
        current_session['total_pnl'] = sum(p.get('pnl', 0) for p in closed_positions)
        current_session['total_wins'] = wins
        current_session['total_losses'] = losses
        
        # Save the updated session
        session_manager._save_sessions()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SUMMARY: Fixed {fixed_count} positions")
        logger.info(f"Total P&L: ${total_pnl:.2f}")
        logger.info(f"Wins: {wins}, Losses: {losses}")
        if wins + losses > 0:
            win_rate = (wins / (wins + losses)) * 100
            logger.info(f"Win Rate: {win_rate:.1f}%")
    else:
        logger.info("No positions needed fixing")

if __name__ == "__main__":
    fix_all_positions()