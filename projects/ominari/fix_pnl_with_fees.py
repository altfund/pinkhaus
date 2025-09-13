#!/usr/bin/env python3
"""Fix P&L calculations to properly account for fees."""

from paper_trading_sessions import PaperTradingSessionManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recalculate_position_pnl(position):
    """Recalculate P&L with proper fee accounting."""
    market_result = position.get('market_result', {})
    
    if not market_result.get('is_finished'):
        return None, 'pending'
    
    resolved_outcome = market_result.get('resolved_outcome')
    position_outcome = position.get('outcome')
    
    # Get stake and execution stake (stake + fees)
    stake = position.get('total_stake', 0)
    execution_stake = position.get('execution_stake', stake)
    avg_odds = position.get('avg_odds', 2.0)
    
    if resolved_outcome and position_outcome:
        if resolved_outcome == position_outcome:
            # Won - payout is based on STAKE (not execution stake)
            # But we paid execution_stake, so P&L = (stake × odds) - execution_stake
            gross_payout = stake * avg_odds
            net_pnl = gross_payout - execution_stake
            return net_pnl, 'won'
        else:
            # Lost - we lose the entire execution stake (including fees)
            return -execution_stake, 'lost'
    
    return 0, 'unknown'

def fix_all_pnl():
    """Fix P&L for all positions and recalculate portfolio."""
    session_manager = PaperTradingSessionManager()
    current_session = session_manager.get_current_session()
    
    if not current_session:
        logger.error("No active session found")
        return
    
    closed_positions = current_session.get('closed_positions', [])
    logger.info(f"Processing {len(closed_positions)} closed positions")
    
    total_pnl = 0
    wins = 0
    losses = 0
    changes = 0
    
    for pos in closed_positions:
        old_pnl = pos.get('pnl', 0)
        new_pnl, result = recalculate_position_pnl(pos)
        
        if new_pnl is not None:
            if abs(old_pnl - new_pnl) > 0.01:  # Changed
                changes += 1
                logger.info(f"{pos['market_name']} - {pos['outcome']}: ${old_pnl:.2f} -> ${new_pnl:.2f}")
            
            pos['pnl'] = new_pnl
            pos['final_pnl'] = new_pnl
            pos['result'] = result
            
            total_pnl += new_pnl
            
            if result == 'won':
                wins += 1
            elif result == 'lost':
                losses += 1
    
    # Recalculate portfolio value
    initial_bankroll = current_session['initial_bankroll']
    
    # Portfolio value = initial + all closed P&L
    portfolio_value = initial_bankroll + total_pnl
    
    # Current bankroll = what's left after all positions (open and closed)
    # This is already tracked correctly in the session
    
    current_session['total_pnl'] = total_pnl
    current_session['portfolio_value'] = portfolio_value
    
    # Save the updated session
    session_manager._save_sessions()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY: Fixed {changes} P&L calculations")
    logger.info(f"Total P&L: ${total_pnl:.2f}")
    logger.info(f"Portfolio value: ${portfolio_value:.2f}")
    logger.info(f"Wins: {wins}, Losses: {losses}")
    if wins + losses > 0:
        win_rate = (wins / (wins + losses)) * 100
        logger.info(f"Win Rate: {win_rate:.1f}%")

if __name__ == "__main__":
    fix_all_pnl()