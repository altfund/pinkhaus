#!/usr/bin/env python3
"""Recalculate portfolio value after settlement fixes."""

from paper_trading_sessions import PaperTradingSessionManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recalculate_portfolio():
    """Recalculate portfolio value based on current state."""
    session_manager = PaperTradingSessionManager()
    current_session = session_manager.get_current_session()
    
    if not current_session:
        logger.info("No active session found")
        return
    
    session_id = current_session['session_id']
    
    # Calculate from scratch
    initial_bankroll = current_session['initial_bankroll']
    current_cash = initial_bankroll  # Start with initial
    
    # Subtract all trades made (both open and closed)
    total_staked = 0
    
    # Open positions
    for pos in current_session.get('positions', {}).values():
        total_staked += pos.get('total_stake', 0)
    
    # Closed positions - only count initial stakes, not P&L
    for pos in current_session.get('closed_positions', []):
        total_staked += pos.get('total_stake', 0)
    
    # Current cash = initial - total staked
    current_cash = initial_bankroll - total_staked
    
    # Add back winnings from closed positions
    total_realized_pnl = 0
    for pos in current_session.get('closed_positions', []):
        if pos.get('result') == 'won':
            # Add back the winnings (payout)
            payout = pos.get('final_value', 0)
            current_cash += payout
        # P&L is already accounted for in the cash calculation
        total_realized_pnl += pos.get('pnl', 0)
    
    # Current positions value (at current odds)
    positions_value = sum(pos.get('current_value', pos.get('total_stake', 0)) 
                         for pos in current_session.get('positions', {}).values())
    
    # Portfolio value = cash + positions value
    portfolio_value = current_cash + positions_value
    
    logger.info(f"Portfolio recalculation:")
    logger.info(f"  Initial bankroll: ${initial_bankroll:.2f}")
    logger.info(f"  Total staked: ${total_staked:.2f}")
    logger.info(f"  Current cash: ${current_cash:.2f}")
    logger.info(f"  Open positions value: ${positions_value:.2f}")
    logger.info(f"  Portfolio value: ${portfolio_value:.2f}")
    logger.info(f"  Total realized P&L: ${total_realized_pnl:.2f}")
    logger.info(f"  ROI: {((portfolio_value - initial_bankroll) / initial_bankroll * 100):.1f}%")
    
    # Update the session
    current_session['current_bankroll'] = current_cash
    current_session['portfolio_value'] = portfolio_value
    current_session['performance']['total_pnl'] = total_realized_pnl
    
    # Save
    session_manager._save_sessions()
    
    logger.info("\nPortfolio updated successfully")

if __name__ == "__main__":
    recalculate_portfolio()