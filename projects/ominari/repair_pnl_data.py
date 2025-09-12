#!/usr/bin/env python3
"""Repair corrupted P&L data in paper trading sessions."""

from paper_trading_sessions import PaperTradingSessionManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def repair_position_data(position):
    """Repair a single position's data."""
    # Get base values
    stake = position.get('total_stake', 0)
    avg_odds = position.get('avg_odds', 0)
    result = position.get('result', 'pending')
    
    # Get fee info
    fee_info = position.get('fee_info', {})
    fee_amount = fee_info.get('fee_amount', 0)
    
    # Fix execution stake if needed
    expected_exec_stake = stake + fee_amount
    exec_stake = position.get('execution_stake', 0)
    
    if abs(exec_stake - expected_exec_stake) > 0.01:
        # Execution stake is wrong
        if fee_amount > 0:
            # We have fee info, use it
            position['execution_stake'] = expected_exec_stake
        else:
            # No fee info, assume 3% fee
            fee_amount = stake * 0.03
            position['execution_stake'] = stake + fee_amount
            position['fee_info'] = {
                'fee_amount': fee_amount,
                'total_fee_pct': 0.03,
                'safebox_fee': 0.02,
                'skew_fee': 0.01
            }
    else:
        # Execution stake looks correct
        position['execution_stake'] = exec_stake
    
    # Recalculate based on result
    if result == 'won':
        # Final value should be gross payout
        position['final_value'] = stake * avg_odds
        # P&L is gross payout minus execution stake
        position['pnl'] = position['final_value'] - position['execution_stake']
        
    elif result == 'lost':
        # Final value is 0
        position['final_value'] = 0
        # P&L is negative execution stake
        position['pnl'] = -position['execution_stake']
        
    else:  # pending or other
        # Leave as is or set to 0
        position['final_value'] = position.get('final_value', 0)
        position['pnl'] = position.get('pnl', 0)
    
    # Calculate ROI
    if position['execution_stake'] > 0 and result in ['won', 'lost']:
        position['roi'] = (position['pnl'] / position['execution_stake']) * 100
    
    return position

def repair_all_positions():
    """Repair all positions in the current session."""
    session_manager = PaperTradingSessionManager()
    current_session = session_manager.get_current_session()
    
    if not current_session:
        logger.error("No active session found")
        return
    
    logger.info("Starting P&L data repair...")
    
    # Track changes
    changes = {
        'final_value_fixed': 0,
        'exec_stake_fixed': 0,
        'pnl_fixed': 0,
        'total_positions': 0
    }
    
    # Fix closed positions
    closed_positions = current_session.get('closed_positions', [])
    logger.info(f"Processing {len(closed_positions)} closed positions")
    
    for pos in closed_positions:
        changes['total_positions'] += 1
        old_final = pos.get('final_value', 0)
        old_exec = pos.get('execution_stake', 0)
        old_pnl = pos.get('pnl', 0)
        
        repair_position_data(pos)
        
        # Track what changed
        if abs(old_final - pos.get('final_value', 0)) > 0.01:
            changes['final_value_fixed'] += 1
            
        if abs(old_exec - pos.get('execution_stake', 0)) > 0.01:
            changes['exec_stake_fixed'] += 1
            
        if abs(old_pnl - pos.get('pnl', 0)) > 0.01:
            changes['pnl_fixed'] += 1
            logger.info(f"{pos['market_name']} - {pos['outcome']}: "
                       f"P&L ${old_pnl:.2f} -> ${pos['pnl']:.2f}")
    
    # Recalculate totals
    total_pnl = sum(p.get('pnl', 0) for p in closed_positions)
    wins = sum(1 for p in closed_positions if p.get('result') == 'won')
    losses = sum(1 for p in closed_positions if p.get('result') == 'lost')
    
    # Update session totals
    current_session['total_pnl'] = total_pnl
    current_session['performance']['total_pnl'] = total_pnl
    current_session['performance']['winning_trades'] = wins
    current_session['performance']['losing_trades'] = losses
    
    # Recalculate portfolio value
    initial_bankroll = current_session['initial_bankroll']
    current_session['portfolio_value'] = initial_bankroll + total_pnl
    
    # Save the updated session
    session_manager._save_sessions()
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("REPAIR SUMMARY:")
    logger.info(f"Total positions processed: {changes['total_positions']}")
    logger.info(f"Final values fixed: {changes['final_value_fixed']}")
    logger.info(f"Execution stakes fixed: {changes['exec_stake_fixed']}")
    logger.info(f"P&L values fixed: {changes['pnl_fixed']}")
    logger.info(f"\nFINAL METRICS:")
    logger.info(f"Total P&L: ${total_pnl:.2f}")
    logger.info(f"Portfolio value: ${current_session['portfolio_value']:.2f}")
    logger.info(f"Wins: {wins}, Losses: {losses}")
    if wins + losses > 0:
        win_rate = (wins / (wins + losses)) * 100
        logger.info(f"Win Rate: {win_rate:.1f}%")

if __name__ == "__main__":
    repair_all_positions()