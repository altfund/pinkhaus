#!/usr/bin/env python3
"""Backfill fee information for existing positions that were created before fee tracking."""

import json
import logging
from fee_calculator import FeeCalculator
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backfill_fees():
    """Add fee information to existing positions based on standard 3% fee structure."""
    
    fee_calc = FeeCalculator()
    
    # Load sessions
    sessions_file = "paper_trading_sessions.json"
    logger.info(f"Loading sessions from {sessions_file}")
    
    try:
        with open(sessions_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading sessions file: {e}")
        return
    
    sessions_updated = 0
    positions_updated = 0
    total_fees_added = 0.0
    
    # Process each session
    for session_id, session in data.get('sessions', {}).items():
        session_fees_added = 0.0
        session_positions_updated = 0
        
        logger.info(f"\nProcessing session {session_id}")
        
        # Process open positions
        positions = session.get('positions', {})
        for position_key, pos in positions.items():
            # Skip if already has fee info
            if 'fee_info' in pos and pos['fee_info']:
                continue
            
            # Get position details
            stake = pos.get('total_stake', 0)
            avg_odds = pos.get('avg_odds', 0)
            
            if stake > 0 and avg_odds > 0:
                # Calculate fees
                fee_info = fee_calc.calculate_entry_fees(stake, avg_odds)
                
                # Add fee info to position
                pos['fee_info'] = fee_info
                pos['execution_stake'] = fee_info['execution_stake']
                
                # Update trades with fee info
                for trade in pos.get('trades', []):
                    if trade.get('stake', 0) > 0 and 'fee_info' not in trade:
                        trade_fee_info = fee_calc.calculate_entry_fees(
                            trade['stake'], 
                            trade.get('odds', avg_odds)
                        )
                        trade['fee_info'] = trade_fee_info
                
                session_fees_added += fee_info['fee_amount']
                session_positions_updated += 1
                
                logger.info(f"  Position {position_key}: Added ${fee_info['fee_amount']:.2f} fees (stake: ${stake:.2f})")
        
        # Process closed positions
        closed_positions = session.get('closed_positions', [])
        for pos in closed_positions:
            # Skip if already has fee info
            if 'fee_info' in pos and pos['fee_info']:
                continue
            
            # Get position details
            stake = pos.get('total_stake', 0)
            avg_odds = pos.get('avg_odds', 0)
            
            if stake > 0 and avg_odds > 0:
                # Calculate fees
                fee_info = fee_calc.calculate_entry_fees(stake, avg_odds)
                
                # Add fee info to position
                pos['fee_info'] = fee_info
                pos['execution_stake'] = fee_info['execution_stake']
                
                # Update trades with fee info
                for trade in pos.get('trades', []):
                    if trade.get('stake', 0) > 0 and 'fee_info' not in trade:
                        trade_fee_info = fee_calc.calculate_entry_fees(
                            trade['stake'], 
                            trade.get('odds', avg_odds)
                        )
                        trade['fee_info'] = trade_fee_info
                
                # Recalculate P&L if position is settled
                if pos.get('result') in ['won', 'lost']:
                    execution_stake = pos['execution_stake']
                    if pos['result'] == 'won':
                        # Won - gross payout minus execution stake
                        gross_payout = stake * avg_odds
                        pos['pnl'] = gross_payout - execution_stake
                    else:
                        # Lost - lose entire execution stake
                        pos['pnl'] = -execution_stake
                
                session_fees_added += fee_info['fee_amount']
                session_positions_updated += 1
                
                logger.info(f"  Closed position {pos.get('market_name', 'Unknown')}: Added ${fee_info['fee_amount']:.2f} fees")
        
        # Update session performance metrics
        if session_positions_updated > 0:
            performance = session.get('performance', {})
            
            # Add to total fees
            current_total_fees = performance.get('total_fees', 0)
            performance['total_fees'] = current_total_fees + session_fees_added
            
            # Recalculate total P&L for closed positions
            total_pnl = 0.0
            for pos in closed_positions:
                total_pnl += pos.get('pnl', 0)
            performance['total_pnl'] = total_pnl
            
            logger.info(f"  Session summary: Updated {session_positions_updated} positions, added ${session_fees_added:.2f} in fees")
            
            sessions_updated += 1
            positions_updated += session_positions_updated
            total_fees_added += session_fees_added
    
    # Save updated sessions
    logger.info(f"\nSaving updated sessions...")
    try:
        # First create a backup
        import shutil
        backup_file = f"{sessions_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(sessions_file, backup_file)
        logger.info(f"Created backup: {backup_file}")
        
        # Save updated data
        with open(sessions_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info("Sessions file updated successfully")
        
    except Exception as e:
        logger.error(f"Error saving sessions file: {e}")
        return
    
    # Summary
    logger.info(f"\n=== Backfill Summary ===")
    logger.info(f"Sessions updated: {sessions_updated}")
    logger.info(f"Positions updated: {positions_updated}")
    logger.info(f"Total fees added: ${total_fees_added:.2f}")
    logger.info(f"Average fee per position: ${total_fees_added/positions_updated:.2f}" if positions_updated > 0 else "N/A")

if __name__ == "__main__":
    backfill_fees()