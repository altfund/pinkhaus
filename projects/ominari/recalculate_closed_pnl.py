#!/usr/bin/env python3
"""Recalculate P&L for closed positions to properly account for fees."""

import json
import logging
from database_v2 import db_manager
from models import Market

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recalculate_closed_positions_pnl():
    """Recalculate P&L for all closed positions based on actual results and fees."""
    
    # Load sessions
    sessions_file = "paper_trading_sessions.json"
    logger.info(f"Loading sessions from {sessions_file}")
    
    try:
        with open(sessions_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading sessions file: {e}")
        return
    
    positions_updated = 0
    total_pnl_before = 0.0
    total_pnl_after = 0.0
    
    # Process each session
    for session_id, session in data.get('sessions', {}).items():
        logger.info(f"\nProcessing session {session_id}")
        
        # Get closed positions
        closed_positions = session.get('closed_positions', [])
        
        # Collect market IDs for batch query
        market_ids = [pos.get('market_id') for pos in closed_positions if pos.get('market_id')]
        
        # Query database for market results
        market_results = {}
        if market_ids:
            try:
                with db_manager.get_db_session() as db:
                    markets = db.query(Market).filter(Market.source_id.in_(market_ids)).all()
                    for market in markets:
                        market_results[market.source_id] = {
                            'is_finished': market.is_finished,
                            'resolved_outcome': market.resolved_outcome,
                            'home_score': market.home_score,
                            'away_score': market.away_score
                        }
            except Exception as e:
                logger.error(f"Error querying markets: {e}")
        
        # Process each closed position
        for pos in closed_positions:
            market_id = pos.get('market_id')
            stake = pos.get('total_stake', 0)
            avg_odds = pos.get('avg_odds', 0)
            outcome = pos.get('outcome')
            execution_stake = pos.get('execution_stake', stake)
            old_pnl = pos.get('pnl', 0)
            
            total_pnl_before += old_pnl
            
            # Get market result
            market_info = market_results.get(market_id, {})
            
            if market_info.get('is_finished') and market_info.get('resolved_outcome'):
                # Market finished - calculate actual P&L
                resolved = market_info['resolved_outcome']
                
                # Map resolved outcome to our outcome format
                resolved_mapped = resolved
                if resolved == 'option_1':
                    resolved_mapped = 'Home'
                elif resolved == 'option_2':
                    resolved_mapped = 'Away'
                elif resolved == 'option_3':
                    resolved_mapped = 'Draw'
                
                if outcome == resolved_mapped:
                    # Won - gross payout minus execution stake
                    gross_payout = stake * avg_odds
                    new_pnl = gross_payout - execution_stake
                    pos['result'] = 'won'
                else:
                    # Lost - lose entire execution stake
                    new_pnl = -execution_stake
                    pos['result'] = 'lost'
                
                pos['pnl'] = new_pnl
                total_pnl_after += new_pnl
                
                if abs(new_pnl - old_pnl) > 0.01:
                    logger.info(f"  Position {pos.get('market_name', 'Unknown')} - {outcome}:")
                    logger.info(f"    Result: {pos['result']}")
                    logger.info(f"    Stake: ${stake:.2f}, Execution stake: ${execution_stake:.2f}")
                    logger.info(f"    Old P&L: ${old_pnl:.2f}, New P&L: ${new_pnl:.2f}, Difference: ${new_pnl - old_pnl:.2f}")
                    positions_updated += 1
            else:
                # Market not finished or no result - keep existing P&L
                total_pnl_after += old_pnl
        
        # Update session performance metrics
        performance = session.get('performance', {})
        
        # Recalculate total P&L
        session_total_pnl = sum(pos.get('pnl', 0) for pos in closed_positions)
        performance['total_pnl'] = session_total_pnl
        
        # Count wins and losses
        wins = sum(1 for pos in closed_positions if pos.get('result') == 'won')
        losses = sum(1 for pos in closed_positions if pos.get('result') == 'lost')
        performance['winning_trades'] = wins
        performance['losing_trades'] = losses
        
        logger.info(f"  Session summary: {wins} wins, {losses} losses, Total P&L: ${session_total_pnl:.2f}")
    
    # Save updated sessions
    logger.info(f"\nSaving updated sessions...")
    try:
        with open(sessions_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info("Sessions file updated successfully")
        
    except Exception as e:
        logger.error(f"Error saving sessions file: {e}")
        return
    
    # Summary
    logger.info(f"\n=== P&L Recalculation Summary ===")
    logger.info(f"Positions updated: {positions_updated}")
    logger.info(f"Total P&L before: ${total_pnl_before:.2f}")
    logger.info(f"Total P&L after: ${total_pnl_after:.2f}")
    logger.info(f"Difference (fees impact): ${total_pnl_after - total_pnl_before:.2f}")

if __name__ == "__main__":
    recalculate_closed_positions_pnl()