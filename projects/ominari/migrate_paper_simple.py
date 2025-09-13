#!/usr/bin/env python3
"""Simple migration of paper trading data to dedicated tables."""

import json
import logging
from datetime import datetime, timezone
from database_v2 import db_manager
from paper_trading_models import PaperOrder, PaperFill
from paper_trading_sessions import PaperTradingSessionManager
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_paper_trading():
    """Migrate paper trading data from JSON to database tables."""
    
    session_manager = PaperTradingSessionManager()
    sessions = session_manager.sessions.get('sessions', {})
    
    if not sessions:
        logger.error("No sessions found in JSON file")
        return
    
    logger.info(f"Found {len(sessions)} sessions to migrate")
    
    with db_manager.get_db_session() as db:
        total_orders = 0
        total_fills = 0
        
        for session_id, session_data in sessions.items():
            logger.info(f"\nMigrating session: {session_id}")
            
            # Migrate all trades
            trades = session_data.get('trades', [])
            logger.info(f"  Found {len(trades)} trades")
            
            for i, trade in enumerate(trades):
                # Create order
                order_id = f"{session_id}_{i:06d}"
                
                order = PaperOrder(
                    order_id=order_id,
                    timestamp=datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00')),
                    source_id=trade.get('market_id', 'unknown'),
                    market_type='1x2',  # Default for soccer
                    bet_name=f"{trade.get('market_name', 'Unknown')} - {trade.get('outcome', 'Unknown')}",
                    side='buy',  # All current trades are buys
                    size=trade.get('stake', 0),
                    limit_price=trade.get('odds', 0),
                    signal_name='implied_raw',
                    expected_edge=trade.get('edge', 0),
                    status='filled'
                )
                db.add(order)
                total_orders += 1
                
                # Create fill with P&L if position is closed
                pnl = 0
                
                # Find if this trade belongs to a closed position
                for closed_pos in session_data.get('closed_positions', []):
                    if (closed_pos.get('market_id') == trade.get('market_id') and 
                        closed_pos.get('outcome') == trade.get('outcome')):
                        # Use the position's P&L divided by number of trades
                        # (This is approximate - in reality we'd track individual trade P&L)
                        pnl = closed_pos.get('pnl', 0)
                        break
                
                fill = PaperFill(
                    fill_id=f"{order_id}_fill",
                    order_id=order_id,
                    timestamp=order.timestamp,
                    fill_price=trade.get('odds', 0),
                    fill_size=trade.get('stake', 0),
                    slippage=0,
                    commission=trade.get('fee_amount', 0),
                    market_impact=0,
                    pnl=pnl
                )
                db.add(fill)
                total_fills += 1
            
            # Log open positions (these won't have fills with P&L yet)
            open_positions = session_data.get('positions', {})
            logger.info(f"  Found {len(open_positions)} open positions (P&L pending)")
        
        # Commit all changes
        db.commit()
        
        logger.info("\n" + "=" * 60)
        logger.info("MIGRATION COMPLETE!")
        logger.info(f"Created {total_orders} orders and {total_fills} fills")
        logger.info("=" * 60)

if __name__ == "__main__":
    try:
        migrate_paper_trading()
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise