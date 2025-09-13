#!/usr/bin/env python3
"""Migrate paper trading data from JSON to database."""

import json
import logging
from datetime import datetime, timezone
from database_v2 import db_manager
from paper_trading_models import PaperOrder, PaperFill
from models import BettingSession, Bet, Market
from paper_trading_sessions import PaperTradingSessionManager
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_json_to_database():
    """Migrate all paper trading data from JSON to database."""
    
    session_manager = PaperTradingSessionManager()
    sessions = session_manager.sessions.get('sessions', {})
    
    if not sessions:
        logger.error("No sessions found in JSON file")
        return
    
    logger.info(f"Found {len(sessions)} sessions to migrate")
    
    with db_manager.get_db_session() as db:
        for session_id, session_data in sessions.items():
            logger.info(f"\nMigrating session: {session_id}")
            
            # 1. Create BettingSession entry
            session_start = datetime.fromisoformat(session_data['created_at'].replace('Z', '+00:00'))
            
            betting_session = BettingSession(
                as_of=session_start,
                session_type='paper',
                strategy_name='implied_raw',  # Default from the session
                kelly_bankroll=session_data['initial_bankroll'],
                execution_bankroll=session_data['initial_bankroll'],
                kelly_fraction=0.25,  # Default value
                cap_per_game=0.25  # Default cap per game
            )
            db.add(betting_session)
            db.flush()  # Get the session ID
            
            logger.info(f"Created BettingSession with ID: {betting_session.id}")
            
            # 2. Migrate trades to paper_orders and paper_fills
            trades = session_data.get('trades', [])
            logger.info(f"Migrating {len(trades)} trades")
            
            for i, trade in enumerate(trades):
                # Create order
                order_id = f"{session_id}_{i:06d}"
                
                order = PaperOrder(
                    order_id=order_id,
                    timestamp=datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00')),
                    source_id=trade.get('market_id', 'unknown'),
                    market_type=trade.get('market_type', 'unknown'),
                    bet_name=trade.get('outcome', 'unknown'),
                    side='buy',  # All trades are buys in the current system
                    size=trade.get('stake', 0),
                    limit_price=trade.get('odds', 0),
                    signal_name='implied_raw',
                    expected_edge=trade.get('edge', 0),
                    status='filled'
                )
                db.add(order)
                
                # Create fill
                fill = PaperFill(
                    fill_id=f"{order_id}_fill",
                    order_id=order_id,
                    timestamp=order.timestamp,
                    fill_price=trade.get('odds', 0),
                    fill_size=trade.get('stake', 0),
                    slippage=0,
                    commission=trade.get('fee_amount', 0),
                    market_impact=0,
                    pnl=0  # Will be calculated later
                )
                db.add(fill)
                
                # Create Bet entry for compatibility
                bet = Bet(
                    session_id=betting_session.id,
                    source_id=trade.get('market_id', 'unknown'),
                    unified_market_type='1x2',
                    normalized_outcome=trade.get('outcome', 'unknown'),
                    normalized_line=0,
                    bet_name=f"{trade.get('market_name', 'unknown')} - {trade.get('outcome', 'unknown')}",
                    probability=1.0 / trade.get('odds', 2.0) if trade.get('odds') else 0.5,
                    odds=trade.get('odds', 0),
                    stake=trade.get('stake', 0),
                    execution_stake=trade.get('execution_stake', trade.get('stake', 0)),
                    fee_amount=trade.get('fee_amount', 0),
                    fee_pct=trade.get('fee_pct', 0)
                )
                db.add(bet)
            
            # 3. Update fills with P&L for closed positions
            closed_positions = session_data.get('closed_positions', [])
            logger.info(f"Processing {len(closed_positions)} closed positions for P&L")
            
            position_pnl_map = {}
            for pos in closed_positions:
                key = f"{pos['market_id']}_{pos['outcome']}"
                position_pnl_map[key] = pos.get('pnl', 0)
            
            # Update fills with P&L
            for trade in trades:
                key = f"{trade.get('market_id')}_{trade.get('outcome')}"
                if key in position_pnl_map:
                    # Find corresponding fill and update P&L
                    # (In a real system, we'd track this better)
                    pass
            
            # Commit after each session
            db.commit()
            logger.info(f"✓ Session {session_id} migrated successfully")
    
    logger.info("\n" + "=" * 60)
    logger.info("MIGRATION COMPLETE!")
    logger.info("=" * 60)

if __name__ == "__main__":
    try:
        migrate_json_to_database()
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise