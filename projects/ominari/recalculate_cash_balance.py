#!/usr/bin/env python3
"""Recalculate cash balance based on actual positions"""

import os
os.environ['PG_PORT'] = '5999'

from paper_trading_postgres_integrated import PaperTradingSessionManager
from database_v2 import db_manager
from datetime import datetime, timezone
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recalculate_cash_balance():
    """Properly calculate cash balance from positions"""
    manager = PaperTradingSessionManager()
    session_id = manager.get_current_session()
    
    if not session_id:
        logger.error("No active session found")
        return
        
    logger.info(f"🔧 Recalculating cash balance for session {session_id}")
    
    # Get all positions
    positions = manager.get_positions(session_id)
    
    # Get initial bankroll
    with manager.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT initial_bankroll FROM paper_trading_sessions WHERE session_id = %s",
                (session_id,)
            )
            result = cur.fetchone()
            if not result:
                logger.error("Session not found")
                return
                
            initial_bankroll = float(result['initial_bankroll'] if isinstance(result, dict) else result[0])
    
    logger.info(f"Initial bankroll: ${initial_bankroll:,.2f}")
    
    # Calculate current cash
    cash_balance = initial_bankroll
    
    # Process each position
    position_summary = {
        'pending': {'count': 0, 'stake': 0},
        'open': {'count': 0, 'stake': 0},
        'settled': {'count': 0, 'payout': 0, 'stake': 0},
        'cancelled': {'count': 0}
    }
    
    for pos in positions:
        status = pos['status']
        stake = float(pos['stake'])
        
        if status in ['pending', 'open']:
            # Money is tied up in these positions
            cash_balance -= stake
            position_summary[status]['count'] += 1
            position_summary[status]['stake'] += stake
            
        elif status == 'settled':
            # Position resolved - we get payout back (if any)
            payout = float(pos.get('payout', 0))
            cash_balance += payout - stake  # Net effect
            position_summary[status]['count'] += 1
            position_summary[status]['payout'] += payout
            position_summary[status]['stake'] += stake
            
        elif status == 'cancelled':
            # Stake was returned
            position_summary[status]['count'] += 1
    
    # Log summary
    logger.info("\n📊 POSITION SUMMARY:")
    logger.info(f"Pending: {position_summary['pending']['count']} positions, ${position_summary['pending']['stake']:.2f} staked")
    logger.info(f"Open: {position_summary['open']['count']} positions, ${position_summary['open']['stake']:.2f} staked")
    logger.info(f"Settled: {position_summary['settled']['count']} positions, ${position_summary['settled']['payout']:.2f} payout")
    logger.info(f"Cancelled: {position_summary['cancelled']['count']} positions")
    
    total_exposure = position_summary['pending']['stake'] + position_summary['open']['stake']
    total_pnl = position_summary['settled']['payout'] - position_summary['settled']['stake']
    
    logger.info(f"\nTotal exposure: ${total_exposure:.2f}")
    logger.info(f"Total P&L: ${total_pnl:+.2f}")
    logger.info(f"\n💰 Calculated cash balance: ${cash_balance:.2f}")
    
    # Create a new snapshot with correct balance
    with manager.get_connection() as conn:
        with conn.cursor() as cur:
            # Insert new snapshot
            cur.execute("""
                INSERT INTO paper_trading_snapshots 
                (session_id, snapshot_time, cash_balance, positions_value, portfolio_value, 
                 total_pnl, win_count, loss_count, pending_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                session_id,
                datetime.now(timezone.utc),
                cash_balance,
                total_exposure,  # positions_value
                cash_balance + total_exposure,  # portfolio value = cash + exposure
                total_pnl,
                sum(1 for p in positions if p['status'] == 'settled' and float(p.get('payout', 0)) > float(p['stake'])),
                sum(1 for p in positions if p['status'] == 'settled' and float(p.get('payout', 0)) <= 0),
                position_summary['pending']['count'] + position_summary['open']['count']
            ))
            conn.commit()
            
    logger.info("\n✅ Cash balance recalculated and snapshot created!")
    
    # Verify the update
    updated_session = manager.get_session(session_id)
    logger.info(f"\nNew bankroll: ${updated_session['current_bankroll']:,.2f}")

def main():
    logger.info("💰 RECALCULATING CASH BALANCE")
    logger.info("=" * 60)
    
    recalculate_cash_balance()

if __name__ == "__main__":
    main()