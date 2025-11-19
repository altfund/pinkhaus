#!/usr/bin/env python3
"""
Fix Paper Trading Portfolio Issues
Resets broken session and implements proper position size controls
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone

# Load environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_env import load_dotenv
load_dotenv()
os.environ['PG_PORT'] = '5999'

from paper_trading_sessions import PaperTradingSessionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_paper_trading_portfolio():
    """Fix the broken paper trading portfolio with massive over-leveraging."""
    logger.info("🔧 Starting Paper Trading Portfolio Fix")
    
    manager = PaperTradingSessionManager()
    
    # Load current sessions
    sessions_data = manager._load_sessions()
    
    if not sessions_data or 'sessions' not in sessions_data:
        logger.error("No sessions data found")
        return
    
    # Find the broken session
    broken_session_id = "20250912_181426"
    broken_session = sessions_data['sessions'].get(broken_session_id)
    
    if not broken_session:
        logger.error(f"Broken session {broken_session_id} not found")
        return
    
    logger.info(f"Found broken session: {broken_session_id}")
    logger.info(f"Current bankroll: ${broken_session['current_bankroll']:,.2f}")
    logger.info(f"Open positions: {len(broken_session['positions'])}")
    
    # Calculate total exposure
    total_stake = sum(pos['total_stake'] for pos in broken_session['positions'].values())
    logger.info(f"Total position stakes: ${total_stake:,.2f}")
    
    # Reset the broken session
    logger.info("💥 Resetting broken session...")
    
    # End the broken session
    broken_session['status'] = 'ended'
    broken_session['ended_at'] = datetime.now(timezone.utc).isoformat()
    broken_session['end_reason'] = 'RESET_DUE_TO_OVER_LEVERAGING'
    
    # Move all positions to closed with reset reason
    for pos_key, position in broken_session['positions'].items():
        position['status'] = 'closed'
        position['closed_at'] = datetime.now(timezone.utc).isoformat()
        position['close_reason'] = 'SYSTEM_RESET_OVER_LEVERAGE'
        position['pnl'] = -position['total_stake']  # Mark as total loss
        broken_session['closed_positions'].append(position)
    
    # Clear positions
    broken_session['positions'] = {}
    
    # Update performance
    broken_session['performance']['total_pnl'] = -total_stake
    broken_session['performance']['losing_trades'] = len(broken_session['closed_positions'])
    broken_session['performance']['pending_trades'] = 0
    
    # Reset bankroll to properly account for the reset
    broken_session['current_bankroll'] = broken_session['initial_bankroll'] - total_stake
    broken_session['portfolio_value'] = broken_session['current_bankroll']
    
    # Create a new clean session for future trading
    logger.info("✨ Creating new clean session...")
    
    new_session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_FIXED"
    
    new_session = {
        "session_id": new_session_id,
        "session_name": f"Fixed Session {new_session_id}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "initial_bankroll": 10000.0,
        "current_bankroll": 10000.0,
        "portfolio_value": 10000.0,
        "positions": {},
        "closed_positions": [],
        "trades": [],
        "performance": {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "pending_trades": 0,
            "total_pnl": 0.0,
            "total_stake": 0.0,
            "max_drawdown": 0.0,
            "peak_value": 10000.0
        },
        "status": "active",
        "position_size_controls": {
            # Based on Thorp's "Beat the Dealer" and modern portfolio theory
            "max_position_pct": 0.02,  # 2% max per position (1/4 Kelly for safety)
            "max_total_exposure_pct": 0.15,  # 15% max total exposure (diversification)
            "min_bet_abs": 10.0,  # $10 minimum bet (transaction cost efficiency)
            "max_bet_abs": 200.0,  # $200 maximum bet (2% of $10K)
            "kelly_fraction": 0.25,  # Conservative Kelly fraction (Thorp recommendation)
            "max_concurrent_positions": 10,  # Limit concurrent bets for diversification
            "protocol": "Modified Kelly with Thorp Safety Margins",
            "risk_tolerance": "conservative"  # Conservative approach for paper trading
        }
    }
    
    # Add the new session
    sessions_data['sessions'][new_session_id] = new_session
    
    # Update manager
    manager.sessions = sessions_data
    manager.current_session_id = new_session_id
    manager._save_sessions()
    
    logger.info("✅ Portfolio Fix Complete!")
    logger.info(f"Ended broken session: {broken_session_id}")
    logger.info(f"Total loss recorded: ${total_stake:,.2f}")
    logger.info(f"Created new session: {new_session_id}")
    logger.info(f"New session bankroll: $10,000.00")
    
    # Add position size validation method to manager
    enhance_session_manager_with_controls(manager)
    
    return new_session_id

def enhance_session_manager_with_controls(manager):
    """Add position size validation to the session manager."""
    
    # Store original record_trades method
    original_record_trades = manager.record_trades
    
    def validate_and_record_trades(session_id: str, trades: list) -> bool:
        """Enhanced record_trades with position size validation."""
        session = manager.sessions["sessions"].get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False
        
        # Get position controls - default to Thorp protocol
        controls = session.get("position_size_controls", {
            "max_position_pct": 0.02,  # 2% Thorp recommendation
            "max_total_exposure_pct": 0.15,  # 15% total portfolio exposure
            "min_bet_abs": 10.0,
            "max_bet_abs": 200.0,  # 2% of $10K
            "kelly_fraction": 0.25,
            "max_concurrent_positions": 10
        })
        
        current_bankroll = session["current_bankroll"]
        max_bet = min(controls["max_bet_abs"], current_bankroll * controls["max_position_pct"])
        
        # Calculate current exposure
        current_exposure = sum(pos["total_stake"] for pos in session["positions"].values())
        max_total_exposure = current_bankroll * controls["max_total_exposure_pct"]
        
        # Validate each trade
        validated_trades = []
        for trade in trades:
            stake = abs(trade.get("stake", 0))
            
            # Skip zero stakes
            if stake < 0.01:
                continue
            
            # Check minimum bet
            if stake < controls["min_bet_abs"]:
                logger.warning(f"Stake ${stake:.2f} below minimum ${controls['min_bet_abs']:.2f}, skipping")
                continue
            
            # Check maximum bet
            if stake > max_bet:
                logger.warning(f"Stake ${stake:.2f} exceeds max ${max_bet:.2f}, capping")
                trade["stake"] = max_bet if trade.get("stake", 0) >= 0 else -max_bet
                stake = max_bet
            
            # Check total exposure
            if current_exposure + stake > max_total_exposure:
                available_exposure = max_total_exposure - current_exposure
                if available_exposure >= controls["min_bet_abs"]:
                    logger.warning(f"Reducing stake to fit exposure limit: ${available_exposure:.2f}")
                    trade["stake"] = available_exposure if trade.get("stake", 0) >= 0 else -available_exposure
                    current_exposure += available_exposure
                else:
                    logger.warning(f"Skipping trade - would exceed exposure limit")
                    continue
            else:
                current_exposure += stake
            
            validated_trades.append(trade)
        
        # Record validated trades
        if validated_trades:
            logger.info(f"Recording {len(validated_trades)} validated trades (rejected {len(trades) - len(validated_trades)})")
            return original_record_trades(session_id, validated_trades)
        else:
            logger.warning(f"All {len(trades)} trades rejected by position size controls")
            return True
    
    # Replace the method
    manager.record_trades = validate_and_record_trades
    logger.info("✅ Enhanced session manager with position size controls")

if __name__ == "__main__":
    fix_paper_trading_portfolio()