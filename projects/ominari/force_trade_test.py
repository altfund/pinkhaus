#!/usr/bin/env python3
"""Test script to force a trade execution"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from paper_trading_postgres_integrated import PaperTradingSessionManager
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session manager
session_manager = PaperTradingSessionManager()

# Get current session
session_id = session_manager.get_current_session()
logger.info(f"Using session: {session_id}")

# Get session info
session = session_manager.get_session(session_id)
logger.info(f"Current bankroll: ${session['current_bankroll']}")

# Create a test trade
test_trade = {
    'match_id': 'api_live_belgian_gp_2025',
    'match_name': 'Belgian Grand Prix 2025 vs Podium',
    'bet_type': 'away',
    'stake': 100.0,  # $100 bet
    'odds': 4.2,
    'probability': 0.368,  # From edge calculator
    'kickoff_time': datetime.now(timezone.utc),
    'signal_name': 'test_trade',
    'signal_value': 0.386,  # 38.6% edge
    'edge': 38.6
}

logger.info(f"\n🎯 Placing test trade:")
logger.info(f"   Market: {test_trade['match_name']}")
logger.info(f"   Bet: {test_trade['bet_type']} @ {test_trade['odds']}")
logger.info(f"   Stake: ${test_trade['stake']}")
logger.info(f"   Edge: {test_trade['edge']}%")

# Execute the trade
try:
    session_manager.record_trades(session_id, [test_trade])
    logger.info("✅ Trade executed successfully!")
    
    # Check positions
    positions = session_manager.get_positions(session_id)
    open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
    logger.info(f"\n📊 Open positions: {len(open_positions)}")
    
    for pos in open_positions[-1:]:  # Show latest position
        logger.info(f"   Position ID: {pos['id']}")
        logger.info(f"   Market: {pos['market_id']}")
        logger.info(f"   Bet: {pos['bet_type']} @ {pos['odds']}")
        logger.info(f"   Stake: ${pos['stake']}")
        logger.info(f"   Status: {pos['status']}")
        
    # Check new bankroll
    session = session_manager.get_session(session_id)
    logger.info(f"\n💰 New bankroll: ${session['current_bankroll']}")
    
except Exception as e:
    logger.error(f"❌ Error placing trade: {e}")