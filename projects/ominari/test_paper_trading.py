#!/usr/bin/env python3
"""Test paper trading functionality with existing markets"""

import logging
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market
from sqlalchemy import func, and_
from simple_paper_trading import execute_simple_paper_trades
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_paper_trading():
    """Test paper trading with existing market data."""
    
    logger.info("🧪 Testing Paper Trading System")
    logger.info("=" * 50)
    
    # Check for active markets
    with db_manager.get_db_session() as db:
        # Get count of non-finished markets
        active_count = db.query(func.count(Market.source_id)).filter(
            Market.is_finished == False
        ).scalar()
        
        logger.info(f"📊 Found {active_count:,} active markets in database")
        
        # Get some sample active markets
        sample_markets = db.query(Market).filter(
            Market.is_finished == False
        ).limit(5).all()
        
        if sample_markets:
            logger.info("\nSample active markets:")
            for market in sample_markets:
                logger.info(f"  • {market.home_team} vs {market.away_team}")
                logger.info(f"    Sport: {market.sport}, Maturity: {market.maturity_date}")
        
        # Check if we have recent odds data
        from models import Odd
        recent_odds_count = 0
        for market in sample_markets[:3]:
            odds_count = db.query(func.count(Odd.source_id)).filter(
                Odd.source_id == market.source_id
            ).scalar()
            recent_odds_count += odds_count
            if odds_count > 0:
                logger.info(f"    {market.home_team} vs {market.away_team}: {odds_count} odds")
    
    logger.info("\n🎲 Testing Paper Trading Execution")
    logger.info("-" * 50)
    
    try:
        # Run paper trading without quotes (using stored odds)
        execute_simple_paper_trades(use_sessions=True, use_execution_quotes=False)
        logger.info("✅ Paper trading execution completed")
        
        # Check if any trades were recorded
        try:
            with open('paper_trading_sessions.json', 'r') as f:
                sessions_data = json.load(f)
                
            sessions = sessions_data.get('sessions', {})
            if sessions:
                latest_session = max(sessions.keys())
                session_data = sessions[latest_session]
                
                logger.info(f"\n📈 Session {latest_session} Summary:")
                logger.info(f"  • Total capital: ${session_data.get('total_capital', 0):,.2f}")
                logger.info(f"  • Available capital: ${session_data.get('available_capital', 0):,.2f}")
                logger.info(f"  • Positions: {len(session_data.get('positions', {}))}")
                logger.info(f"  • Total P&L: ${session_data.get('total_pnl', 0):,.2f}")
                
                # Show recent positions
                positions = session_data.get('positions', {})
                if positions:
                    logger.info("\n  Recent positions:")
                    for i, (pos_id, pos) in enumerate(list(positions.items())[-3:]):
                        logger.info(f"    - {pos.get('bet_name', 'Unknown')}")
                        logger.info(f"      Size: ${pos.get('size', 0):,.2f}, Status: {pos.get('status', 'unknown')}")
                
        except Exception as e:
            logger.warning(f"Could not read session data: {e}")
            
    except Exception as e:
        logger.error(f"❌ Paper trading test failed: {e}", exc_info=True)
    
    logger.info("\n✅ Paper Trading Test Complete!")

if __name__ == "__main__":
    test_paper_trading()