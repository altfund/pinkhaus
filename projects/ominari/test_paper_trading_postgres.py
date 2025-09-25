#!/usr/bin/env python3
"""
Test the updated paper trading engine with PostgreSQL integration
"""

import asyncio
import logging
from datetime import datetime, timezone
from paper_trading_engine import PaperTradingEngine, PaperOrder
from database_v2 import db_manager
from models import Market

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_paper_trading_postgres():
    """Test paper trading with real PostgreSQL data."""
    logger.info("🧪 Testing Paper Trading Engine with PostgreSQL")
    logger.info("=" * 60)
    
    try:
        # Initialize paper trading engine
        engine = PaperTradingEngine(
            initial_capital=10000,
            commission_rate=0.002
        )
        
        logger.info(f"✅ Created paper trading engine")
        logger.info(f"   Session ID: {engine.session_id}")
        logger.info(f"   Initial Capital: ${engine.initial_capital:,.2f}")
        
        # Get a real market from our database to trade on
        with db_manager.get_db_session() as db:
            markets = db.query(Market).filter(
                Market.maturity_date > datetime.now(timezone.utc),
                Market.sport == 'Soccer'
            ).limit(3).all()
            
            if not markets:
                logger.error("No markets found in database!")
                return
            
            logger.info(f"✅ Found {len(markets)} markets to trade on")
            
            # Test placing orders on real markets
            for i, market in enumerate(markets):
                logger.info(f"\n📈 Testing trade {i+1}: {market.home_team} vs {market.away_team}")
                
                # Create a test order
                order = PaperOrder(
                    order_id=f"TEST_{i+1}_{int(datetime.now().timestamp())}",
                    timestamp=datetime.now(timezone.utc),
                    source_id=market.source_id,
                    market_type="winner",
                    bet_name="Home",
                    side="buy",
                    size=100,  # $100 bet
                    limit_price=2.50,  # Limit price of 2.50
                    signal_name="test_signal",
                    expected_edge=0.05
                )
                
                # Submit the order
                fill = await engine.submit_order(order)
                
                if fill:
                    logger.info(f"   ✅ Order filled!")
                    logger.info(f"   Fill Price: ${fill.fill_price:.3f}")
                    logger.info(f"   Fill Size: {fill.fill_size}")
                    logger.info(f"   Slippage: ${fill.slippage:.4f}")
                    logger.info(f"   Commission: ${fill.commission:.2f}")
                else:
                    logger.warning(f"   ❌ Order not filled")
        
        # Get performance report
        logger.info(f"\n📊 PERFORMANCE REPORT")
        logger.info("=" * 40)
        
        metrics = engine.calculate_performance()
        logger.info(f"Current Capital: ${engine.current_capital:,.2f}")
        logger.info(f"Portfolio Value: ${engine.get_portfolio_value():,.2f}")
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info(f"Total Commission: ${metrics['total_commission']:.2f}")
        logger.info(f"Total Slippage: ${metrics['total_slippage']:.4f}")
        
        # Show open positions
        positions = engine.get_open_positions()
        if not positions.empty:
            logger.info(f"\n📋 OPEN POSITIONS ({len(positions)})")
            for _, pos in positions.iterrows():
                logger.info(f"   • {pos['market_id']}: {pos['size']:.2f} @ ${pos['entry_price']:.3f}")
        
        # Show position exposure
        exposure = engine.get_position_exposure()
        logger.info(f"\n🎯 POSITION EXPOSURE")
        logger.info(f"   Total Exposure: {exposure['total_exposure']:.1%}")
        logger.info(f"   Position Count: {exposure['position_count']}")
        logger.info(f"   Largest Position: {exposure['largest_position']:.1%}")
        
        # Get session info
        session_info = engine.get_session_info()
        logger.info(f"\n📋 SESSION INFO")
        logger.info(f"   Session ID: {session_info['session_id']}")
        logger.info(f"   Status: {session_info['status']}")
        logger.info(f"   Open Positions: {session_info['open_positions']}")
        logger.info(f"   Total Trades: {session_info['total_trades']}")
        
        logger.info(f"\n✅ Paper trading engine test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error testing paper trading engine: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_paper_trading_postgres())