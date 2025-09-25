#!/usr/bin/env python3
"""
Run Live Paper Trading with PostgreSQL Integration
"""

import os
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

# Set PostgreSQL environment first
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from paper_trading_engine import PaperTradingEngine, PaperOrder
from database_v2 import db_manager
from models import Market, Odd
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleLiveTradingStrategy:
    """Simple live trading strategy using real-time odds."""
    
    def __init__(self, engine: PaperTradingEngine):
        self.engine = engine
        self.min_edge = 0.05  # 5% minimum edge
        self.max_position_pct = 0.1  # 10% max per position
        
    async def evaluate_market(self, market: Market) -> Optional[PaperOrder]:
        """Evaluate a market for trading opportunity."""
        try:
            # Get current odds
            with db_manager.get_db_session() as db:
                odds = db.query(Odd).filter(
                    Odd.source_id == market.source_id
                ).order_by(Odd.updated_at.desc()).limit(3).all()
                
                if not odds:
                    return None
                
                # Simple strategy: bet on home team if odds > 2.5 (underdog value)
                home_odds = next((o for o in odds if o.outcome == 'Home'), None)
                away_odds = next((o for o in odds if o.outcome == 'Away'), None)
                
                if not home_odds or not away_odds:
                    return None
                
                # Calculate implied probabilities
                home_prob = 1 / home_odds.decimal_odds
                away_prob = 1 / away_odds.decimal_odds
                total_prob = home_prob + away_prob
                
                # Look for value in high-vig markets
                if total_prob > 1.1:  # 10%+ overround
                    # Bet on underdog
                    if home_odds.decimal_odds > away_odds.decimal_odds:
                        # Home is underdog
                        stake = min(100, self.engine.current_capital * self.max_position_pct)
                        
                        return PaperOrder(
                            order_id=f"LIVE_{market.source_id}_{int(datetime.now().timestamp())}",
                            timestamp=datetime.now(timezone.utc),
                            source_id=market.source_id,
                            market_type="winner",
                            bet_name="Home",
                            side="buy",
                            size=stake,
                            limit_price=home_odds.decimal_odds,
                            signal_name="underdog_value",
                            expected_edge=(total_prob - 1) / 2  # Half the overround as edge
                        )
                    else:
                        # Away is underdog
                        stake = min(100, self.engine.current_capital * self.max_position_pct)
                        
                        return PaperOrder(
                            order_id=f"LIVE_{market.source_id}_{int(datetime.now().timestamp())}",
                            timestamp=datetime.now(timezone.utc),
                            source_id=market.source_id,
                            market_type="winner",
                            bet_name="Away",
                            side="buy",
                            size=stake,
                            limit_price=away_odds.decimal_odds,
                            signal_name="underdog_value",
                            expected_edge=(total_prob - 1) / 2
                        )
                        
        except Exception as e:
            logger.error(f"Error evaluating market {market.source_id}: {e}")
            
        return None


async def run_live_paper_trading(duration_minutes: int = 60):
    """Run live paper trading for specified duration."""
    logger.info("🚀 Starting Live Paper Trading")
    logger.info("=" * 60)
    
    # Initialize paper trading engine
    engine = PaperTradingEngine(
        initial_capital=5000,
        commission_rate=0.002
    )
    
    strategy = SimpleLiveTradingStrategy(engine)
    
    logger.info(f"📊 Trading Session Started")
    logger.info(f"   Session ID: {engine.session_id}")
    logger.info(f"   Initial Capital: ${engine.initial_capital:,.2f}")
    logger.info(f"   Duration: {duration_minutes} minutes")
    
    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    cycle_count = 0
    total_orders = 0
    successful_fills = 0
    
    # Run for specified duration
    for _ in range(duration_minutes // 5):  # Check every 5 minutes
        cycle_count += 1
        logger.info(f"\n🔄 Trading Cycle {cycle_count}")
        
        try:
            # Get markets starting in next 6 hours
            with db_manager.get_db_session() as db:
                markets = db.query(Market).filter(
                    Market.maturity_date > datetime.now(timezone.utc),
                    Market.maturity_date < datetime.now(timezone.utc).replace(hour=23, minute=59),
                    Market.is_finished == False
                ).limit(20).all()
                
                logger.info(f"   Evaluating {len(markets)} markets...")
                
                # Evaluate each market
                orders_this_cycle = 0
                for market in markets:
                    # Skip if we already have a position
                    if market.source_id in [p.market_id for p in engine.position_manager.positions.values()]:
                        continue
                        
                    order = await strategy.evaluate_market(market)
                    
                    if order:
                        logger.info(f"   📈 Trading opportunity: {market.home_team} vs {market.away_team}")
                        logger.info(f"      Sport: {market.sport}")
                        logger.info(f"      Position: {order.bet_name}")
                        logger.info(f"      Odds: {order.limit_price:.2f}")
                        logger.info(f"      Stake: ${order.size:.2f}")
                        
                        # Execute order
                        fill = await engine.submit_order(order)
                        total_orders += 1
                        
                        if fill:
                            successful_fills += 1
                            orders_this_cycle += 1
                            logger.info(f"      ✅ Order filled at {fill.fill_price:.2f}")
                        else:
                            logger.info(f"      ❌ Order not filled")
                
                if orders_this_cycle == 0:
                    logger.info("   No trading opportunities found")
                    
            # Report current status
            metrics = engine.calculate_performance()
            portfolio_value = engine.get_portfolio_value()
            
            logger.info(f"\n💰 Portfolio Status:")
            logger.info(f"   Capital: ${engine.current_capital:.2f}")
            logger.info(f"   Portfolio Value: ${portfolio_value:.2f}")
            logger.info(f"   Open Positions: {len(engine.position_manager.positions)}")
            logger.info(f"   Total Trades: {successful_fills}")
            
            if metrics['total_trades'] > 0:
                logger.info(f"   Total Commission: ${metrics['total_commission']:.2f}")
                logger.info(f"   Win Rate: {metrics['win_rate']:.1%}")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            
        # Wait 5 minutes before next cycle
        await asyncio.sleep(300)
        
        # Check if we should stop
        if datetime.now(timezone.utc) >= end_time:
            break
    
    # Final report
    logger.info("\n" + "="*60)
    logger.info("📊 LIVE PAPER TRADING SESSION COMPLETE")
    logger.info("="*60)
    
    final_metrics = engine.calculate_performance()
    final_value = engine.get_portfolio_value()
    
    logger.info(f"Initial Capital: ${engine.initial_capital:,.2f}")
    logger.info(f"Final Portfolio Value: ${final_value:,.2f}")
    logger.info(f"Total Return: ${final_value - engine.initial_capital:,.2f}")
    logger.info(f"Return %: {((final_value / engine.initial_capital) - 1) * 100:.2f}%")
    logger.info(f"Total Orders: {total_orders}")
    logger.info(f"Successful Fills: {successful_fills}")
    logger.info(f"Fill Rate: {(successful_fills / total_orders * 100) if total_orders > 0 else 0:.1f}%")
    
    # Show open positions
    positions = engine.get_open_positions()
    if not positions.empty:
        logger.info(f"\n📋 Open Positions:")
        for _, pos in positions.iterrows():
            logger.info(f"   • {pos['market_id']}: {pos['size']:.2f} @ ${pos['entry_price']:.2f}")
    
    # Stop session
    session_summary = engine.stop_session()
    logger.info(f"\n✅ Session {session_summary['session_id']} stopped successfully")
    
    return session_summary


if __name__ == "__main__":
    # Run for 2 minutes as a demo
    asyncio.run(run_live_paper_trading(duration_minutes=2))