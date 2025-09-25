#!/usr/bin/env python3
"""
Simple Live Paper Trading without complex dependencies
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import psycopg2

# Set PostgreSQL environment first
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market, Odd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimpleOrder:
    """Simple order representation."""
    order_id: str
    market_id: str
    home_team: str
    away_team: str
    position: str  # 'Home', 'Away', 'Draw'
    stake: float
    odds: float
    timestamp: datetime
    signal: str
    expected_edge: float


@dataclass
class SimpleFill:
    """Simple fill representation."""
    fill_id: str
    order: SimpleOrder
    fill_price: float
    fill_time: datetime
    commission: float
    slippage: float


class SimplePaperTradingEngine:
    """Simplified paper trading engine."""
    
    def __init__(self, initial_capital: float = 5000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = 0.002
        self.active_positions: Dict[str, SimpleOrder] = {}
        self.completed_trades: List[SimpleFill] = []
        self.session_id = f"SIMPLE_{int(datetime.now().timestamp())}"
        
        # Initialize database tables
        self._init_db()
        
    def _init_db(self):
        """Initialize paper trading tables in PostgreSQL."""
        conn = psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
        cur = conn.cursor()
        
        # Create simple paper trading table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS simple_paper_trades (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                order_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                position TEXT NOT NULL,
                stake DECIMAL(10,2),
                odds DECIMAL(6,3),
                fill_price DECIMAL(6,3),
                commission DECIMAL(8,4),
                result TEXT,
                pnl DECIMAL(10,2),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        conn.commit()
        cur.close()
        conn.close()
    
    def execute_order(self, order: SimpleOrder) -> Optional[SimpleFill]:
        """Execute a paper order."""
        # Check capital
        if order.stake > self.capital:
            logger.warning(f"Insufficient capital for order {order.order_id}")
            return None
            
        # Simulate execution with small slippage
        slippage = 0.01 * order.odds  # 1% slippage
        fill_price = order.odds + slippage
        commission = order.stake * self.commission_rate
        
        # Create fill
        fill = SimpleFill(
            fill_id=f"FILL_{int(datetime.now().timestamp())}",
            order=order,
            fill_price=fill_price,
            fill_time=datetime.now(timezone.utc),
            commission=commission,
            slippage=slippage
        )
        
        # Update capital and positions
        self.capital -= (order.stake + commission)
        self.active_positions[order.market_id] = order
        self.completed_trades.append(fill)
        
        # Store in database
        self._store_trade(order, fill)
        
        return fill
    
    def _store_trade(self, order: SimpleOrder, fill: SimpleFill):
        """Store trade in database."""
        conn = psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO simple_paper_trades 
            (session_id, order_id, market_id, position, stake, odds, fill_price, commission)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            self.session_id, order.order_id, order.market_id, order.position,
            order.stake, order.odds, fill.fill_price, fill.commission
        ))
        
        conn.commit()
        cur.close()
        conn.close()
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status."""
        total_value = self.capital
        
        # Add value of active positions
        for position in self.active_positions.values():
            total_value += position.stake
            
        return {
            'session_id': self.session_id,
            'initial_capital': self.initial_capital,
            'current_capital': self.capital,
            'portfolio_value': total_value,
            'active_positions': len(self.active_positions),
            'total_trades': len(self.completed_trades),
            'total_commission': sum(f.commission for f in self.completed_trades),
            'return_pct': ((total_value / self.initial_capital) - 1) * 100
        }


def evaluate_market(market: Market) -> Optional[SimpleOrder]:
    """Evaluate a market for trading opportunity."""
    try:
        # Get current odds
        with db_manager.get_db_session() as db:
            odds = db.query(Odd).filter(
                Odd.source_id == market.source_id
            ).order_by(Odd.updated_at.desc()).limit(3).all()
            
            if not odds:
                return None
            
            # Get home and away odds (case insensitive)
            home_odds = next((o for o in odds if o.outcome and o.outcome.lower() == 'home'), None)
            away_odds = next((o for o in odds if o.outcome and o.outcome.lower() == 'away'), None)
            
            if not home_odds or not away_odds:
                return None
            
            # Calculate total implied probability
            home_prob = 1 / home_odds.decimal_odds
            away_prob = 1 / away_odds.decimal_odds
            total_prob = home_prob + away_prob
            
            # Debug logging
            logger.debug(f"Market {market.source_id}: Home={home_odds.decimal_odds:.2f}, Away={away_odds.decimal_odds:.2f}, Total Prob={total_prob:.3f}")
            
            # Simpler strategy: just bet on underdogs with reasonable odds
            if home_odds.decimal_odds > away_odds.decimal_odds and home_odds.decimal_odds > 2.5:
                # Home is underdog with good odds
                return SimpleOrder(
                    order_id=f"ORDER_{market.source_id}_{int(datetime.now().timestamp())}",
                    market_id=market.source_id,
                    home_team=market.home_team,
                    away_team=market.away_team,
                    position="Home",
                    stake=50.0,  # Fixed $50 stake
                    odds=home_odds.decimal_odds,
                    timestamp=datetime.now(timezone.utc),
                    signal="underdog_value",
                    expected_edge=0.05  # Assume 5% edge
                )
            elif away_odds.decimal_odds > home_odds.decimal_odds and away_odds.decimal_odds > 2.5:
                # Away is underdog with good odds
                return SimpleOrder(
                    order_id=f"ORDER_{market.source_id}_{int(datetime.now().timestamp())}",
                    market_id=market.source_id,
                    home_team=market.home_team,
                    away_team=market.away_team,
                    position="Away",
                    stake=50.0,
                    odds=away_odds.decimal_odds,
                    timestamp=datetime.now(timezone.utc),
                    signal="underdog_value",
                    expected_edge=0.05
                )
                        
    except Exception as e:
        logger.error(f"Error evaluating market {market.source_id}: {e}")
        
    return None


def run_simple_live_paper_trading(duration_minutes: int = 10):
    """Run simple live paper trading."""
    logger.info("🚀 Starting Simple Live Paper Trading")
    logger.info("=" * 60)
    
    engine = SimplePaperTradingEngine(initial_capital=1000)
    
    logger.info(f"📊 Trading Session: {engine.session_id}")
    logger.info(f"💰 Initial Capital: ${engine.initial_capital:.2f}")
    logger.info(f"⏱️  Duration: {duration_minutes} minutes")
    
    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    cycle_count = 0
    total_orders = 0
    
    while datetime.now(timezone.utc) < end_time:
        cycle_count += 1
        logger.info(f"\n🔄 Trading Cycle {cycle_count}")
        
        try:
            # Get upcoming markets
            with db_manager.get_db_session() as db:
                markets = db.query(Market).filter(
                    Market.maturity_date > datetime.now(timezone.utc),
                    Market.maturity_date < datetime.now(timezone.utc) + timedelta(hours=24),
                    Market.is_finished == False
                ).limit(10).all()
                
                logger.info(f"   📈 Evaluating {len(markets)} markets...")
                
                orders_this_cycle = 0
                
                for market in markets:
                    # Skip if we already have a position
                    if market.source_id in engine.active_positions:
                        continue
                        
                    order = evaluate_market(market)
                    
                    if order:
                        logger.info(f"\n   🎯 Trading Opportunity Found!")
                        logger.info(f"      Match: {order.home_team} vs {order.away_team}")
                        logger.info(f"      Position: {order.position}")
                        logger.info(f"      Odds: {order.odds:.2f}")
                        logger.info(f"      Stake: ${order.stake:.2f}")
                        logger.info(f"      Expected Edge: {order.expected_edge:.1%}")
                        
                        fill = engine.execute_order(order)
                        total_orders += 1
                        
                        if fill:
                            orders_this_cycle += 1
                            logger.info(f"      ✅ Order filled at {fill.fill_price:.2f}")
                            logger.info(f"      Commission: ${fill.commission:.2f}")
                        else:
                            logger.info(f"      ❌ Order rejected")
                
                if orders_this_cycle == 0:
                    logger.info("   No trading opportunities found")
                    
            # Show portfolio status
            status = engine.get_portfolio_status()
            logger.info(f"\n💼 Portfolio Status:")
            logger.info(f"   Cash: ${status['current_capital']:.2f}")
            logger.info(f"   Portfolio Value: ${status['portfolio_value']:.2f}")
            logger.info(f"   Return: {status['return_pct']:+.1f}%")
            logger.info(f"   Active Positions: {status['active_positions']}")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            
        # Wait 30 seconds before next cycle
        time.sleep(30)
    
    # Final report
    logger.info("\n" + "="*60)
    logger.info("📊 TRADING SESSION COMPLETE")
    logger.info("="*60)
    
    final_status = engine.get_portfolio_status()
    
    logger.info(f"Session ID: {final_status['session_id']}")
    logger.info(f"Initial Capital: ${final_status['initial_capital']:.2f}")
    logger.info(f"Final Portfolio Value: ${final_status['portfolio_value']:.2f}")
    logger.info(f"Total Return: ${final_status['portfolio_value'] - final_status['initial_capital']:+.2f}")
    logger.info(f"Return %: {final_status['return_pct']:+.1f}%")
    logger.info(f"Total Orders: {total_orders}")
    logger.info(f"Total Fills: {final_status['total_trades']}")
    logger.info(f"Total Commission: ${final_status['total_commission']:.2f}")
    
    if engine.active_positions:
        logger.info(f"\n📋 Active Positions:")
        for market_id, position in engine.active_positions.items():
            logger.info(f"   • {position.home_team} vs {position.away_team}")
            logger.info(f"     Position: {position.position} @ {position.odds:.2f}")
            logger.info(f"     Stake: ${position.stake:.2f}")
    
    return final_status


if __name__ == "__main__":
    # Run for 2 minutes as a quick demo
    run_simple_live_paper_trading(duration_minutes=2)