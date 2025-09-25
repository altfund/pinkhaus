#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Trading Engine with Real-time Quote Collection
Simulates betting execution with realistic market conditions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import asyncio
import aiohttp
from dataclasses import dataclass, asdict
import time
from position_manager import PositionManager, Position, RiskLimits
import os
import psycopg2
from database_v2 import db_manager
from models import Market, Odd

# Set PostgreSQL environment for paper trading
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """Market quote snapshot."""
    source_id: str
    timestamp: datetime
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    mid_price: float
    spread: float
    liquidity_score: float
    
    
@dataclass
class PaperOrder:
    """Paper trading order."""
    order_id: str
    timestamp: datetime
    source_id: str
    market_type: str
    bet_name: str
    side: str  # 'buy' or 'sell'
    size: float
    limit_price: Optional[float]
    signal_name: str
    expected_edge: float
    
    
@dataclass
class PaperFill:
    """Simulated order execution."""
    fill_id: str
    order_id: str
    timestamp: datetime
    fill_price: float
    fill_size: float
    slippage: float
    commission: float
    market_impact: float


class QuoteCollector:
    """Collects real-time quotes from Overtime Markets."""
    
    def __init__(self, api_base_url: str):
        self.api_base_url = api_base_url
        self._init_database()
        
    def _init_database(self):
        """Initialize quote storage in PostgreSQL."""
        conn = psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
        cur = conn.cursor()
        
        # Create quotes table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS quotes (
                id SERIAL PRIMARY KEY,
                source_id TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                bid_price DECIMAL(10,4),
                bid_size DECIMAL(10,2),
                ask_price DECIMAL(10,4),
                ask_size DECIMAL(10,2),
                mid_price DECIMAL(10,4),
                spread DECIMAL(10,4),
                liquidity_score DECIMAL(5,4),
                raw_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_quotes_source_time 
            ON quotes(source_id, timestamp)
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        
    async def fetch_quote(self, source_id: str) -> Optional[Quote]:
        """Fetch current quote for a market from PostgreSQL odds data."""
        try:
            # Get current odds from our database
            with db_manager.get_db_session() as db:
                # Get market and odds for this source_id
                market = db.query(Market).filter(Market.source_id == source_id).first()
                if not market:
                    logger.warning(f"Market not found: {source_id}")
                    return None
                
                # Get current odds for this market
                odds = db.query(Odd).filter(
                    Odd.source_id == source_id
                ).order_by(Odd.updated_at.desc()).limit(3).all()
                
                if not odds:
                    logger.warning(f"No odds found for market: {source_id}")
                    return None
                
                # Convert odds to bid/ask prices with realistic spread
                home_odds = next((o.decimal_odds for o in odds if o.outcome == 'Home'), None)
                away_odds = next((o.decimal_odds for o in odds if o.outcome == 'Away'), None)
                
                if not home_odds or not away_odds:
                    logger.warning(f"Missing home/away odds for: {source_id}")
                    return None
                
                # Use home odds as base price for quote generation
                base_price = home_odds
                spread_pct = 0.02 + np.random.random() * 0.03  # 2-5% spread
                
                bid_price = base_price * (1 - spread_pct/2)
                ask_price = base_price * (1 + spread_pct/2)
                
                quote = Quote(
                    source_id=source_id,
                    timestamp=datetime.now(timezone.utc),
                    bid_price=bid_price,
                    bid_size=100 + np.random.random() * 900,
                    ask_price=ask_price,
                    ask_size=100 + np.random.random() * 900,
                    mid_price=(bid_price + ask_price) / 2,
                    spread=ask_price - bid_price,
                    liquidity_score=np.random.random()
                )
                
                self._store_quote(quote)
                return quote
                
        except Exception as e:
            logger.error(f"Error fetching quote for {source_id}: {e}")
            return None
            
    def _store_quote(self, quote: Quote):
        """Store quote in PostgreSQL database."""
        conn = psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO quotes (
                source_id, timestamp, bid_price, bid_size,
                ask_price, ask_size, mid_price, spread,
                liquidity_score, raw_data
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            quote.source_id, quote.timestamp, quote.bid_price,
            quote.bid_size, quote.ask_price, quote.ask_size,
            quote.mid_price, quote.spread, quote.liquidity_score,
            json.dumps(asdict(quote), default=str)
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
    def get_historical_quotes(self, source_id: str, 
                            start_time: datetime,
                            end_time: datetime) -> pd.DataFrame:
        """Retrieve historical quotes for analysis."""
        conn = psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
        
        df = pd.read_sql_query("""
            SELECT * FROM quotes
            WHERE source_id = %s
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
        """, conn, params=(source_id, start_time, end_time))
        
        conn.close()
        return df


class MarketImpactModel:
    """Models market impact and slippage."""
    
    def __init__(self, base_impact: float = 0.001, 
                 size_factor: float = 0.0001):
        self.base_impact = base_impact
        self.size_factor = size_factor
        
    def estimate_impact(self, size: float, liquidity: float, 
                       volatility: float) -> float:
        """Estimate market impact for order size."""
        # Square-root market impact model
        normalized_size = size / max(liquidity, 1.0)
        impact = self.base_impact * np.sqrt(normalized_size)
        
        # Adjust for volatility
        impact *= (1 + volatility)
        
        return min(impact, 0.05)  # Cap at 5%
        
    def calculate_slippage(self, quote: Quote, size: float, 
                          side: str) -> Tuple[float, float]:
        """Calculate execution price with slippage."""
        if side == 'buy':
            base_price = quote.ask_price
            available_size = quote.ask_size
        else:
            base_price = quote.bid_price
            available_size = quote.bid_size
            
        # Simple linear slippage model
        if size <= available_size:
            slippage = 0.0
        else:
            excess_ratio = (size - available_size) / available_size
            slippage = quote.spread * excess_ratio * 0.5
            
        if side == 'buy':
            execution_price = base_price + slippage
        else:
            execution_price = base_price - slippage
            
        return execution_price, slippage


class PaperTradingEngine:
    """Simulates betting execution with realistic conditions."""
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 commission_rate: float = 0.002,
                 risk_limits: Optional[RiskLimits] = None,
                 session_id: Optional[str] = None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        self.session_id = session_id
        self.session_active = False
        
        self.quote_collector = QuoteCollector("https://api.overtime.markets")
        self.impact_model = MarketImpactModel()
        
        self.orders: Dict[str, PaperOrder] = {}
        self.fills: List[PaperFill] = []
        self.positions: Dict[str, float] = {}  # Legacy simple tracking
        
        # Advanced position management
        self.position_manager = PositionManager(
            capital=initial_capital,
            risk_limits=risk_limits or RiskLimits(
                max_position_size=0.1,  # Max 10% per position
                max_cluster_exposure=0.25,  # Max 25% in correlated bets
                max_total_exposure=0.6,  # Max 60% total
                max_loss_per_position=0.02,  # 2% stop loss
                min_rebalance_threshold=0.05,
                min_trade_size=0.001
            )
        )
        
        self._init_database()
        
        # Load or create session
        if session_id:
            self.load_session(session_id)
        else:
            self.create_session()
        
    def _init_database(self):
        """Initialize paper trading tables in PostgreSQL."""
        conn = psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
        cur = conn.cursor()
        
        # Orders table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_orders (
                order_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                source_id TEXT NOT NULL,
                market_type TEXT,
                bet_name TEXT,
                side TEXT,
                size DECIMAL(15,6),
                limit_price DECIMAL(10,4),
                signal_name TEXT,
                expected_edge DECIMAL(6,4),
                status TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Fills table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_fills (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                fill_price DECIMAL(10,4),
                fill_size DECIMAL(15,6),
                slippage DECIMAL(8,6),
                commission DECIMAL(10,4),
                market_impact DECIMAL(10,4),
                pnl DECIMAL(15,6),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                FOREIGN KEY (order_id) REFERENCES paper_orders(order_id)
            )
        """)
        
        # Performance tracking
        cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_performance (
                timestamp TIMESTAMP WITH TIME ZONE PRIMARY KEY,
                capital DECIMAL(15,6),
                positions_value DECIMAL(15,6),
                total_value DECIMAL(15,6),
                daily_pnl DECIMAL(15,6),
                total_pnl DECIMAL(15,6),
                sharpe_ratio DECIMAL(8,4),
                max_drawdown DECIMAL(8,4),
                win_rate DECIMAL(6,4),
                avg_win DECIMAL(15,6),
                avg_loss DECIMAL(15,6)
            )
        """)
        
        # Session management
        cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_sessions (
                session_id TEXT PRIMARY KEY,
                start_time TIMESTAMP WITH TIME ZONE NOT NULL,
                end_time TIMESTAMP WITH TIME ZONE,
                initial_capital DECIMAL(15,6) NOT NULL,
                final_capital DECIMAL(15,6),
                status TEXT CHECK(status IN ('active', 'completed', 'archived')) DEFAULT 'active',
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Position tracking with session linkage
        cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_positions (
                position_id TEXT PRIMARY KEY,
                session_id TEXT,
                market_id TEXT NOT NULL,
                bet_name TEXT,
                size DECIMAL(15,6) NOT NULL,
                entry_price DECIMAL(10,4) NOT NULL,
                entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
                exit_price DECIMAL(10,4),
                exit_time TIMESTAMP WITH TIME ZONE,
                status TEXT CHECK(status IN ('open', 'closed')) DEFAULT 'open',
                pnl DECIMAL(15,6),
                metadata JSONB,
                FOREIGN KEY (session_id) REFERENCES paper_sessions(session_id)
            )
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        
    async def submit_order(self, order: PaperOrder) -> PaperFill:
        """Submit a paper order and simulate execution."""
        logger.info(f"Submitting paper order: {order.order_id}")
        
        # Fetch current quote
        quote = await self.quote_collector.fetch_quote(order.source_id)
        if not quote:
            logger.error(f"No quote available for {order.source_id}")
            return None
            
        # Calculate execution details
        exec_price, slippage = self.impact_model.calculate_slippage(
            quote, order.size, order.side
        )
        
        # Check limit price
        if order.limit_price:
            if order.side == 'buy' and exec_price > order.limit_price:
                logger.info(f"Order {order.order_id} not filled - price too high")
                return None
            elif order.side == 'sell' and exec_price < order.limit_price:
                logger.info(f"Order {order.order_id} not filled - price too low")
                return None
        
        # Create Position object for risk checking
        new_position = Position(
            market_id=order.source_id,
            bet_name=order.bet_name,
            size=order.size if order.side == 'buy' else -order.size,
            entry_price=exec_price,
            current_price=exec_price,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Check risk limits before execution
        is_update = order.source_id in self.position_manager.positions
        valid, reason = self.position_manager.check_risk_limits(new_position, is_update=is_update)
        
        if not valid:
            logger.warning(f"Order {order.order_id} rejected - Risk limit: {reason}")
            return None
                
        # Calculate costs
        commission = order.size * exec_price * self.commission_rate
        market_impact = self.impact_model.estimate_impact(
            order.size, 
            (quote.bid_size + quote.ask_size) / 2,
            0.02  # Assumed volatility
        )
        
        # Create fill
        fill = PaperFill(
            fill_id=f"F{int(time.time() * 1000)}",
            order_id=order.order_id,
            timestamp=datetime.now(timezone.utc),
            fill_price=exec_price,
            fill_size=order.size,
            slippage=slippage,
            commission=commission,
            market_impact=market_impact * exec_price * order.size
        )
        
        # Store order in memory
        self.orders[order.order_id] = order
        
        # Update positions and capital
        self._process_fill(order, fill)
        
        # Store in database
        self._store_order(order)
        self._store_fill(fill)
        
        return fill
        
    def _process_fill(self, order: PaperOrder, fill: PaperFill):
        """Update positions and capital after fill."""
        position_key = f"{order.source_id}_{order.bet_name}"
        
        # Update legacy position tracking
        if order.side == 'buy':
            self.positions[position_key] = \
                self.positions.get(position_key, 0) + fill.fill_size
            self.current_capital -= \
                (fill.fill_size * fill.fill_price + fill.commission)
        else:
            self.positions[position_key] = \
                self.positions.get(position_key, 0) - fill.fill_size
            self.current_capital += \
                (fill.fill_size * fill.fill_price - fill.commission)
        
        # Update advanced position tracking
        existing_position = self.position_manager.positions.get(order.source_id)
        
        if existing_position:
            # Update existing position
            new_size = existing_position.size + (fill.fill_size if order.side == 'buy' else -fill.fill_size)
            if abs(new_size) < 0.0001:  # Position closed
                del self.position_manager.positions[order.source_id]
                logger.info(f"Position closed: {order.source_id}")
            else:
                # Update position with new average price
                total_cost = existing_position.size * existing_position.entry_price
                fill_cost = fill.fill_size * fill.fill_price * (1 if order.side == 'buy' else -1)
                new_avg_price = abs((total_cost + fill_cost) / new_size)
                
                updated_position = Position(
                    market_id=order.source_id,
                    bet_name=order.bet_name,
                    size=new_size,
                    entry_price=new_avg_price,
                    current_price=fill.fill_price,
                    timestamp=datetime.now(timezone.utc)
                )
                self.position_manager.add_position(updated_position)
        else:
            # New position
            new_position = Position(
                market_id=order.source_id,
                bet_name=order.bet_name,
                size=fill.fill_size if order.side == 'buy' else -fill.fill_size,
                entry_price=fill.fill_price,
                current_price=fill.fill_price,
                timestamp=datetime.now(timezone.utc)
            )
            self.position_manager.add_position(new_position)
            
            # Save to database
            self.save_position(new_position, fill)
        
        # Update position manager capital
        self.position_manager.capital = self.current_capital
                
        self.fills.append(fill)
        
    def _store_order(self, order: PaperOrder):
        """Store order in PostgreSQL database."""
        conn = psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO paper_orders (
                order_id, timestamp, source_id, market_type,
                bet_name, side, size, limit_price,
                signal_name, expected_edge, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            order.order_id, order.timestamp, order.source_id,
            order.market_type, order.bet_name, order.side,
            order.size, order.limit_price, order.signal_name,
            order.expected_edge, 'filled'
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
    def _store_fill(self, fill: PaperFill):
        """Store fill in PostgreSQL database."""
        conn = psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO paper_fills (
                fill_id, order_id, timestamp, fill_price,
                fill_size, slippage, commission, market_impact
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            fill.fill_id, fill.order_id, fill.timestamp,
            fill.fill_price, fill.fill_size, fill.slippage,
            fill.commission, fill.market_impact
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
    def calculate_performance(self) -> Dict[str, float]:
        """Calculate current performance metrics."""
        if not self.fills:
            return {
                'total_pnl': 0,
                'total_value': self.current_capital,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_trades': 0,
                'total_commission': 0,
                'total_slippage': 0
            }
            
        # Calculate P&L for each fill
        pnls = []
        for fill in self.fills:
            # This is simplified - would need outcome data
            pnl = -fill.commission  # At minimum, we pay commission
            pnls.append(pnl)
            
        total_pnl = sum(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        metrics = {
            'total_pnl': total_pnl,
            'total_value': self.current_capital + sum(self.positions.values()),
            'win_rate': len(wins) / len(pnls) if pnls else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'total_trades': len(self.fills),
            'total_commission': sum(f.commission for f in self.fills),
            'total_slippage': sum(f.slippage for f in self.fills)
        }
        
        # Calculate Sharpe ratio (simplified)
        if len(pnls) > 1:
            returns = pd.Series(pnls).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                metrics['sharpe_ratio'] = \
                    np.sqrt(252) * returns.mean() / returns.std()
            else:
                metrics['sharpe_ratio'] = 0
        else:
            metrics['sharpe_ratio'] = 0
            
        return metrics
        
    def generate_report(self) -> pd.DataFrame:
        """Generate detailed performance report."""
        metrics = self.calculate_performance()
        
        report = pd.DataFrame([{
            'timestamp': datetime.now(timezone.utc),
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_pnl': metrics['total_pnl'],
            'total_value': metrics['total_value'],
            'return_pct': (metrics['total_value'] - self.initial_capital) / 
                         self.initial_capital * 100,
            'total_trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'total_commission': metrics['total_commission'],
            'total_slippage': metrics['total_slippage'],
            'avg_win': metrics['avg_win'],
            'avg_loss': metrics['avg_loss']
        }])
        
        return report
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value including open positions."""
        # Cash plus value of open positions
        position_value = 0
        for position in self.position_manager.positions.values():
            position_value += position.current_value
        
        return self.current_capital + position_value
    
    def get_open_positions(self) -> pd.DataFrame:
        """Get DataFrame of current open positions."""
        summary = self.position_manager.get_position_summary()
        
        # Add bet_name to the summary if we have it in our orders
        if not summary.empty:
            for idx, row in summary.iterrows():
                market_id = row['market_id']
                # Find the original order to get bet_name
                for order_id, order in self.orders.items():
                    if order.source_id == market_id:
                        summary.at[idx, 'bet_name'] = order.bet_name
                        break
        
        return summary
    
    def get_position_exposure(self) -> Dict[str, Any]:
        """Get position exposure analytics."""
        return {
            'total_exposure': self.position_manager.get_total_exposure(),
            'cluster_exposure': self.position_manager.net_cluster_positions(),
            'position_count': len(self.position_manager.positions),
            'largest_position': max(
                (abs(p.current_value) for p in self.position_manager.positions.values()),
                default=0
            ) / self.initial_capital if self.position_manager.positions else 0
        }
    
    def update_position_prices(self, current_prices: Dict[str, float]):
        """Update current prices for all positions."""
        for market_id, price in current_prices.items():
            if market_id in self.position_manager.positions:
                position = self.position_manager.positions[market_id]
                old_price = position.current_price
                position.current_price = price
                
                # Update in database if price changed significantly
                if abs(price - old_price) / old_price > 0.001:  # 0.1% change
                    self.update_position(market_id, price)
                    
                logger.debug(f"Updated price for {market_id}: ${price:.3f}")
    
    def check_stop_losses(self) -> List[str]:
        """Check and return positions that hit stop loss."""
        stopped = self.position_manager.check_stop_losses()
        
        # Auto-close stopped positions
        for market_id in stopped:
            position = self.position_manager.positions.get(market_id)
            if position:
                # Create closing order
                close_order = PaperOrder(
                    order_id=f"SL_{int(time.time() * 1000)}",
                    timestamp=datetime.now(timezone.utc),
                    source_id=market_id,
                    market_type="",
                    bet_name=position.bet_name,
                    side='sell' if position.size > 0 else 'buy',
                    size=abs(position.size),
                    limit_price=None,  # Market order
                    signal_name='stop_loss',
                    expected_edge=0
                )
                # Note: This would need to be async in practice
                logger.warning(f"Stop loss triggered for {market_id}, closing position")
        
        return stopped
    
    def create_session(self, name: Optional[str] = None) -> str:
        """Create a new trading session."""
        self.session_id = f"S{int(time.time() * 1000)}"
        
        metadata = {
            "name": name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "initial_capital": self.initial_capital,
            "commission_rate": self.commission_rate,
            "risk_limits": asdict(self.position_manager.risk_limits)
        }
        
        conn = psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO paper_sessions (
                session_id, start_time, initial_capital, status, metadata
            ) VALUES (%s, %s, %s, %s, %s)
        """, (
            self.session_id,
            datetime.now(timezone.utc),
            self.initial_capital,
            'active',
            json.dumps(metadata)
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        self.session_active = True
        logger.info(f"Created new session: {self.session_id}")
        return self.session_id
    
    def load_session(self, session_id: str):
        """Load an existing session."""
        conn = psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
        cur = conn.cursor()
        
        # Load session info
        cur.execute("""
            SELECT * FROM paper_sessions WHERE session_id = %s
        """, (session_id,))
        
        row = cur.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Session {session_id} not found")
        
        # Parse session data
        cols = [desc[0] for desc in cur.description]
        session = dict(zip(cols, row))
        
        if session['status'] != 'active':
            conn.close()
            raise ValueError(f"Session {session_id} is not active (status: {session['status']})")
        
        self.session_id = session_id
        self.initial_capital = float(session['initial_capital'])
        self.session_active = True
        
        # Load metadata
        if session['metadata']:
            metadata = session['metadata']  # Already JSON in PostgreSQL JSONB
            self.commission_rate = metadata.get('commission_rate', self.commission_rate)
        
        # Load open positions for this session
        cur.execute("""
            SELECT * FROM paper_positions 
            WHERE session_id = %s AND status = 'open'
        """, (session_id,))
        
        for row in cur.fetchall():
            pos_data = dict(zip([desc[0] for desc in cur.description], row))
            
            # Recreate position in position manager
            position = Position(
                market_id=pos_data['market_id'],
                bet_name=pos_data['bet_name'],
                size=float(pos_data['size']),
                entry_price=float(pos_data['entry_price']),
                current_price=float(pos_data['entry_price']),  # Will be updated
                timestamp=pos_data['entry_time']
            )
            self.position_manager.add_position(position)
        
        # Load recent fills for this session
        cur.execute("""
            SELECT f.* FROM paper_fills f
            JOIN paper_orders o ON f.order_id = o.order_id
            WHERE o.created_at >= (SELECT start_time FROM paper_sessions WHERE session_id = %s)
            ORDER BY f.timestamp DESC
            LIMIT 100
        """, (session_id,))
        
        # TODO: Reconstruct fills if needed
        
        conn.close()
        logger.info(f"Loaded session {session_id} with {len(self.position_manager.positions)} open positions")
    
    def stop_session(self) -> Dict[str, Any]:
        """Stop the current session and calculate final metrics."""
        if not self.session_active:
            raise ValueError("No active session to stop")
        
        # Calculate final metrics
        final_metrics = self.calculate_performance()
        final_value = self.get_portfolio_value()
        
        # Update session record
        conn = psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
        cur = conn.cursor()
        
        cur.execute("""
            UPDATE paper_sessions 
            SET end_time = %s, final_capital = %s, status = 'completed'
            WHERE session_id = %s
        """, (
            datetime.now(timezone.utc),
            final_value,
            self.session_id
        ))
        
        # Close all open positions in database
        cur.execute("""
            UPDATE paper_positions 
            SET status = 'closed', exit_time = %s, exit_price = entry_price
            WHERE session_id = %s AND status = 'open'
        """, (
            datetime.now(timezone.utc),
            self.session_id
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        self.session_active = False
        
        return {
            "session_id": self.session_id,
            "duration": "calculated",  # TODO: Calculate actual duration
            "initial_capital": self.initial_capital,
            "final_capital": final_value,
            "total_return": (final_value - self.initial_capital) / self.initial_capital,
            "metrics": final_metrics
        }
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        if not self.session_id:
            return {"status": "No active session"}
        
        conn = psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
        cur = conn.cursor()
        
        cur.execute("""
            SELECT * FROM paper_sessions WHERE session_id = %s
        """, (self.session_id,))
        
        row = cur.fetchone()
        if row:
            cols = [desc[0] for desc in cur.description]
            session = dict(zip(cols, row))
            
            # Add current metrics
            session['current_capital'] = self.current_capital
            session['portfolio_value'] = self.get_portfolio_value()
            session['open_positions'] = len(self.position_manager.positions)
            session['total_trades'] = len(self.fills)
            
            conn.close()
            return session
        
        conn.close()
        return {"status": "Session not found"}
    
    def save_position(self, position: Position, fill: PaperFill):
        """Save position to PostgreSQL database."""
        if not self.session_id:
            return
        
        position_id = f"P{int(time.time() * 1000)}"
        
        metadata = {
            "fill_id": fill.fill_id,
            "order_id": fill.order_id,
            "slippage": fill.slippage,
            "commission": fill.commission
        }
        
        conn = psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO paper_positions (
                position_id, session_id, market_id, bet_name,
                size, entry_price, entry_time, status, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            position_id,
            self.session_id,
            position.market_id,
            position.bet_name,
            position.size,
            position.entry_price,
            position.timestamp,
            'open',
            json.dumps(metadata)
        ))
        
        conn.commit()
        cur.close()
        conn.close()
    
    def update_position(self, market_id: str, current_price: float, pnl: Optional[float] = None):
        """Update position price and P&L in PostgreSQL database."""
        if not self.session_id:
            return
        
        conn = psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
        cur = conn.cursor()
        
        if pnl is not None:
            # Position closed
            cur.execute("""
                UPDATE paper_positions 
                SET exit_price = %s, exit_time = %s, status = 'closed', pnl = %s
                WHERE session_id = %s AND market_id = %s AND status = 'open'
            """, (
                current_price,
                datetime.now(timezone.utc),
                pnl,
                self.session_id,
                market_id
            ))
        
        conn.commit()
        cur.close()
        conn.close()
    
    def load_positions(self):
        """Load all positions from PostgreSQL database on startup."""
        if not self.session_id:
            return
        
        conn = psycopg2.connect(
            host='localhost',
            port=5999,
            database='ominari_production',
            user='ominari_user',
            password='ominari_2025_secure'
        )
        cur = conn.cursor()
        
        # Load all positions for current session
        cur.execute("""
            SELECT * FROM paper_positions 
            WHERE session_id = %s
            ORDER BY entry_time DESC
        """, (self.session_id,))
        
        positions = []
        for row in cur.fetchall():
            cols = [desc[0] for desc in cur.description]
            pos_data = dict(zip(cols, row))
            positions.append(pos_data)
        
        conn.close()
        
        # Reconstruct position state
        logger.info(f"Loaded {len(positions)} positions from database")
        return positions


async def main():
    """Example usage of paper trading engine."""
    engine = PaperTradingEngine(initial_capital=10000)
    
    # Example order
    order = PaperOrder(
        order_id=f"O{int(time.time() * 1000)}",
        timestamp=datetime.now(timezone.utc),
        source_id="0x12345",
        market_type="winner",
        bet_name="TeamA",
        side="buy",
        size=100,
        limit_price=2.05,
        signal_name="implied_probability",
        expected_edge=0.03
    )
    
    # Submit order
    fill = await engine.submit_order(order)
    if fill:
        logger.info(f"Order filled: {fill.fill_price} "
                   f"(slippage: {fill.slippage:.4f})")
    
    # Generate report
    report = engine.generate_report()
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())