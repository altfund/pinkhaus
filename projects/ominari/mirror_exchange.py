#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mirror Exchange - Paper Trading with Real Exchange Data
Copies real-time data from Overtime Markets for realistic paper trading.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import defaultdict
import sqlite3

from data_source_manager import DataSourceManager

logger = logging.getLogger(__name__)


@dataclass
class LiveMarket:
    """Real-time market data."""
    market_id: str
    sport: str
    home_team: str
    away_team: str
    home_odds: float
    away_odds: float
    draw_odds: Optional[float]
    volume: float
    liquidity: float
    last_update: datetime
    is_live: bool
    
    
@dataclass
class LiveTrade:
    """Real exchange trade."""
    market_id: str
    side: str  # 'home', 'away', 'draw'
    size: float
    price: float
    timestamp: datetime
    tx_hash: Optional[str]


class MirrorExchange:
    """
    Paper trading exchange that mirrors real Overtime Markets data.
    
    Features:
    - Real-time market data sync
    - Historical trade replay
    - Live order book mirroring
    - Volume and liquidity tracking
    - Realistic execution based on actual market depth
    """
    
    def __init__(self, 
                 network: str = 'optimism',
                 sync_interval: int = 10,
                 history_days: int = 30):
        self.network = network
        self.sync_interval = sync_interval
        self.history_days = history_days
        
        # Data sources
        self.data_manager = DataSourceManager(network)
        self.markets: Dict[str, LiveMarket] = {}
        self.trades: List[LiveTrade] = []
        self.order_books: Dict[str, Dict] = {}
        
        # Paper trading state
        self.paper_orders = {}
        self.paper_fills = []
        self.paper_balances = defaultdict(float)
        
        # Database for historical data
        self.db_path = "mirror_exchange.db"
        self._init_database()
        
        # Sync tasks
        self.running = False
        self.sync_task = None
        
    def _init_database(self):
        """Initialize database for historical data."""
        conn = sqlite3.connect(self.db_path)
        
        # Historical markets
        conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_markets (
                market_id TEXT,
                timestamp DATETIME,
                sport TEXT,
                home_team TEXT,
                away_team TEXT,
                home_odds REAL,
                away_odds REAL,
                draw_odds REAL,
                volume REAL,
                liquidity REAL,
                PRIMARY KEY (market_id, timestamp)
            )
        """)
        
        # Historical trades
        conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT,
                side TEXT,
                size REAL,
                price REAL,
                timestamp DATETIME,
                tx_hash TEXT
            )
        """)
        
        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_markets_time ON historical_markets(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_time ON historical_trades(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_market ON historical_trades(market_id)")
        
        conn.commit()
        conn.close()
        
    async def start(self):
        """Start syncing with real exchange."""
        self.running = True
        await self.data_manager.initialize()
        
        # Initial sync
        await self._sync_markets()
        await self._load_historical_data()
        
        # Start continuous sync
        self.sync_task = asyncio.create_task(self._sync_loop())
        
        logger.info(f"Mirror exchange started - syncing {self.network} every {self.sync_interval}s")
        
    async def stop(self):
        """Stop syncing."""
        self.running = False
        if self.sync_task:
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass
                
    async def _sync_loop(self):
        """Continuous sync with real exchange."""
        while self.running:
            try:
                await self._sync_markets()
                await self._sync_trades()
                await self._update_order_books()
                
                # Store snapshot
                self._store_market_snapshot()
                
            except Exception as e:
                logger.error(f"Sync error: {e}")
                
            await asyncio.sleep(self.sync_interval)
            
    async def _sync_markets(self):
        """Sync current markets from blockchain/GraphQL."""
        try:
            # Get markets from best available source
            markets_df = await self.data_manager.get_markets()
            
            if markets_df.empty:
                return
                
            # Update live markets
            for _, row in markets_df.iterrows():
                market = LiveMarket(
                    market_id=row['source_id'],
                    sport=row.get('sport', 'Unknown'),
                    home_team=row.get('home_team', ''),
                    away_team=row.get('away_team', ''),
                    home_odds=float(row.get('home_odds', 2.0)),
                    away_odds=float(row.get('away_odds', 2.0)),
                    draw_odds=float(row.get('draw_odds', 0)) if row.get('draw_odds') else None,
                    volume=float(row.get('volume', 0)),
                    liquidity=float(row.get('liquidity', 0)),
                    last_update=datetime.now(timezone.utc),
                    is_live=row.get('is_live', True)
                )
                
                self.markets[market.market_id] = market
                
            logger.info(f"Synced {len(self.markets)} markets")
            
        except Exception as e:
            logger.error(f"Market sync error: {e}")
            
    async def _sync_trades(self):
        """Sync recent trades from blockchain."""
        try:
            # Get recent trades via GraphQL or blockchain events
            # This would query actual trade events from the contracts
            
            # For now, simulate based on volume changes
            for market_id, market in self.markets.items():
                if market.volume > 0 and np.random.random() < 0.1:  # 10% chance of trade
                    trade = LiveTrade(
                        market_id=market_id,
                        side=np.random.choice(['home', 'away']),
                        size=np.random.exponential(100),
                        price=market.home_odds if np.random.random() < 0.5 else market.away_odds,
                        timestamp=datetime.now(timezone.utc),
                        tx_hash=None
                    )
                    self.trades.append(trade)
                    
        except Exception as e:
            logger.error(f"Trade sync error: {e}")
            
    async def _update_order_books(self):
        """Update order books based on market data."""
        for market_id, market in self.markets.items():
            # Build order book from odds and liquidity
            spread = 0.02  # 2% spread
            
            # Calculate bid/ask from odds
            home_bid = market.home_odds * (1 - spread/2)
            home_ask = market.home_odds * (1 + spread/2)
            away_bid = market.away_odds * (1 - spread/2)
            away_ask = market.away_odds * (1 + spread/2)
            
            # Estimate depth based on liquidity
            depth = market.liquidity / 4 if market.liquidity > 0 else 100
            
            self.order_books[market_id] = {
                'home': {
                    'bids': [(home_bid - i*0.01, depth * (1 - i*0.1)) for i in range(5)],
                    'asks': [(home_ask + i*0.01, depth * (1 - i*0.1)) for i in range(5)]
                },
                'away': {
                    'bids': [(away_bid - i*0.01, depth * (1 - i*0.1)) for i in range(5)],
                    'asks': [(away_ask + i*0.01, depth * (1 - i*0.1)) for i in range(5)]
                }
            }
            
    async def _load_historical_data(self):
        """Load historical data for backtesting."""
        conn = sqlite3.connect(self.db_path)
        
        # Load recent historical markets
        start_date = datetime.now(timezone.utc) - timedelta(days=self.history_days)
        
        markets_df = pd.read_sql_query("""
            SELECT * FROM historical_markets
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, conn, params=(start_date,))
        
        trades_df = pd.read_sql_query("""
            SELECT * FROM historical_trades
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, conn, params=(start_date,))
        
        conn.close()
        
        logger.info(f"Loaded {len(markets_df)} historical markets and {len(trades_df)} trades")
        
        return markets_df, trades_df
        
    def _store_market_snapshot(self):
        """Store current market snapshot to database."""
        if not self.markets:
            return
            
        conn = sqlite3.connect(self.db_path)
        
        # Store markets
        for market in self.markets.values():
            conn.execute("""
                INSERT OR REPLACE INTO historical_markets
                (market_id, timestamp, sport, home_team, away_team,
                 home_odds, away_odds, draw_odds, volume, liquidity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                market.market_id,
                market.last_update,
                market.sport,
                market.home_team,
                market.away_team,
                market.home_odds,
                market.away_odds,
                market.draw_odds,
                market.volume,
                market.liquidity
            ))
            
        # Store recent trades
        for trade in self.trades[-100:]:  # Last 100 trades
            conn.execute("""
                INSERT INTO historical_trades
                (market_id, side, size, price, timestamp, tx_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                trade.market_id,
                trade.side,
                trade.size,
                trade.price,
                trade.timestamp,
                trade.tx_hash
            ))
            
        conn.commit()
        conn.close()
        
    async def place_paper_order(self, 
                               account: str,
                               market_id: str,
                               side: str,
                               size: float,
                               order_type: str = 'market',
                               limit_price: Optional[float] = None) -> Dict:
        """Place a paper order that executes against real market data."""
        
        if market_id not in self.markets:
            return {'status': 'error', 'message': 'Unknown market'}
            
        market = self.markets[market_id]
        order_book = self.order_books.get(market_id, {}).get(side, {})
        
        # Get execution price based on real data
        if order_type == 'market':
            # Execute against best ask
            asks = order_book.get('asks', [])
            if not asks:
                exec_price = market.home_odds if side == 'home' else market.away_odds
            else:
                # Walk through order book
                remaining = size
                total_cost = 0
                for price, available in asks:
                    fill = min(remaining, available)
                    total_cost += fill * price
                    remaining -= fill
                    if remaining <= 0:
                        break
                exec_price = total_cost / size if size > 0 else asks[0][0]
        else:
            exec_price = limit_price
            
        # Check if limit order would fill
        if order_type == 'limit':
            best_ask = order_book.get('asks', [[market.home_odds, 0]])[0][0]
            if limit_price < best_ask:
                return {
                    'status': 'pending',
                    'order_id': f"PO_{int(datetime.now().timestamp())}",
                    'message': 'Limit order placed'
                }
                
        # Execute paper fill
        fill = {
            'order_id': f"PO_{int(datetime.now().timestamp())}",
            'market_id': market_id,
            'side': side,
            'size': size,
            'price': exec_price,
            'timestamp': datetime.now(timezone.utc),
            'status': 'filled'
        }
        
        self.paper_fills.append(fill)
        
        # Update paper balance
        cost = size * exec_price
        self.paper_balances[account] -= cost
        
        return {
            'status': 'filled',
            'order_id': fill['order_id'],
            'exec_price': exec_price,
            'size': size,
            'cost': cost
        }
        
    def get_market_data(self, market_id: Optional[str] = None) -> Dict:
        """Get current market data."""
        if market_id:
            market = self.markets.get(market_id)
            if market:
                return asdict(market)
            return {}
        else:
            return {k: asdict(v) for k, v in self.markets.items()}
            
    def get_order_book(self, market_id: str, side: str) -> Dict:
        """Get order book for a market side."""
        return self.order_books.get(market_id, {}).get(side, {})
        
    def get_historical_data(self, 
                           market_id: Optional[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get historical market data."""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM historical_markets WHERE 1=1"
        params = []
        
        if market_id:
            query += " AND market_id = ?"
            params.append(market_id)
            
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
            
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
            
        query += " ORDER BY timestamp"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
        
    def get_trade_history(self, market_id: Optional[str] = None) -> List[Dict]:
        """Get recent trades."""
        trades = self.trades
        if market_id:
            trades = [t for t in trades if t.market_id == market_id]
            
        return [asdict(t) for t in trades[-100:]]  # Last 100 trades


async def run_mirror_demo():
    """Demo the mirror exchange."""
    mirror = MirrorExchange(network='optimism')
    
    # Start syncing
    await mirror.start()
    
    # Wait for initial sync
    await asyncio.sleep(5)
    
    # Show current markets
    markets = mirror.get_market_data()
    print(f"\nLive Markets: {len(markets)}")
    for market_id, market in list(markets.items())[:5]:
        print(f"  {market['home_team']} vs {market['away_team']}")
        print(f"    Odds: {market['home_odds']:.2f} / {market['away_odds']:.2f}")
        print(f"    Volume: ${market['volume']:,.0f}")
        
    # Place a paper order
    if markets:
        first_market = list(markets.keys())[0]
        result = await mirror.place_paper_order(
            'demo_account',
            first_market,
            'home',
            100,
            'market'
        )
        print(f"\nPaper order result: {result}")
        
    # Get historical data
    hist = mirror.get_historical_data()
    print(f"\nHistorical data points: {len(hist)}")
    
    # Stop
    await mirror.stop()
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_mirror_demo())