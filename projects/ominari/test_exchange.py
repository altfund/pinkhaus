#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mock Exchange for Testing
Simulates a realistic betting exchange for comprehensive testing.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict
import threading
import uuid

logger = logging.getLogger(__name__)


@dataclass
class OrderBook:
    """Represents an order book for a market."""
    market_id: str
    timestamp: datetime
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]  # [(price, size), ...]
    
    @property
    def best_bid(self) -> Optional[Tuple[float, float]]:
        """Get best bid (highest price)."""
        return max(self.bids, key=lambda x: x[0]) if self.bids else None
        
    @property
    def best_ask(self) -> Optional[Tuple[float, float]]:
        """Get best ask (lowest price)."""
        return min(self.asks, key=lambda x: x[0]) if self.asks else None
        
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        bid = self.best_bid
        ask = self.best_ask
        if bid and ask:
            return (bid[0] + ask[0]) / 2
        elif bid:
            return bid[0]
        elif ask:
            return ask[0]
        return 0
        
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        bid = self.best_bid
        ask = self.best_ask
        if bid and ask:
            return ask[0] - bid[0]
        return 0


@dataclass 
class Order:
    """Represents an order."""
    order_id: str
    market_id: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market' or 'limit'
    size: float
    price: Optional[float]  # None for market orders
    timestamp: datetime
    status: str  # 'pending', 'filled', 'partial', 'cancelled', 'rejected'
    filled_size: float = 0
    avg_fill_price: float = 0
    fills: List[Dict] = None
    
    def __post_init__(self):
        if self.fills is None:
            self.fills = []


@dataclass
class Fill:
    """Represents an order fill."""
    fill_id: str
    order_id: str
    market_id: str
    size: float
    price: float
    timestamp: datetime
    liquidity: str  # 'maker' or 'taker'
    fee: float


class MockExchange:
    """
    Mock betting exchange for testing.
    
    Features:
    - Realistic order book simulation
    - Market and limit orders
    - Partial fills
    - Slippage and market impact
    - Random walk price movements
    - Latency simulation
    - Order rejection scenarios
    """
    
    def __init__(self, 
                 initial_markets: Optional[List[str]] = None,
                 latency_ms: Tuple[int, int] = (5, 50),
                 rejection_rate: float = 0.02,
                 partial_fill_rate: float = 0.1):
        self.markets = {}  # market_id -> OrderBook
        self.orders = {}   # order_id -> Order
        self.fills = []    # List of Fill objects
        self.balances = defaultdict(float)  # account -> balance
        
        # Configuration
        self.latency_range = latency_ms
        self.rejection_rate = rejection_rate
        self.partial_fill_rate = partial_fill_rate
        
        # Price simulation
        self.price_volatility = 0.001
        self.price_momentum = 0.7
        self.last_prices = {}
        
        # Initialize markets
        if initial_markets:
            for market_id in initial_markets:
                self._initialize_market(market_id)
                
        # Start background tasks
        self.running = True
        self.price_thread = threading.Thread(target=self._price_updater)
        self.price_thread.daemon = True
        self.price_thread.start()
        
    def _initialize_market(self, market_id: str):
        """Initialize a market with random order book."""
        # Random starting price between 1.5 and 4.0
        mid_price = 1.5 + np.random.random() * 2.5
        
        # Generate order book levels
        bids = []
        asks = []
        
        # Create 5 levels on each side
        for i in range(5):
            # Bids (below mid)
            bid_price = mid_price - (i + 1) * 0.01
            bid_size = 100 + np.random.exponential(200)
            bids.append((bid_price, bid_size))
            
            # Asks (above mid)
            ask_price = mid_price + (i + 1) * 0.01
            ask_size = 100 + np.random.exponential(200)
            asks.append((ask_price, ask_size))
            
        self.markets[market_id] = OrderBook(
            market_id=market_id,
            timestamp=datetime.now(timezone.utc),
            bids=sorted(bids, reverse=True),
            asks=sorted(asks)
        )
        
        self.last_prices[market_id] = mid_price
        
    async def connect(self, account_id: str, initial_balance: float = 10000):
        """Connect to the exchange with an account."""
        # Simulate connection latency
        await self._simulate_latency()
        
        self.balances[account_id] = initial_balance
        logger.info(f"Connected to MockExchange: account={account_id}, balance=${initial_balance}")
        
        return {
            'status': 'connected',
            'account_id': account_id,
            'balance': initial_balance,
            'markets': list(self.markets.keys())
        }
        
    async def get_markets(self) -> List[Dict[str, Any]]:
        """Get available markets."""
        await self._simulate_latency()
        
        markets = []
        for market_id, book in self.markets.items():
            markets.append({
                'market_id': market_id,
                'status': 'open',
                'bid': book.best_bid[0] if book.best_bid else None,
                'ask': book.best_ask[0] if book.best_ask else None,
                'mid': book.mid_price,
                'spread': book.spread,
                'last_update': book.timestamp.isoformat()
            })
            
        return markets
        
    async def get_order_book(self, market_id: str) -> Dict[str, Any]:
        """Get order book for a market."""
        await self._simulate_latency()
        
        if market_id not in self.markets:
            raise ValueError(f"Unknown market: {market_id}")
            
        book = self.markets[market_id]
        
        return {
            'market_id': market_id,
            'timestamp': book.timestamp.isoformat(),
            'bids': [{'price': p, 'size': s} for p, s in book.bids],
            'asks': [{'price': p, 'size': s} for p, s in book.asks],
            'mid_price': book.mid_price,
            'spread': book.spread
        }
        
    async def place_order(self, 
                         account_id: str,
                         market_id: str,
                         side: str,
                         order_type: str,
                         size: float,
                         price: Optional[float] = None) -> Order:
        """Place an order."""
        await self._simulate_latency()
        
        # Validate inputs
        if market_id not in self.markets:
            raise ValueError(f"Unknown market: {market_id}")
            
        if side not in ['buy', 'sell']:
            raise ValueError(f"Invalid side: {side}")
            
        if order_type not in ['market', 'limit']:
            raise ValueError(f"Invalid order type: {order_type}")
            
        if order_type == 'limit' and price is None:
            raise ValueError("Limit orders require a price")
            
        # Check for random rejection
        if np.random.random() < self.rejection_rate:
            order = Order(
                order_id=str(uuid.uuid4()),
                market_id=market_id,
                side=side,
                order_type=order_type,
                size=size,
                price=price,
                timestamp=datetime.now(timezone.utc),
                status='rejected'
            )
            self.orders[order.order_id] = order
            logger.warning(f"Order rejected: {order.order_id}")
            return order
            
        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            market_id=market_id,
            side=side,
            order_type=order_type,
            size=size,
            price=price,
            timestamp=datetime.now(timezone.utc),
            status='pending'
        )
        
        self.orders[order.order_id] = order
        
        # Process order
        await self._process_order(account_id, order)
        
        return order
        
    async def _process_order(self, account_id: str, order: Order):
        """Process an order against the order book."""
        book = self.markets[order.market_id]
        remaining_size = order.size
        
        # Get relevant side of book
        if order.side == 'buy':
            levels = book.asks  # Buy from asks
        else:
            levels = book.bids  # Sell to bids
            
        # Check for partial fill
        if np.random.random() < self.partial_fill_rate:
            # Only fill part of the order
            fill_ratio = 0.3 + np.random.random() * 0.6
            max_fill = order.size * fill_ratio
        else:
            max_fill = order.size
            
        # Process fills
        for price, available_size in levels:
            if remaining_size <= 0 or order.filled_size >= max_fill:
                break
                
            # Check price for limit orders
            if order.order_type == 'limit':
                if order.side == 'buy' and price > order.price:
                    break
                elif order.side == 'sell' and price < order.price:
                    break
                    
            # Calculate fill size
            fill_size = min(remaining_size, available_size, max_fill - order.filled_size)
            
            # Apply market impact
            impact_price = self._calculate_impact_price(price, fill_size, order.side)
            
            # Create fill
            fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=order.order_id,
                market_id=order.market_id,
                size=fill_size,
                price=impact_price,
                timestamp=datetime.now(timezone.utc),
                liquidity='taker',
                fee=fill_size * impact_price * 0.002  # 0.2% fee
            )
            
            self.fills.append(fill)
            order.fills.append(asdict(fill))
            
            # Update order
            order.filled_size += fill_size
            order.avg_fill_price = (
                (order.avg_fill_price * (order.filled_size - fill_size) + 
                 impact_price * fill_size) / order.filled_size
            )
            
            remaining_size -= fill_size
            
            # Update balance
            if order.side == 'buy':
                cost = fill_size * impact_price + fill.fee
                self.balances[account_id] -= cost
            else:
                revenue = fill_size * impact_price - fill.fee
                self.balances[account_id] += revenue
                
        # Update order status
        if order.filled_size >= order.size:
            order.status = 'filled'
        elif order.filled_size > 0:
            order.status = 'partial'
        else:
            order.status = 'pending'
            
    def _calculate_impact_price(self, base_price: float, size: float, side: str) -> float:
        """Calculate price with market impact."""
        # Impact increases with size
        impact_pct = 0.0001 * np.sqrt(size / 100)
        
        if side == 'buy':
            # Buying pushes price up
            return base_price * (1 + impact_pct)
        else:
            # Selling pushes price down
            return base_price * (1 - impact_pct)
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        await self._simulate_latency()
        
        if order_id not in self.orders:
            return False
            
        order = self.orders[order_id]
        
        if order.status in ['filled', 'cancelled', 'rejected']:
            return False
            
        order.status = 'cancelled'
        logger.info(f"Order cancelled: {order_id}")
        
        return True
        
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status."""
        await self._simulate_latency()
        
        return self.orders.get(order_id)
        
    async def get_fills(self, account_id: str, limit: int = 100) -> List[Fill]:
        """Get recent fills."""
        await self._simulate_latency()
        
        # In real implementation, filter by account
        return self.fills[-limit:]
        
    async def get_balance(self, account_id: str) -> float:
        """Get account balance."""
        await self._simulate_latency()
        
        return self.balances.get(account_id, 0)
        
    def _price_updater(self):
        """Background thread to update prices."""
        while self.running:
            try:
                for market_id, book in self.markets.items():
                    # Random walk with momentum
                    last_price = self.last_prices.get(market_id, book.mid_price)
                    
                    # Generate price change
                    innovation = np.random.normal(0, self.price_volatility)
                    momentum = (last_price - book.mid_price) * self.price_momentum
                    price_change = innovation + momentum
                    
                    # Update all levels
                    new_bids = []
                    new_asks = []
                    
                    for price, size in book.bids:
                        new_price = price * (1 + price_change)
                        # Random size changes
                        new_size = size * (0.8 + np.random.random() * 0.4)
                        new_bids.append((new_price, new_size))
                        
                    for price, size in book.asks:
                        new_price = price * (1 + price_change)
                        new_size = size * (0.8 + np.random.random() * 0.4)
                        new_asks.append((new_price, new_size))
                        
                    # Update book
                    book.bids = sorted(new_bids, reverse=True)
                    book.asks = sorted(new_asks)
                    book.timestamp = datetime.now(timezone.utc)
                    
                    self.last_prices[market_id] = book.mid_price
                    
            except Exception as e:
                logger.error(f"Price update error: {e}")
                
            time.sleep(1)  # Update every second
            
    async def _simulate_latency(self):
        """Simulate network latency."""
        latency_ms = np.random.randint(self.latency_range[0], self.latency_range[1])
        await asyncio.sleep(latency_ms / 1000)
        
    def shutdown(self):
        """Shutdown the exchange."""
        self.running = False
        

async def run_test_scenario():
    """Run a test scenario with the mock exchange."""
    # Create exchange
    exchange = MockExchange(
        initial_markets=['NFL_GAME_1', 'NBA_GAME_1', 'EPL_GAME_1']
    )
    
    # Connect
    account = await exchange.connect('test_account', 10000)
    print(f"Connected: {account}")
    
    # Get markets
    markets = await exchange.get_markets()
    print(f"\nAvailable markets: {len(markets)}")
    for market in markets:
        print(f"  {market['market_id']}: {market['bid']:.2f}/{market['ask']:.2f}")
        
    # Place some orders
    print("\nPlacing orders...")
    
    # Market order
    order1 = await exchange.place_order(
        'test_account',
        'NFL_GAME_1',
        'buy',
        'market',
        100
    )
    print(f"Market order: {order1.status}, filled: {order1.filled_size} @ {order1.avg_fill_price:.3f}")
    
    # Limit order
    book = await exchange.get_order_book('NBA_GAME_1')
    limit_price = book['mid_price'] - 0.05
    
    order2 = await exchange.place_order(
        'test_account',
        'NBA_GAME_1',
        'buy',
        'limit',
        150,
        limit_price
    )
    print(f"Limit order: {order2.status}, filled: {order2.filled_size}")
    
    # Check balance
    balance = await exchange.get_balance('test_account')
    print(f"\nFinal balance: ${balance:.2f}")
    
    # Shutdown
    exchange.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_test_scenario())