#!/usr/bin/env python3
"""SQLAlchemy models for paper trading tables."""

from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Integer
from sqlalchemy.sql import func
from models import Base

class PaperOrder(Base):
    """Paper trading order record."""
    __tablename__ = "paper_orders"
    
    order_id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    source_id = Column(String, nullable=False)  # market_id
    market_type = Column(String)
    bet_name = Column(String)
    side = Column(String)  # 'buy' or 'sell'
    size = Column(Float)
    limit_price = Column(Float)
    signal_name = Column(String)
    expected_edge = Column(Float)
    status = Column(String)  # 'pending', 'filled', 'cancelled'
    created_at = Column(DateTime, server_default=func.current_timestamp())


class PaperFill(Base):
    """Paper trading fill/execution record."""
    __tablename__ = "paper_fills"
    
    fill_id = Column(String, primary_key=True)
    order_id = Column(String, ForeignKey("paper_orders.order_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    fill_price = Column(Float)
    fill_size = Column(Float)
    slippage = Column(Float)
    commission = Column(Float)
    market_impact = Column(Float)
    pnl = Column(Float)
    created_at = Column(DateTime, server_default=func.current_timestamp())


class PaperPerformance(Base):
    """Paper trading performance snapshot."""
    __tablename__ = "paper_performance"
    
    timestamp = Column(DateTime, primary_key=True)
    capital = Column(Float)
    positions_value = Column(Float)
    total_value = Column(Float)
    daily_pnl = Column(Float)
    total_pnl = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    created_at = Column(DateTime, server_default=func.current_timestamp())