#!/usr/bin/env python3
"""Optimized SQLAlchemy models for paper trading with reduced storage."""

from sqlalchemy import (
    Column, String, Float, DateTime, ForeignKey, Integer, BigInteger,
    SmallInteger, Enum, JSON, Index, Numeric, Boolean
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from models import Base
import enum


class PositionStatus(enum.Enum):
    """Position status enum."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class SessionStatus(enum.Enum):
    """Session status enum."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class TradeType(enum.Enum):
    """Trade type enum."""
    OPEN = "open"
    REBALANCE = "rebalance"
    CLOSE = "close"


class Outcome(enum.Enum):
    """Market outcome enum."""
    HOME = "Home"
    DRAW = "Draw"
    AWAY = "Away"


class Result(enum.Enum):
    """Position result enum."""
    WON = "won"
    LOST = "lost"
    VOID = "void"


class PaperTradingSession(Base):
    """Paper trading session - top level container."""
    __tablename__ = "paper_trading_sessions"
    
    session_id = Column(String(20), primary_key=True)  # e.g. '20250907_135452'
    session_name = Column(String(100))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    initial_bankroll = Column(Numeric(12, 2), nullable=False)
    status = Column(Enum(SessionStatus), nullable=False, default=SessionStatus.ACTIVE)
    strategy_config = Column(JSON)  # Flexible config storage
    
    # Relationships
    positions = relationship("PaperTradingPosition", back_populates="session")
    trades = relationship("PaperTradingTrade", back_populates="session")
    snapshots = relationship("PaperTradingSnapshot", back_populates="session")
    
    __table_args__ = (
        Index('idx_pts_created_at', 'created_at'),
        Index('idx_pts_status', 'status'),
    )


class PaperTradingPosition(Base):
    """Paper trading position - represents a bet on an outcome."""
    __tablename__ = "paper_trading_positions"
    
    position_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(20), ForeignKey("paper_trading_sessions.session_id"), nullable=False)
    market_id = Column(String(68), nullable=False)  # Hex market ID
    outcome = Column(Enum(Outcome), nullable=False)
    
    # Position details
    stake = Column(Numeric(10, 2), nullable=False)
    execution_stake = Column(Numeric(10, 2), nullable=False)  # stake + fees
    avg_odds = Column(Numeric(6, 3), nullable=False)
    
    # Fees stored as basis points (100 = 1%)
    safebox_fee_bps = Column(SmallInteger, nullable=False, default=200)  # 2%
    skew_fee_bps = Column(SmallInteger, nullable=False, default=100)     # 1%
    
    # Status and timing
    status = Column(Enum(PositionStatus), nullable=False, default=PositionStatus.OPEN)
    opened_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    closed_at = Column(DateTime(timezone=True))
    maturity_date = Column(DateTime(timezone=True), nullable=False)
    
    # Results (NULL for open positions)
    final_value = Column(Numeric(10, 2))
    pnl = Column(Numeric(10, 2))
    result = Column(Enum(Result))
    resolved_outcome = Column(Enum(Outcome))
    
    # Relationships
    session = relationship("PaperTradingSession", back_populates="positions")
    trades = relationship("PaperTradingTrade", back_populates="position")
    
    __table_args__ = (
        Index('idx_ptp_session_market', 'session_id', 'market_id', 'outcome'),
        Index('idx_ptp_status_session', 'status', 'session_id'),
        Index('idx_ptp_maturity', 'maturity_date'),
    )
    
    @property
    def safebox_fee_pct(self):
        """Get safebox fee as percentage."""
        return self.safebox_fee_bps / 10000.0
    
    @property
    def skew_fee_pct(self):
        """Get skew fee as percentage."""
        return self.skew_fee_bps / 10000.0
    
    @property
    def total_fee_pct(self):
        """Get total fee as percentage."""
        return (self.safebox_fee_bps + self.skew_fee_bps) / 10000.0


class PaperTradingTrade(Base):
    """Individual trade action on a position."""
    __tablename__ = "paper_trading_trades"
    
    trade_id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(Integer, ForeignKey("paper_trading_positions.position_id"), nullable=False)
    session_id = Column(String(20), ForeignKey("paper_trading_sessions.session_id"), nullable=False)
    
    # Trade details
    trade_type = Column(Enum(TradeType), nullable=False)
    stake_delta = Column(Numeric(10, 2), nullable=False)  # Can be negative
    odds = Column(Numeric(6, 3), nullable=False)
    fee_amount = Column(Numeric(8, 2), nullable=False, default=0)
    
    # Timing
    traded_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    
    # Relationships
    position = relationship("PaperTradingPosition", back_populates="trades")
    session = relationship("PaperTradingSession", back_populates="trades")
    
    __table_args__ = (
        Index('idx_ptt_session_time', 'session_id', 'traded_at'),
    )


class PaperTradingSnapshot(Base):
    """Performance snapshot at a point in time."""
    __tablename__ = "paper_trading_snapshots"
    
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(20), ForeignKey("paper_trading_sessions.session_id"), nullable=False)
    snapshot_time = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    
    # Portfolio values
    cash_balance = Column(Numeric(12, 2), nullable=False)
    positions_value = Column(Numeric(12, 2), nullable=False)
    portfolio_value = Column(Numeric(12, 2), nullable=False)
    
    # Performance metrics
    total_pnl = Column(Numeric(10, 2), nullable=False)
    daily_pnl = Column(Numeric(10, 2))
    win_count = Column(SmallInteger, nullable=False, default=0)
    loss_count = Column(SmallInteger, nullable=False, default=0)
    pending_count = Column(SmallInteger, nullable=False, default=0)
    
    # Risk metrics as percentages
    max_drawdown = Column(Numeric(5, 2))  # e.g. 5.25 for 5.25%
    sharpe_ratio = Column(Numeric(5, 2))
    
    # Relationships
    session = relationship("PaperTradingSession", back_populates="snapshots")
    
    __table_args__ = (
        Index('idx_pts_session_time', 'session_id', 'snapshot_time'),
    )


class MarketName(Base):
    """Normalized market names to save space."""
    __tablename__ = "market_names"
    
    market_id = Column(String(68), primary_key=True)
    market_name = Column(String(200), nullable=False)
    home_team = Column(String(100))
    away_team = Column(String(100))
    
    __table_args__ = (
        Index('idx_mn_market_name', 'market_name'),
    )