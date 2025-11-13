#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQL Normalized Models
Models for the normalized PostgreSQL schema with lookup tables and optimized storage.
"""

from sqlalchemy import (
    Column,
    String,
    Boolean,
    Integer,
    Float,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
    BigInteger,
    SmallInteger,
    func,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.hybrid import hybrid_property
from typing import Optional
from datetime import datetime

Base = declarative_base()

# Set schema for all tables
def set_schema():
    for table in Base.metadata.tables.values():
        table.schema = 'ominari'


# =================== LOOKUP TABLES ===================

class LookupSport(Base):
    __tablename__ = "lu_sports"
    __table_args__ = {"schema": "ominari"}

    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<Sport({self.name})>"


class LookupTeam(Base):
    __tablename__ = "lu_teams"
    __table_args__ = {"schema": "ominari"}

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    sport_id = Column(Integer, ForeignKey("ominari.lu_sports.id"), nullable=False)
    created_at = Column(DateTime, default=func.now())

    # Relationship
    sport = relationship("LookupSport", backref="teams")

    def __repr__(self):
        return f"<Team({self.name})>"


class LookupSource(Base):
    __tablename__ = "lu_sources"
    __table_args__ = {"schema": "ominari"}

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(String(255))
    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<Source({self.name})>"


class LookupMarketType(Base):
    __tablename__ = "lu_market_types"
    __table_args__ = {"schema": "ominari"}

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(String(255))
    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<MarketType({self.name})>"


class LookupOutcome(Base):
    __tablename__ = "lu_outcomes"
    __table_args__ = {"schema": "ominari"}

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(String(255))
    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<Outcome({self.name})>"


# =================== MAIN TABLES ===================

class MarketNormalized(Base):
    __tablename__ = "markets_normalized"
    __table_args__ = {"schema": "ominari"}

    id = Column(BigInteger, primary_key=True)
    external_id = Column(String(100), nullable=False)
    # FIXED: Changed from SmallInteger FK to String to handle blockchain hex IDs
    source_id = Column(String(66), nullable=False, index=True)  # e.g., "0x323032..."
    sport_id = Column(SmallInteger, ForeignKey("ominari.lu_sports.id"), nullable=False)
    home_team_id = Column(Integer, ForeignKey("ominari.lu_teams.id"), nullable=True)
    away_team_id = Column(Integer, ForeignKey("ominari.lu_teams.id"), nullable=True)
    market_type_id = Column(SmallInteger, ForeignKey("ominari.lu_market_types.id"), nullable=False)
    start_time = Column(DateTime, nullable=False)
    maturity_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    is_finished = Column(Boolean, default=False, nullable=False)
    winning_outcome_id = Column(SmallInteger, ForeignKey("ominari.lu_outcomes.id"), nullable=True)
    market_metadata = Column(Text, nullable=True)  # JSON data for additional fields

    # Relationships
    sport = relationship("LookupSport")
    home_team = relationship("LookupTeam", foreign_keys=[home_team_id])
    away_team = relationship("LookupTeam", foreign_keys=[away_team_id])
    market_type = relationship("LookupMarketType")
    winning_outcome = relationship("LookupOutcome")

    def __repr__(self):
        return f"<MarketNormalized(source_id={self.source_id[:10]}...)>"


class OddsNormalized(Base):
    __tablename__ = "odds_normalized"
    __table_args__ = {"schema": "ominari"}

    id = Column(BigInteger, primary_key=True)
    market_id = Column(BigInteger, ForeignKey("ominari.markets_normalized.id"), nullable=False)
    outcome_id = Column(SmallInteger, ForeignKey("ominari.lu_outcomes.id"), nullable=False)
    source_id = Column(Integer, ForeignKey("ominari.lu_sources.id"), nullable=False)
    odds_decimal = Column(Float, nullable=False)
    odds_american = Column(Integer, nullable=True)
    implied_probability = Column(Float, nullable=True)
    line = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    market = relationship("MarketNormalized")
    outcome = relationship("LookupOutcome")
    source = relationship("LookupSource")

    def __repr__(self):
        return f"<OddsNormalized(market_id={self.market_id}, odds={self.odds_decimal})>"


# Legacy models for backward compatibility
class Market(Base):
    __tablename__ = "market"

    source_id = Column(String(66), primary_key=True)  # Changed to handle blockchain hex IDs
    source = Column(String)
    sport = Column(String)
    league_name = Column(String)
    nation = Column(String)
    governing_body = Column(String)
    market_type = Column(String)
    home_team = Column(String)
    away_team = Column(String)

    # game fields
    game_status = Column(String)
    is_finished = Column(Boolean)
    tournament = Column(String)
    tournament_round = Column(String)
    home_score = Column(Integer)
    away_score = Column(Integer)
    home_score_by_period = Column(Text)  # JSON array
    away_score_by_period = Column(Text)  # JSON array
    start_time = Column(DateTime)
    last_update = Column(DateTime)

    position_names = Column(String)

    maturity_date = Column(DateTime)
    updated_at = Column(DateTime, server_default=func.current_timestamp())

    @hybrid_property
    def resolved_outcome(self) -> Optional[str]:
        if not self.is_finished:
            return None
        if self.home_score > self.away_score:
            return "Home"
        if self.home_score < self.away_score:
            return "Away"
        return "Draw"

    __table_args__ = (
        Index("idx_market_performance", "source", "sport", "market_type", "maturity_date"),
    )


class Odd(Base):
    __tablename__ = "odd"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String(66), ForeignKey("market.source_id"), nullable=False)  # Changed to String
    position = Column(Integer)
    market_type = Column(String, nullable=False)
    line = Column(Float)
    outcome = Column(String, nullable=False)
    source = Column(String, nullable=False)
    bookmaker = Column(String, nullable=False)
    american_odds = Column(Float)
    decimal_odds = Column(Float)
    normalized_implied = Column(Float)
    updated_at = Column(DateTime, server_default=func.current_timestamp())

    __table_args__ = (
        Index("idx_odd_performance", "bookmaker", "market_type", "source_id", "updated_at"),
    )


class Team(Base):
    __tablename__ = "team"
    team_name = Column(String, primary_key=True)
    league = Column(String)
    updated_at = Column(DateTime, server_default=func.current_timestamp())


class Match(Base):
    __tablename__ = "match"
    match_id = Column(Integer, primary_key=True, autoincrement=True)
    overtime_source_id = Column(String(66), ForeignKey("market.source_id"), nullable=False)  # Changed to String
    odds_api_source_id = Column(String(66), ForeignKey("market.source_id"), nullable=False)  # Changed to String
    confidence = Column(Float)
    updated_at = Column(DateTime, server_default=func.current_timestamp())
    __table_args__ = (UniqueConstraint("overtime_source_id", "odds_api_source_id"),)


# Betting session models
SESSION_TYPES = ("backtest", "paper", "live", "simulation")

class BettingSession(Base):
    __tablename__ = "betting_session"

    id = Column(Integer, primary_key=True, autoincrement=True)
    as_of = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.current_timestamp())

    # context tags
    session_type = Column(String, nullable=False, index=True, default="simulation")
    strategy_name = Column(String, nullable=True, index=True)

    # core parameters
    kelly_bankroll = Column(Float, nullable=False)
    execution_bankroll = Column(Float, nullable=False)
    kelly_fraction = Column(Float, nullable=False)
    cap_per_game = Column(Float, nullable=False)
    cap_per_bet = Column(Float, nullable=False)
    cap_per_game_market = Column(Float, nullable=False)
    min_bet_abs = Column(Float, nullable=False)
    min_bet_pct = Column(Float, nullable=False)
    abs_game_limit = Column(Integer, nullable=True)
    min_break_minutes = Column(Float, nullable=False)
    avg_game_duration_minutes = Column(Float, nullable=False)

    # signal-blending config
    signal_weights = Column(Text, nullable=True)

    # metrics
    num_open_markets = Column(Integer, nullable=True)
    num_games = Column(Integer, nullable=True)
    num_bets_recommended = Column(Integer, nullable=True)
    expected_return = Column(Float, nullable=True)
    normalized_return = Column(Float, nullable=True)
    total_original_stake = Column(Float, nullable=True)
    total_trimmed_stake = Column(Float, nullable=True)
    total_fee_amount = Column(Float, nullable=True)


class Bet(Base):
    __tablename__ = "bet"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("betting_session.id"), nullable=False, index=True)
    source_id = Column(String(66), ForeignKey("market.source_id"), nullable=False)  # Changed to String
    unified_market_type = Column(String, nullable=False)
    normalized_outcome = Column(String, nullable=False)
    normalized_line = Column(Float, nullable=False)
    bet_name = Column(String, nullable=True)
    probability = Column(Float, nullable=True)
    odds = Column(Float, nullable=True)
    stake = Column(Float, nullable=True)
    execution_stake = Column(Float, nullable=True)
    fee_amount = Column(Float, nullable=True)
    fee_pct = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.current_timestamp())

    __table_args__ = (Index("idx_bet_on_market", "source_id", "unified_market_type"),)