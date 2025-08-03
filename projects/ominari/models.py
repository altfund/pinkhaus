#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 17:28:16 2025

@author: ess
"""

from sqlalchemy import (
    Column, String, Boolean, Integer,
    Float, Text, DateTime, ForeignKey, UniqueConstraint, Index, Enum, func, case
)
from sqlalchemy.ext.declarative import declarative_base
#from sqlalchemy.sql import func
from sqlalchemy.ext.hybrid    import hybrid_property
from typing import Optional


Base = declarative_base()

class TableMetadata(Base):
    __tablename__ = "table_metadata"
    table_name  = Column(String, primary_key=True)
    last_updated= Column(DateTime)

class Market(Base):
    __tablename__ = "market"
    source_id            = Column(String, primary_key=True)
    source               = Column(String)
    sport                = Column(String)
    league_name          = Column(String)
    market_type          = Column(String)
    home_team            = Column(String)
    away_team            = Column(String)
    
    # game fields
    game_status          = Column(String)
    is_finished          = Column(Boolean)
    tournament           = Column(String)
    tournament_round     = Column(String)
    home_score           = Column(Integer)
    away_score           = Column(Integer)
    home_score_by_period = Column(Text)   # JSON array
    away_score_by_period = Column(Text)   # JSON array
    start_time           = Column(DateTime)
    last_update          = Column(DateTime)
    
    position_names       = Column(String)
    
    maturity_date        = Column(DateTime)
    updated_at           = Column(DateTime, server_default=func.current_timestamp())
    
    @hybrid_property
    def resolved_outcome(self) -> Optional[str]:
        if not self.is_finished:
            return None
        if self.home_score > self.away_score:
            return "Home"
        if self.home_score < self.away_score:
            return "Away"
        return "Draw"

    @resolved_outcome.expression
    def resolved_outcome(cls):
        return case(
            # whens as positional tuples, not a list
            (cls.is_finished == False,        None),
            (cls.home_score > cls.away_score, "Home"),
            (cls.home_score < cls.away_score, "Away"),
            else_="Draw"
        )

    
    __table_args__ = (
        # speed up: filter on (source, sport, market_type, maturity_date)
        Index(
            "idx_market_performance",
            "source", "sport", "market_type", "maturity_date"
        ),
    )    

    # FK to game would go here if you still had a `game` table:
    # game_id             = Column(String, ForeignKey("game.game_id"))

class Match(Base):
    __tablename__ = "match"
    match_id            = Column(Integer, primary_key=True, autoincrement=True)
    overtime_source_id  = Column(String, ForeignKey("market.source_id"), nullable=False)
    odds_api_source_id  = Column(String, ForeignKey("market.source_id"), nullable=False)
    confidence          = Column(Float)
    updated_at          = Column(DateTime, server_default=func.current_timestamp())
    __table_args__ = (
        UniqueConstraint("overtime_source_id", "odds_api_source_id"),
    )

class Odd(Base):
    __tablename__ = "odd"
    id                   = Column(Integer, primary_key=True, autoincrement=True)
    source_id            = Column(String, ForeignKey("market.source_id"), nullable=False)
    position             = Column(Integer) #, nullable=False, server_default=-1)   # ← new
    market_type          = Column(String, nullable=False)
    line                 = Column(Float)
    outcome              = Column(String, nullable=False)
    source               = Column(String, nullable=False)
    bookmaker            = Column(String, nullable=False)
    american_odds        = Column(Float)
    decimal_odds         = Column(Float)
    normalized_implied   = Column(Float)
    updated_at           = Column(DateTime, server_default=func.current_timestamp())
    
    #__table_args__ = (
        # prevent inserting the same position twice
    #    UniqueConstraint("source_id", "position", name="uq_odd_source_position"),
    #)
    __table_args__ = (
        # speed up: filter on (bookmaker, market_type, source_id, updated_at)
        Index(
            "idx_odd_performance",
            "bookmaker", "market_type", "source_id", "updated_at"
        ),
    )

class Team(Base):
    __tablename__ = "team"
    team_name   = Column(String, primary_key=True)
    league      = Column(String)
    updated_at  = Column(DateTime, server_default=func.current_timestamp())
    
# define your allowed session types:
SESSION_TYPES = ("backtest", "paper", "live", "simulation")

class BettingSession(Base):
    __tablename__ = "betting_session"

    id                         = Column(Integer, primary_key=True, autoincrement=True)
    # the exact market‐snapshot time you pass as `as_of`
    as_of                      = Column(DateTime, nullable=False, index=True)
    # when this row was actually written
    created_at                 = Column(DateTime, server_default=func.current_timestamp())

    # context tags
    session_type               = Column(
                                    Enum(*SESSION_TYPES, name="session_type_enum"),
                                    nullable=False,
                                    index=True,
                                    default="simulation",
                                )
    strategy_name              = Column(String, nullable=True, index=True)

    # ─── your core parameters (from the function signature) ─────────────────────
    kelly_bankroll             = Column(Float, nullable=False)  # e.g. 1.0                  :contentReference[oaicite:19]{index=19}
    execution_bankroll         = Column(Float, nullable=False)  # e.g. 15.0                 :contentReference[oaicite:20]{index=20}
    kelly_fraction             = Column(Float, nullable=False)  # e.g. 0.25                 :contentReference[oaicite:21]{index=21}
    cap_per_game               = Column(Float, nullable=False)  # e.g. 0.25                 :contentReference[oaicite:22]{index=22}
    cap_per_bet                = Column(Float, nullable=False)  # e.g. 0.25                 :contentReference[oaicite:23]{index=23}
    cap_per_game_market        = Column(Float, nullable=False)  # e.g. 0.10                 :contentReference[oaicite:24]{index=24}
    min_bet_abs                = Column(Float, nullable=False)  # e.g. 0.5                  :contentReference[oaicite:25]{index=25}
    min_bet_pct                = Column(Float, nullable=False)  # e.g. 0.0025               :contentReference[oaicite:26]{index=26}
    abs_game_limit             = Column(Integer, nullable=True)  # may be None             :contentReference[oaicite:27]{index=27}
    min_break_minutes          = Column(Float, nullable=False)  # e.g. 360.0               :contentReference[oaicite:28]{index=28}
    avg_game_duration_minutes  = Column(Float, nullable=False)  # e.g. 180.0               :contentReference[oaicite:29]{index=29}

    # store your signal‐blending config as JSON text
    signal_weights             = Column(Text, nullable=True)    # e.g. '{"implied_raw":1.0}' :contentReference[oaicite:30]{index=30}

    # ─── the main metrics you already calculate in the session ─────────────────
    num_open_markets           = Column(Integer, nullable=True)  # len(open_markets)       :contentReference[oaicite:31]{index=31}
    num_games                  = Column(Integer, nullable=True)  # number of games in chunk :contentReference[oaicite:32]{index=32}
    num_bets_recommended       = Column(Integer, nullable=True)  # bets_to_place_df.shape[0]
    expected_return            = Column(Float, nullable=True)    # your calculate_expected_kelly_return :contentReference[oaicite:33]{index=33}
    normalized_return          = Column(Float, nullable=True)    # expected/total stake    :contentReference[oaicite:34]{index=34}
    total_original_stake       = Column(Float, nullable=True)    # sum(original_stake)     :contentReference[oaicite:35]{index=35}
    total_trimmed_stake        = Column(Float, nullable=True)    # sum(final stake)
    total_fee_amount           = Column(Float, nullable=True)    # df["fee_amount"].sum()  

    # add any further indices or constraints as you see fit…


class Bet(Base):
    __tablename__ = "bet"
    id                     = Column(Integer, primary_key=True, autoincrement=True)
    session_id             = Column(Integer, ForeignKey("betting_session.id"), nullable=False, index=True)
    source_id              = Column(String,  ForeignKey("market.source_id"), nullable=False)
    unified_market_type    = Column(String,  nullable=False)
    normalized_outcome     = Column(String,  nullable=False)
    normalized_line        = Column(Float,   nullable=False)
    bet_name               = Column(String,  nullable=True)
    probability            = Column(Float,   nullable=True)
    odds                   = Column(Float,   nullable=True)
    stake                  = Column(Float,   nullable=True)
    execution_stake        = Column(Float,   nullable=True)
    fee_amount             = Column(Float,   nullable=True)
    fee_pct                = Column(Float,   nullable=True)
    created_at             = Column(DateTime, server_default=func.current_timestamp())

    __table_args__ = (
        Index("idx_bet_on_market", "source_id", "unified_market_type"),
    )
