#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 17:28:16 2025

@author: ess
"""

from sqlalchemy import (
    Column, String, Boolean, Integer,
    Float, Text, DateTime, ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

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
