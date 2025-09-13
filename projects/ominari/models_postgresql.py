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


class LookupBookmaker(Base):
    __tablename__ = "lu_bookmakers"
    __table_args__ = {"schema": "ominari"}

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(String(255))
    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<Bookmaker({self.name})>"


# =================== MAIN DATA TABLES ===================

class MarketNormalized(Base):
    __tablename__ = "markets_normalized"
    __table_args__ = (
        Index('idx_markets_sport_date', 'sport_id', 'start_time'),
        Index('idx_markets_teams', 'home_team_id', 'away_team_id'),
        Index('idx_markets_external_id', 'external_id'),
        {"schema": "ominari"}
    )

    id = Column(BigInteger, primary_key=True)
    external_id = Column(String(100), unique=True, nullable=False)

    # Foreign keys to lookup tables (matching actual schema)
    source_id = Column(SmallInteger, ForeignKey("ominari.lu_sources.id"), nullable=False)
    sport_id = Column(SmallInteger, ForeignKey("ominari.lu_sports.id"))
    home_team_id = Column(Integer, ForeignKey("ominari.lu_teams.id"))
    away_team_id = Column(Integer, ForeignKey("ominari.lu_teams.id"))
    market_type_id = Column(SmallInteger, ForeignKey("ominari.lu_market_types.id"))

    # Date/time fields
    start_time = Column(DateTime(timezone=True))
    maturity_date = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    # Status fields (matching actual schema)
    is_finished = Column(Boolean, default=False)
    winning_outcome_id = Column(SmallInteger, ForeignKey("ominari.lu_outcomes.id"))

    # Metadata field (renamed to avoid SQLAlchemy conflict)
    market_metadata = Column("metadata", Text)  # jsonb in PostgreSQL, but Text for SQLAlchemy compatibility

    # Relationships
    source = relationship("LookupSource", backref="markets")
    sport = relationship("LookupSport", backref="markets")
    home_team = relationship("LookupTeam", foreign_keys=[home_team_id], backref="home_markets")
    away_team = relationship("LookupTeam", foreign_keys=[away_team_id], backref="away_markets")
    market_type = relationship("LookupMarketType", backref="markets")

    # Hybrid properties for backward compatibility
    @hybrid_property
    def source_id_str(self):
        return self.external_id

    @hybrid_property
    def sport_name(self):
        return self.sport.name if self.sport else None

    @hybrid_property
    def home_team_name(self):
        return self.home_team.name if self.home_team else None

    @hybrid_property
    def away_team_name(self):
        return self.away_team.name if self.away_team else None

    def __repr__(self):
        return f"<Market({self.external_id}: {self.sport_name} - {self.home_team_name} vs {self.away_team_name})>"


class OddsNormalized(Base):
    __tablename__ = "odds_normalized"
    __table_args__ = (
        Index('idx_odds_market_outcome', 'market_id', 'outcome_id'),
        Index('idx_odds_bookmaker', 'bookmaker_id'),
        Index('idx_odds_timestamp', 'updated_at_ts'),
        {"schema": "ominari"}
    )

    id = Column(BigInteger, primary_key=True)

    # Foreign keys
    market_id = Column(BigInteger, ForeignKey("ominari.markets_normalized.id"), nullable=False)
    outcome_id = Column(Integer, ForeignKey("ominari.lu_outcomes.id"), nullable=False)
    bookmaker_id = Column(Integer, ForeignKey("ominari.lu_bookmakers.id"), nullable=False)

    # Optimized odds storage (integers for performance)
    decimal_odds_x1000 = Column(Integer)  # Store 1.50 as 1500
    american_odds = Column(SmallInteger)  # -110, +150, etc.
    implied_prob_x10000 = Column(Integer)  # Store 0.6667 as 6667

    # Timestamp as Unix timestamp for performance
    updated_at_ts = Column(BigInteger, nullable=False)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    market = relationship("MarketNormalized", backref="odds")
    outcome = relationship("LookupOutcome", backref="odds")
    bookmaker = relationship("LookupBookmaker", backref="odds")

    # Hybrid properties for easy access to decimal values
    @hybrid_property
    def decimal_odds(self):
        return self.decimal_odds_x1000 / 1000.0 if self.decimal_odds_x1000 else None

    @decimal_odds.setter
    def decimal_odds(self, value):
        self.decimal_odds_x1000 = int(value * 1000) if value else None

    @hybrid_property
    def implied_probability(self):
        return self.implied_prob_x10000 / 10000.0 if self.implied_prob_x10000 else None

    @implied_probability.setter
    def implied_probability(self, value):
        self.implied_prob_x10000 = int(value * 10000) if value else None

    @hybrid_property
    def updated_at(self):
        return datetime.fromtimestamp(self.updated_at_ts) if self.updated_at_ts else None

    @updated_at.setter
    def updated_at(self, value):
        self.updated_at_ts = int(value.timestamp()) if value else None

    def __repr__(self):
        return f"<Odds({self.market.external_id if self.market else 'N/A'}: {self.outcome.name if self.outcome else 'N/A'} @ {self.decimal_odds})>"


# =================== LEGACY COMPATIBILITY ===================

# Backward compatibility aliases
Market = MarketNormalized
Odd = OddsNormalized

# Legacy table for betting sessions (if needed)
class BettingSession(Base):
    __tablename__ = "betting_sessions"
    __table_args__ = {"schema": "ominari"}

    id = Column(BigInteger, primary_key=True)
    session_name = Column(String(255), nullable=False)
    start_time = Column(DateTime, default=func.now())
    end_time = Column(DateTime)
    total_bets = Column(Integer, default=0)
    total_stake = Column(Float, default=0.0)
    total_profit = Column(Float, default=0.0)
    roi_percent = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<BettingSession({self.session_name}: {self.total_bets} bets, ROI: {self.roi_percent:.2f}%)>"


class Bet(Base):
    __tablename__ = "bets"
    __table_args__ = {"schema": "ominari"}

    id = Column(BigInteger, primary_key=True)
    session_id = Column(BigInteger, ForeignKey("ominari.betting_sessions.id"), nullable=False)
    market_id = Column(BigInteger, ForeignKey("ominari.markets_normalized.id"), nullable=False)
    outcome_id = Column(Integer, ForeignKey("ominari.lu_outcomes.id"), nullable=False)

    # Bet details
    stake = Column(Float, nullable=False)
    odds = Column(Float, nullable=False)
    potential_payout = Column(Float, nullable=False)

    # Result
    is_winner = Column(Boolean)
    actual_payout = Column(Float, default=0.0)
    profit_loss = Column(Float, default=0.0)

    # Timestamps
    placed_at = Column(DateTime, default=func.now())
    settled_at = Column(DateTime)

    # Relationships
    session = relationship("BettingSession", backref="bets")
    market = relationship("MarketNormalized", backref="bets")
    outcome = relationship("LookupOutcome", backref="bets")

    def __repr__(self):
        return f"<Bet({self.market.external_id if self.market else 'N/A'}: {self.stake} @ {self.odds})>"


# =================== BLOCKCHAIN INTEGRATION TABLES ===================

class BlockchainMarket(Base):
    __tablename__ = "blockchain_markets"
    __table_args__ = {"schema": "ominari"}

    id = Column(BigInteger, primary_key=True)
    market_id = Column(BigInteger, ForeignKey("ominari.markets_normalized.id"))

    # Blockchain specific fields
    chain_id = Column(Integer, nullable=False)  # 10=Optimism, 42161=Arbitrum, etc.
    contract_address = Column(String(42), nullable=False)  # Ethereum address
    market_index = Column(BigInteger, nullable=False)

    # Market state
    total_volume = Column(BigInteger, default=0)  # Wei amount
    total_fees = Column(BigInteger, default=0)    # Wei amount
    block_number = Column(BigInteger)
    block_timestamp = Column(BigInteger)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationship
    market = relationship("MarketNormalized", backref="blockchain_data")

    # Index for efficient blockchain queries
    __table_args__ = (
        Index('idx_blockchain_chain_contract', 'chain_id', 'contract_address'),
        Index('idx_blockchain_market_index', 'market_index'),
        {"schema": "ominari"}
    )

    def __repr__(self):
        return f"<BlockchainMarket(chain={self.chain_id}, contract={self.contract_address[:10]}..., idx={self.market_index})>"


# =================== UTILITY FUNCTIONS ===================

def get_or_create_lookup(session, model_class, name, **kwargs):
    """Get or create a lookup table entry."""
    instance = session.query(model_class).filter(model_class.name == name).first()
    if not instance:
        instance = model_class(name=name, **kwargs)
        session.add(instance)
        session.commit()
        session.refresh(instance)
    return instance


def create_all_tables(engine):
    """Create all tables in PostgreSQL."""
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    """Test the models with PostgreSQL."""
    from database import engine, SessionLocal
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print("Testing PostgreSQL models...")

    try:
        # Test database connection
        with engine.connect() as conn:
            print("✅ Database connection successful")

        # Test session creation
        session = SessionLocal()

        # Query some data
        sport_count = session.query(LookupSport).count()
        team_count = session.query(LookupTeam).count()
        market_count = session.query(MarketNormalized).count()

        print(f"📊 Current data:")
        print(f"  Sports: {sport_count}")
        print(f"  Teams: {team_count}")
        print(f"  Markets: {market_count}")

        # Test a join query
        if market_count > 0:
            sample_market = (session.query(MarketNormalized)
                           .join(LookupSport)
                           .join(LookupTeam, MarketNormalized.home_team_id == LookupTeam.id)
                           .first())

            if sample_market:
                print(f"\n📝 Sample market:")
                print(f"  {sample_market.external_id}")
                print(f"  Sport: {sample_market.sport_name}")
                print(f"  Teams: {sample_market.home_team_name} vs {sample_market.away_team_name}")

        session.close()
        print("\n✅ PostgreSQL models working correctly!")

    except Exception as e:
        logger.error(f"Model test failed: {e}")
        print("❌ PostgreSQL models test failed!")