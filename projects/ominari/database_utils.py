#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database utility functions using SQLAlchemy ORM for optimized queries.
Provides chunked and paginated query patterns to handle large datasets.
"""

import pandas as pd
from sqlalchemy.orm import Session, Query
from sqlalchemy import func, and_
from typing import Optional, List, Dict, Any, Generator, Tuple
from datetime import datetime, timezone
import logging
from models import Market, Odd, Base
from database import SessionLocal

logger = logging.getLogger(__name__)

# Default chunk size for batch operations
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_PAGE_SIZE = 5000


def chunked_query(
    query: Query, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> Generator[List, None, None]:
    """
    Execute a query in chunks to avoid memory issues with large result sets.

    Args:
        query: SQLAlchemy query object
        chunk_size: Number of records to fetch per chunk

    Yields:
        Lists of records, chunk by chunk
    """
    offset = 0
    while True:
        chunk = query.limit(chunk_size).offset(offset).all()
        if not chunk:
            break
        yield chunk
        offset += chunk_size

        # Allow other operations between chunks
        if offset % (chunk_size * 10) == 0:
            logger.info(f"Processed {offset} records...")


def fetch_open_markets_optimized(
    as_of: pd.Timestamp,
    session: Optional[Session] = None,
    chunk_size: int = DEFAULT_PAGE_SIZE,
) -> pd.DataFrame:
    """
    Fetch open markets using optimized ORM queries with chunking.
    Replaces the raw SQL query in evaluate_open_markets.py
    """
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize(timezone.utc)

    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True

    try:
        # First, get the latest update time for each market before as_of
        # Using a subquery for better performance
        latest_odds_subq = (
            session.query(Odd.source_id, func.max(Odd.updated_at).label("max_updated"))
            .filter(Odd.updated_at <= as_of)
            .group_by(Odd.source_id)
            .subquery()
        )

        # Main query joining odds with their latest timestamps and markets
        query = (
            session.query(
                Odd.source_id,
                Odd.market_type.label("odds_market_type"),
                Odd.outcome.label("bet_name"),
                Odd.decimal_odds,
                Odd.updated_at.label("as_of_time"),
                Market,
            )
            .join(
                latest_odds_subq,
                and_(
                    Odd.source_id == latest_odds_subq.c.source_id,
                    Odd.updated_at == latest_odds_subq.c.max_updated,
                ),
            )
            .outerjoin(Market, Market.source_id == Odd.source_id)
        )

        # Process in chunks to avoid memory issues
        all_records = []
        for chunk in chunked_query(query, chunk_size):
            records = []
            for row in chunk:
                # Handle tuple unpacking more carefully
                if isinstance(row, tuple):
                    odd = row[0]
                    market = row[1] if len(row) > 1 else None
                else:
                    odd = row
                    market = None
                    
                record = {
                    "source_id": odd.source_id,
                    "odds_market_type": odd.odds_market_type,
                    "bet_name": odd.bet_name,
                    "decimal_odds": odd.decimal_odds,
                    "as_of_time": odd.as_of_time,
                }

                # Add market attributes if available
                if market:
                    for col in Market.__table__.columns:
                        if col.name != "source_id":  # Avoid duplicate
                            value = getattr(market, col.name)
                            # Handle empty datetime strings
                            if col.type.__class__.__name__ == 'DateTime' and value == '':
                                value = None
                            record[col.name] = value

                records.append(record)

            all_records.extend(records)

        df = pd.DataFrame(all_records)
        if not df.empty and "as_of_time" in df.columns:
            df["as_of_time"] = pd.to_datetime(df["as_of_time"], utc=True)

        return df

    finally:
        if close_session:
            session.close()


def fetch_odds_window_optimized(
    sport: str = "Soccer",
    market_type: str = "winner",
    bookmaker: str = "overtime_markets",
    session: Optional[Session] = None,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Get the earliest and latest timestamps for odds data.
    Replaces _load_odds_window in backtest.py
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True

    try:
        # Query for odds window
        odds_result = (
            session.query(
                func.min(Odd.updated_at).label("earliest_odds"),
                func.max(Odd.updated_at).label("latest_odds"),
            )
            .join(Market, Market.source_id == Odd.source_id)
            .filter(
                Odd.bookmaker == bookmaker,
                Odd.market_type == market_type,
                Market.sport == sport,
            )
            .first()
        )

        # Query for maturity dates
        maturity_result = (
            session.query(
                func.min(Market.maturity_date).label("earliest_mat"),
                func.max(Market.maturity_date).label("latest_mat"),
            )
            .filter(Market.sport == sport)
            .first()
        )

        def to_utc(ts) -> Optional[pd.Timestamp]:
            if ts is None:
                return None
            ts = pd.to_datetime(ts)
            if ts.tzinfo is None:
                return ts.tz_localize("UTC")
            return ts.tz_convert("UTC")

        # Combine results
        start_candidates = [
            ts
            for ts in [
                to_utc(odds_result.earliest_odds) if odds_result else None,
                to_utc(maturity_result.earliest_mat) if maturity_result else None,
            ]
            if ts is not None
        ]

        end_candidates = [
            ts
            for ts in [
                to_utc(odds_result.latest_odds) if odds_result else None,
                to_utc(maturity_result.latest_mat) if maturity_result else None,
            ]
            if ts is not None
        ]

        if not start_candidates or not end_candidates:
            return None, None

        return min(start_candidates), max(end_candidates)

    finally:
        if close_session:
            session.close()


def fetch_markets_in_window(
    start_dt: datetime,
    end_dt: datetime,
    session: Optional[Session] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Fetch markets within a time window using ORM.
    Replaces _load_markets in backtest.py
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True

    try:
        query = session.query(
            Market.maturity_date, Market.home_team, Market.away_team
        ).filter(Market.maturity_date.between(start_dt, end_dt))

        all_records = []
        for chunk in chunked_query(query, chunk_size):
            records = [
                {
                    "maturity_date": market.maturity_date,
                    "home_team": market.home_team,
                    "away_team": market.away_team,
                }
                for market in chunk
            ]
            all_records.extend(records)

        df = pd.DataFrame(all_records)
        if not df.empty:
            df["maturity_date"] = pd.to_datetime(df["maturity_date"], utc=True)

        return df

    finally:
        if close_session:
            session.close()


def batch_insert_records(
    records: List[Dict[str, Any]],
    model_class: Base,
    session: Optional[Session] = None,
    batch_size: int = 1000,
    commit_frequency: int = 10000,
) -> None:
    """
    Efficiently insert multiple records in batches.

    Args:
        records: List of dictionaries containing record data
        model_class: SQLAlchemy model class (e.g., Market, Odd)
        session: Database session (creates new if None)
        batch_size: Number of records to insert per batch
        commit_frequency: Commit after this many records
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True

    try:
        total_inserted = 0

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]

            # Create model instances
            instances = [model_class(**record) for record in batch]

            # Bulk insert
            session.bulk_save_objects(instances)

            total_inserted += len(batch)

            # Periodic commit to avoid holding too much in memory
            if total_inserted % commit_frequency == 0:
                session.commit()
                logger.info(f"Committed {total_inserted} records")

        # Final commit
        session.commit()
        logger.info(f"Successfully inserted {total_inserted} records")

    except Exception as e:
        session.rollback()
        logger.error(f"Error in batch insert: {e}")
        raise
    finally:
        if close_session:
            session.close()


def upsert_records_orm(
    records: List[Dict[str, Any]],
    model_class: Base,
    unique_keys: List[str],
    session: Optional[Session] = None,
    batch_size: int = 500,
) -> None:
    """
    Upsert (insert or update) records using ORM.
    More efficient than the raw SQL approach for smaller batches.

    Args:
        records: List of dictionaries containing record data
        model_class: SQLAlchemy model class
        unique_keys: List of column names that form the unique constraint
        session: Database session
        batch_size: Number of records to process per batch
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True

    try:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]

            for record in batch:
                # Add updated_at timestamp
                record["updated_at"] = datetime.now(timezone.utc)

                # Build filter conditions for unique keys
                filter_conditions = [
                    getattr(model_class, key) == record[key] for key in unique_keys
                ]

                # Check if record exists
                existing = session.query(model_class).filter(*filter_conditions).first()

                if existing:
                    # Update existing record
                    for key, value in record.items():
                        setattr(existing, key, value)
                else:
                    # Insert new record
                    instance = model_class(**record)
                    session.add(instance)

            # Commit batch
            session.commit()

            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"Processed {i + batch_size} records")

    except Exception as e:
        session.rollback()
        logger.error(f"Error in upsert: {e}")
        raise
    finally:
        if close_session:
            session.close()


def get_latest_odds_for_markets(
    market_ids: List[str],
    as_of: Optional[datetime] = None,
    session: Optional[Session] = None,
    chunk_size: int = 1000,
) -> pd.DataFrame:
    """
    Get the latest odds for a list of market IDs.
    Processes in chunks to handle large lists efficiently.
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True

    try:
        all_results = []

        # Process market IDs in chunks
        for i in range(0, len(market_ids), chunk_size):
            chunk_ids = market_ids[i : i + chunk_size]

            # Subquery for latest update time per market
            if as_of:
                latest_subq = (
                    session.query(
                        Odd.source_id, func.max(Odd.updated_at).label("max_updated")
                    )
                    .filter(Odd.source_id.in_(chunk_ids), Odd.updated_at <= as_of)
                    .group_by(Odd.source_id)
                    .subquery()
                )
            else:
                latest_subq = (
                    session.query(
                        Odd.source_id, func.max(Odd.updated_at).label("max_updated")
                    )
                    .filter(Odd.source_id.in_(chunk_ids))
                    .group_by(Odd.source_id)
                    .subquery()
                )

            # Main query
            chunk_results = (
                session.query(Odd)
                .join(
                    latest_subq,
                    and_(
                        Odd.source_id == latest_subq.c.source_id,
                        Odd.updated_at == latest_subq.c.max_updated,
                    ),
                )
                .all()
            )

            all_results.extend(chunk_results)

        # Convert to DataFrame
        if all_results:
            data = [
                {
                    "source_id": odd.source_id,
                    "market_type": odd.market_type,
                    "outcome": odd.outcome,
                    "decimal_odds": odd.decimal_odds,
                    "updated_at": odd.updated_at,
                }
                for odd in all_results
            ]
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()

    finally:
        if close_session:
            session.close()
