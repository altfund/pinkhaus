#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of the new ORM-based database utilities.
Shows how to use chunked queries to avoid database timeouts with large datasets.
"""

import pandas as pd
from database_utils import (
    fetch_open_markets_optimized,
    fetch_odds_window_optimized,
    get_latest_odds_for_markets,
    chunked_query,
)
from database import SessionLocal
from models import Market
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_fetch_open_markets():
    """Example: Fetch open markets at a specific time."""
    print("\n=== Example: Fetch Open Markets ===")

    # Get markets as of 1 hour ago
    as_of = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)

    # This uses chunked queries internally to avoid timeouts
    df = fetch_open_markets_optimized(as_of, chunk_size=5000)

    print(f"Found {len(df)} open markets as of {as_of}")
    if not df.empty:
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few markets:\n{df.head()}")


def example_get_odds_window():
    """Example: Get the time range of available odds data."""
    print("\n=== Example: Get Odds Window ===")

    # Get the time window for Soccer winner markets
    earliest, latest = fetch_odds_window_optimized(
        sport="Soccer", market_type="winner", bookmaker="overtime_markets"
    )

    if earliest and latest:
        print(f"Odds data available from {earliest} to {latest}")
        print(f"Duration: {latest - earliest}")
    else:
        print("No odds data found for the specified criteria")


def example_chunked_market_query():
    """Example: Process markets in chunks to avoid memory issues."""
    print("\n=== Example: Chunked Market Query ===")

    session = SessionLocal()
    try:
        # Query all markets, but process them in chunks
        query = session.query(Market).filter(Market.sport == "Soccer")

        total_markets = 0
        chunk_count = 0

        # Process in chunks of 1000
        for chunk in chunked_query(query, chunk_size=1000):
            chunk_count += 1
            total_markets += len(chunk)

            # Process each chunk (example: count by league)
            leagues = {}
            for market in chunk:
                league = market.league_name or "Unknown"
                leagues[league] = leagues.get(league, 0) + 1

            print(
                f"Chunk {chunk_count}: {len(chunk)} markets, leagues: {list(leagues.keys())[:3]}..."
            )

            # Break after a few chunks for demo purposes
            if chunk_count >= 3:
                print("(Stopping after 3 chunks for demo...)")
                break

        print(f"Processed {total_markets} markets in {chunk_count} chunks")

    finally:
        session.close()


def example_batch_upsert():
    """Example: Batch upsert with proper chunking."""
    print("\n=== Example: Batch Upsert (Demo Mode) ===")

    # Create some sample data
    sample_odds = [
        {
            "source_id": f"demo_market_{i}",
            "position": 0,
            "market_type": "winner",
            "outcome": "Home",
            "source": "demo",
            "bookmaker": "demo_bookmaker",
            "decimal_odds": 2.0 + (i % 10) * 0.1,
            "american_odds": 100,
            "normalized_implied": 0.5,
        }
        for i in range(10)  # Just 10 records for demo
    ]

    print(f"Demo: Would upsert {len(sample_odds)} odds records")
    print("(Not actually executing to avoid modifying your database)")

    # In real usage, you would call:
    # upsert_records_orm(
    #     records=sample_odds,
    #     model_class=Odd,
    #     unique_keys=['source_id', 'position', 'bookmaker'],
    #     batch_size=500
    # )


def example_latest_odds_for_markets():
    """Example: Get latest odds for specific markets."""
    print("\n=== Example: Get Latest Odds for Markets ===")

    session = SessionLocal()
    try:
        # Get a few market IDs for demo
        sample_markets = session.query(Market.source_id).limit(5).all()
        market_ids = [m.source_id for m in sample_markets]

        if market_ids:
            print(f"Getting latest odds for {len(market_ids)} markets...")

            # This processes the market IDs in chunks internally
            df = get_latest_odds_for_markets(market_ids, session=session)

            print(f"Found odds for {len(df)} markets")
            if not df.empty:
                print(f"Sample data:\n{df.head()}")
        else:
            print("No markets found in database")

    finally:
        session.close()


def main():
    """Run all examples."""
    print("Database Utilities ORM Examples")
    print("=" * 50)
    print("\nThese examples demonstrate how to use the new ORM-based")
    print("database utilities that handle large datasets efficiently")
    print("by using chunking and pagination.\n")

    try:
        example_get_odds_window()
        example_fetch_open_markets()
        example_chunked_market_query()
        example_latest_odds_for_markets()
        example_batch_upsert()

    except Exception as e:
        logger.error(f"Error in examples: {e}")
        raise


if __name__ == "__main__":
    main()
