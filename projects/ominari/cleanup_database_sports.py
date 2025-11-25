#!/usr/bin/env python3
"""
Database Cleanup - Remove Non-Soccer Markets
Cleans up misclassified and non-soccer markets from the database.
"""
import os
os.environ['PG_PORT'] = '5999'

import logging
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_markets():
    """Show what will be cleaned up."""
    with db_manager.get_db_session() as db:
        # Get non-soccer active markets
        non_soccer = db.query(Market).filter(
            Market.sport != 'Soccer',
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).all()

        print("\n🔍 Non-Soccer Active Markets to Delete:")
        print("=" * 70)

        # Group by sport
        by_sport = {}
        for market in non_soccer:
            sport = market.sport or 'Unknown'
            if sport not in by_sport:
                by_sport[sport] = []
            by_sport[sport].append(market)

        for sport in sorted(by_sport.keys()):
            markets = by_sport[sport]
            print(f"\n{sport}: {len(markets)} markets")
            # Show first 3 examples
            for m in markets[:3]:
                print(f"  • {m.home_team[:30]} vs {m.away_team[:30]}")
            if len(markets) > 3:
                print(f"  ... and {len(markets) - 3} more")

        total_to_delete = len(non_soccer)
        print(f"\n📊 Total markets to delete: {total_to_delete}")

        return total_to_delete

def cleanup_non_soccer_markets(dry_run=True):
    """Remove non-soccer markets from database."""
    with db_manager.get_db_session() as db:
        # Get non-soccer active markets
        non_soccer = db.query(Market).filter(
            Market.sport != 'Soccer',
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).all()

        if dry_run:
            logger.info(f"DRY RUN: Would delete {len(non_soccer)} non-soccer markets")
            return 0

        deleted_markets = 0
        deleted_odds = 0

        for market in non_soccer:
            # Delete associated odds first
            odds_count = db.query(Odd).filter(
                Odd.source_id == market.source_id
            ).delete()
            deleted_odds += odds_count

            # Delete market
            db.delete(market)
            deleted_markets += 1

        db.commit()

        logger.info(f"✅ Deleted {deleted_markets} markets and {deleted_odds} odds")
        return deleted_markets

def main():
    """Main cleanup process."""
    print("\n🧹 Database Cleanup - Non-Soccer Markets")
    print("=" * 70)

    # Analyze what will be deleted
    total_to_delete = analyze_markets()

    if total_to_delete == 0:
        print("\n✅ No non-soccer markets to delete!")
        return

    # Confirm before deleting
    print(f"\n⚠️  This will delete {total_to_delete} markets and their odds.")
    response = input("\nProceed with cleanup? (yes/no): ")

    if response.lower() != 'yes':
        print("❌ Cleanup cancelled.")
        return

    # Perform cleanup
    print("\n🗑️  Deleting non-soccer markets...")
    deleted = cleanup_non_soccer_markets(dry_run=False)

    # Show final state
    with db_manager.get_db_session() as db:
        total_markets = db.query(Market).count()
        soccer_markets = db.query(Market).filter(Market.sport == 'Soccer').count()
        active_soccer = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()

    print("\n📊 Final Database State:")
    print(f"Total markets: {total_markets:,}")
    print(f"Soccer markets: {soccer_markets:,} ({soccer_markets/total_markets*100:.1f}%)")
    print(f"Active soccer: {active_soccer:,}")

    print("\n✅ Cleanup complete!")

if __name__ == "__main__":
    main()
