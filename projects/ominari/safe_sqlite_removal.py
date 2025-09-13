#!/usr/bin/env python3
"""
Safe SQLite Database Removal Script
Only run this after confirming PostgreSQL system is stable for 24-48 hours.
"""

import os
import shutil
import gzip
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_postgresql_health():
    """Verify PostgreSQL is working correctly."""
    try:
        from database import test_connection, get_database_stats
        from models import Market, LookupSport

        logger.info("🔍 Checking PostgreSQL health...")

        # Test connection
        if not test_connection():
            logger.error("❌ PostgreSQL connection failed")
            return False

        # Test data integrity
        from database import SessionLocal
        session = SessionLocal()

        market_count = session.query(Market).count()
        sport_count = session.query(LookupSport).count()

        session.close()

        if market_count < 47000:
            logger.error(f"❌ Market count too low: {market_count} (expected >47,000)")
            return False

        if sport_count < 10:
            logger.error(f"❌ Sport count too low: {sport_count} (expected >10)")
            return False

        logger.info(f"✅ PostgreSQL health check passed:")
        logger.info(f"   Markets: {market_count:,}")
        logger.info(f"   Sports: {sport_count}")

        return True

    except Exception as e:
        logger.error(f"❌ PostgreSQL health check failed: {e}")
        return False


def create_sqlite_archive():
    """Create compressed archive of SQLite database."""
    sqlite_path = "sport_odds.db"

    if not os.path.exists(sqlite_path):
        logger.warning("⚠️  SQLite database not found at sport_odds.db")
        return None

    logger.info("📦 Creating SQLite compressed archive...")

    # Check available disk space
    sqlite_size = os.path.getsize(sqlite_path)
    logger.info(f"SQLite database size: {sqlite_size / (1024**3):.1f} GB")

    # Create archive directory
    archive_dir = "sqlite_archive"
    os.makedirs(archive_dir, exist_ok=True)

    # Create compressed backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = f"{archive_dir}/sport_odds_backup_{timestamp}.db.gz"

    try:
        start_time = time.time()

        with open(sqlite_path, 'rb') as f_in:
            with gzip.open(archive_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        elapsed = time.time() - start_time
        archive_size = os.path.getsize(archive_path)
        compression_ratio = (1 - archive_size / sqlite_size) * 100

        logger.info(f"✅ Archive created successfully:")
        logger.info(f"   Location: {archive_path}")
        logger.info(f"   Size: {archive_size / (1024**3):.1f} GB")
        logger.info(f"   Compression: {compression_ratio:.1f}%")
        logger.info(f"   Time: {elapsed:.1f} seconds")

        return archive_path

    except Exception as e:
        logger.error(f"❌ Archive creation failed: {e}")
        return None


def remove_sqlite_safely():
    """Safely remove SQLite database after confirmation."""
    sqlite_path = "sport_odds.db"

    if not os.path.exists(sqlite_path):
        logger.info("✅ SQLite database already removed")
        return True

    sqlite_size = os.path.getsize(sqlite_path)
    logger.info(f"🗑️  Preparing to remove SQLite database:")
    logger.info(f"   File: {sqlite_path}")
    logger.info(f"   Size: {sqlite_size / (1024**3):.1f} GB")

    # Multiple confirmation steps
    print("\n" + "="*60)
    print("⚠️  FINAL CONFIRMATION FOR SQLite REMOVAL")
    print("="*60)
    print("This will permanently delete the 216GB SQLite database.")
    print("Make sure you have:")
    print("  ✅ PostgreSQL running successfully for 24+ hours")
    print("  ✅ All applications tested with PostgreSQL")
    print("  ✅ Compressed backup archive created")
    print("  ✅ Confidence in PostgreSQL system stability")

    confirmation = input("\nType 'DELETE_SQLITE' to confirm removal: ")
    if confirmation != "DELETE_SQLITE":
        logger.info("❌ Removal cancelled - confirmation failed")
        return False

    # Second confirmation with wait
    print("\nWaiting 5 seconds for final consideration...")
    time.sleep(5)

    final_confirm = input("Final confirmation - type 'YES' to delete: ")
    if final_confirm != "YES":
        logger.info("❌ Removal cancelled - final confirmation failed")
        return False

    # Perform removal
    try:
        logger.info("🗑️  Removing SQLite database...")
        os.remove(sqlite_path)

        # Verify removal
        if not os.path.exists(sqlite_path):
            logger.info(f"✅ SQLite database successfully removed")
            logger.info(f"💾 Freed up {sqlite_size / (1024**3):.1f} GB of disk space")
            return True
        else:
            logger.error("❌ SQLite removal failed - file still exists")
            return False

    except Exception as e:
        logger.error(f"❌ SQLite removal failed: {e}")
        return False


def main():
    """Main SQLite removal process."""
    print("🔧 SAFE SQLite DATABASE REMOVAL")
    print("="*50)
    print("This script will safely remove the SQLite database after verification.")
    print("Only run this after PostgreSQL has been stable for 24-48 hours.")
    print()

    # Step 1: Health check
    if not check_postgresql_health():
        print("❌ PostgreSQL health check failed - aborting removal")
        return False

    # Step 2: Create archive
    archive_path = create_sqlite_archive()
    if not archive_path:
        print("❌ Archive creation failed - aborting removal")
        return False

    # Step 3: Remove SQLite
    if remove_sqlite_safely():
        print("\n✅ SQLite removal completed successfully!")
        print(f"📦 Archive available at: {archive_path}")
        print("🎯 PostgreSQL-only system is now fully deployed")
        return True
    else:
        print("\n❌ SQLite removal failed or cancelled")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)