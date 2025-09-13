#!/usr/bin/env python3
"""
Simple Production Migration - Uses proven working pattern from fixed_sqlite_extractor
"""

from fixed_sqlite_extractor import FixedSQLiteExtractor
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_large_migration():
    """Run large-scale migration using proven extraction method."""

    logger.info("🚀 Starting large-scale production migration")

    # Create extractor with 8 months of data
    extractor = FixedSQLiteExtractor(months_back=8)

    # Start with larger batch
    total_migrated = 0
    batch_sizes = [50000, 100000, 200000]  # Progressively larger batches

    start_time = time.time()

    for batch_size in batch_sizes:
        logger.info(f"📦 Attempting migration batch of {batch_size:,} records")

        batch_start = time.time()
        count = extractor.extract_sample_markets(limit=batch_size)
        batch_time = time.time() - batch_start

        if count > 0:
            total_migrated += count
            rate = count / batch_time if batch_time > 0 else 0
            logger.info(f"✅ Migrated {count:,} markets in {batch_time:.2f}s ({rate:.0f} records/sec)")

            # Verify the migration
            if extractor.verify_postgresql_data():
                logger.info(f"✅ Data verification passed for batch")
            else:
                logger.warning("❌ Data verification failed")

        else:
            logger.warning(f"No records extracted in batch of {batch_size:,}")
            break

        # Brief pause between batches
        time.sleep(2)

    total_time = time.time() - start_time
    average_rate = total_migrated / total_time if total_time > 0 else 0

    logger.info(f"🎯 Migration completed: {total_migrated:,} total records in {total_time:.2f}s")
    logger.info(f"📊 Average rate: {average_rate:.0f} records/second")

    return total_migrated


if __name__ == "__main__":
    migrated = run_large_migration()
    if migrated > 0:
        logger.info("✅ Large-scale migration completed successfully!")
    else:
        logger.error("❌ Migration failed")