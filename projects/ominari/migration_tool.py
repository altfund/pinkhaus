#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration script to switch between data collectors
Can be used to toggle between old API-first and new blockchain-first approaches
"""

import asyncio
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_new_collector():
    """Run the new blockchain/GraphQL-first collector."""
    from free_data_pull_v2 import main as new_main
    logger.info("Running new data collector (blockchain/GraphQL priority)")
    await new_main()


def run_old_collector():
    """Run the old API-first collector."""
    import free_data_pull
    logger.info("Running old data collector (API-first)")
    # The old collector uses synchronous code
    free_data_pull.main()


async def run_comparison():
    """Run both collectors and compare results."""
    logger.info("Running comparison mode")
    
    # Run old collector
    logger.info("\n=== Running OLD collector ===")
    start_old = datetime.now()
    try:
        run_old_collector()
        old_duration = (datetime.now() - start_old).total_seconds()
        logger.info(f"Old collector completed in {old_duration:.2f} seconds")
    except Exception as e:
        logger.error(f"Old collector failed: {e}")
    
    # Run new collector
    logger.info("\n=== Running NEW collector ===")
    start_new = datetime.now()
    try:
        await run_new_collector()
        new_duration = (datetime.now() - start_new).total_seconds()
        logger.info(f"New collector completed in {new_duration:.2f} seconds")
    except Exception as e:
        logger.error(f"New collector failed: {e}")
    
    # Compare database results
    logger.info("\n=== Comparison complete ===")


def main():
    parser = argparse.ArgumentParser(description='Data collector migration tool')
    parser.add_argument(
        '--mode', 
        choices=['old', 'new', 'compare'],
        default='new',
        help='Which collector to run (default: new)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test mode - do not actually save data'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'old':
        run_old_collector()
    elif args.mode == 'new':
        asyncio.run(run_new_collector())
    elif args.mode == 'compare':
        asyncio.run(run_comparison())


if __name__ == '__main__':
    main()
