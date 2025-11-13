#!/usr/bin/env python3
"""
Check for real blockchain addresses in markets
"""
import os

# Set environment for PostgreSQL on port 5999
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_blockchain_ids():
    """Check what kinds of IDs we have"""
    
    with db_manager.get_db_session() as db:
        # Check different ID patterns
        logger.info("=== Checking Market ID Patterns ===")
        
        # Get sample of different source_id patterns
        markets = db.query(Market).limit(100).all()
        
        patterns = {}
        for market in markets:
            if market.source_id:
                # Categorize by pattern
                if market.source_id.startswith('v2_0x'):
                    pattern = 'v2_0x (encoded)'
                elif market.source_id.startswith('overtime_real_0x'):
                    pattern = 'overtime_real_0x (encoded)'
                elif market.source_id.startswith('0x') and len(market.source_id) == 42:
                    pattern = '0x (real blockchain address)'
                elif market.source_id.startswith('0x'):
                    pattern = f'0x (length {len(market.source_id)})'
                else:
                    pattern = 'other'
                
                if pattern not in patterns:
                    patterns[pattern] = []
                patterns[pattern].append(market.source_id)
        
        # Show patterns found
        for pattern, ids in patterns.items():
            logger.info(f"\n{pattern}: {len(ids)} examples")
            for id in ids[:3]:  # Show first 3
                logger.info(f"  {id}")
        
        # Check if we have blockchain_id field
        logger.info("\n=== Checking for blockchain_id field ===")
        sample = db.query(Market).first()
        if hasattr(sample, 'blockchain_id'):
            logger.info("Market has blockchain_id field")
            # Check markets with blockchain_id
            with_blockchain = db.query(Market).filter(
                Market.blockchain_id.isnot(None)
            ).limit(10).all()
            
            logger.info(f"\nFound {len(with_blockchain)} markets with blockchain_id:")
            for market in with_blockchain:
                logger.info(f"  {market.home_team} vs {market.away_team}")
                logger.info(f"    source_id: {market.source_id}")
                logger.info(f"    blockchain_id: {market.blockchain_id}")
        else:
            logger.info("Market does NOT have blockchain_id field")

if __name__ == "__main__":
    check_blockchain_ids()