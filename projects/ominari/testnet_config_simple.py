#!/usr/bin/env python3
"""Simple testnet configuration for safe blockchain trading"""

import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Simple testnet configuration
TESTNET_CONFIG = {
    # Use testnets for safe trading
    'use_testnet': True,
    'default_network': 'optimism_sepolia',
    
    # Data source priorities (1 = highest priority)
    'data_source_priority': {
        'blockchain': 1,     # Always prefer blockchain data
        'database': 2,       # Use database as backup
        'api': 3             # Use API only when absolutely needed
    },
    
    # Market filtering
    'supported_sports': ['soccer', 'football'],
    'supported_outcomes': ['home', 'away', 'draw'],  # Win/loss/draw only
    'min_odds': 1.01,  # Minimum valid odds
    'max_odds': 100.0,  # Maximum valid odds
    
    # Time-based chunking settings
    'chunk_hours': 2.0,  # Group matches into 2-hour time chunks
    'max_hours_ahead': 6.0,  # Only trade matches in next 6 hours
    
    # Execution settings
    'paper_trading_mode': True,
    'simulate_blockchain_calls': True,
    'chunk_size': 50,  # Legacy: execution batch size (kept for compatibility)
}

def setup_testnet_environment():
    """Set up environment variables for testnet trading"""
    
    # Set testnet environment variables
    os.environ['TRADING_MODE'] = 'testnet'
    os.environ['USE_TESTNET'] = '1'
    os.environ['BLOCKCHAIN_NETWORK'] = TESTNET_CONFIG['default_network']
    os.environ['PAPER_TRADING'] = '1'
    os.environ['SIMULATE_BLOCKCHAIN'] = '1'
    
    # Set market filtering
    os.environ['SUPPORTED_SPORTS'] = ','.join(TESTNET_CONFIG['supported_sports'])
    os.environ['SUPPORTED_OUTCOMES'] = ','.join(TESTNET_CONFIG['supported_outcomes'])
    os.environ['CHUNK_SIZE'] = str(TESTNET_CONFIG['chunk_size'])
    os.environ['CHUNK_HOURS'] = str(TESTNET_CONFIG['chunk_hours'])
    os.environ['MAX_HOURS_AHEAD'] = str(TESTNET_CONFIG['max_hours_ahead'])
    
    logger.info("✅ Testnet environment configured")
    logger.info(f"Network: {TESTNET_CONFIG['default_network']}")
    logger.info(f"Paper trading: {TESTNET_CONFIG['paper_trading_mode']}")
    logger.info(f"Supported sports: {TESTNET_CONFIG['supported_sports']}")
    logger.info(f"Time chunks: {TESTNET_CONFIG['chunk_hours']} hours")

def is_testnet_mode() -> bool:
    """Check if running in testnet mode"""
    return (
        os.getenv('TRADING_MODE') == 'testnet' or
        os.getenv('USE_TESTNET') == '1' or
        TESTNET_CONFIG['use_testnet']
    )

if __name__ == "__main__":
    setup_testnet_environment()
    print("🧪 Simple testnet configuration ready!")