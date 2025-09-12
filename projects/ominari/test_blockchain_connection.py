#!/usr/bin/env python3
"""Test blockchain connection and debug issues."""

import os
from web3 import Web3
from blockchain_reader import BlockchainReader, BlockchainConfig
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_connection():
    """Test blockchain connection."""
    print("Testing blockchain connection...")
    
    # Test direct Web3 connection
    network = 'optimism'
    config = BlockchainConfig.NETWORKS[network]
    
    # Check RPC URL
    rpc_url = os.getenv(f'{network.upper()}_RPC_URL', config['rpc_url'])
    print(f"RPC URL: {rpc_url}")
    
    try:
        # Create Web3 instance
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        print(f"Web3 connected: {w3.is_connected()}")
        
        if w3.is_connected():
            # Get block number
            block_num = w3.eth.block_number
            print(f"Current block number: {block_num}")
            print(f"Block number type: {type(block_num)}")
            
            # Get chain ID
            chain_id = w3.eth.chain_id
            print(f"Chain ID: {chain_id}")
            
            # Test BlockchainReader
            reader = BlockchainReader(network)
            print(f"\nBlockchainReader check_connection: {reader.check_connection()}")
            
            # Test fetch_recent_markets
            print("\nTesting fetch_recent_markets...")
            markets = reader.fetch_recent_markets(hours_back=1)
            print(f"Markets fetched: {len(markets)}")
            
        else:
            print("ERROR: Web3 not connected!")
            
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_connection()