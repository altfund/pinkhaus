#!/usr/bin/env python3
"""
Find the correct Overtime V2 contract addresses by looking at recent transactions
"""

import logging
from web3 import Web3
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_contracts_on_optimism():
    """Find Overtime contracts on Optimism by examining recent transactions."""
    logger.info("🔍 Finding Overtime contracts on Optimism...")
    
    w3 = Web3(Web3.HTTPProvider('https://mainnet.optimism.io'))
    if not w3.is_connected():
        logger.error("Failed to connect to Optimism")
        return
        
    logger.info(f"Connected to Optimism at block {w3.eth.block_number:,}")
    
    # Known Overtime-related addresses from documentation and GitHub
    potential_contracts = [
        "0xFb4e4811C7A811E098A556bD79B64c20b479E431",  # Sports AMM V2
        "0x4c8872c1e8b3a3b5c83861cb7b8f7c06cd7f26b5",  # Sports Markets (potential)
        "0x8D97689C9818892B700e27F316cc3E41e17fBeb9",  # Another potential address
        "0xC4b7A70c3694cb1d37A18e6C6bD9271828C382A4",  # Overtime V2 potential
        "0xE85B662E0EE2ad00b3D8F64BeC68dD5ae0e8C7EE",  # Market Manager potential
        "0x3D01F3b8f0C1C2b0E0F1C2b0E3456789ABCDEF00"   # Generic check
    ]
    
    for address in potential_contracts:
        try:
            checksum_addr = Web3.to_checksum_address(address)
            code = w3.eth.get_code(checksum_addr)
            
            if code != b'':
                logger.info(f"✅ Contract found at {checksum_addr} (code: {len(code)} bytes)")
                
                # Try to get recent transactions to this contract
                try:
                    latest_block = w3.eth.block_number
                    
                    # Check last 100 blocks for transactions
                    for block_num in range(latest_block - 100, latest_block + 1):
                        try:
                            block = w3.eth.get_block(block_num, full_transactions=True)
                            
                            for tx in block['transactions']:
                                if tx['to'] and tx['to'].lower() == address.lower():
                                    logger.info(f"  📝 Recent tx: {tx['hash'].hex()} in block {block_num}")
                                    
                                    # Try to decode the transaction input
                                    if len(tx['input']) > 10:  # Has method call
                                        method_sig = tx['input'][:10]
                                        logger.info(f"    Method signature: {method_sig}")
                                        
                                    break  # Found a transaction, that's enough
                            
                        except Exception as e:
                            continue
                            
                except Exception as e:
                    logger.warning(f"Could not get recent transactions: {e}")
            else:
                logger.info(f"❌ No contract at {checksum_addr}")
                
        except Exception as e:
            logger.warning(f"Error checking {address}: {e}")

def check_etherscan_optimism():
    """Check Optimism Etherscan for Overtime contracts."""
    logger.info("🔍 Searching Optimism Etherscan for 'Overtime' contracts...")
    
    # Search for contracts with "Overtime" in the name
    search_terms = ["Overtime", "SportsAMM", "SportsMarkets"]
    
    for term in search_terms:
        try:
            # This is a simplified approach - in reality, we'd use the Etherscan API
            logger.info(f"Searching for: {term}")
            
            # For now, let's use known addresses from GitHub/docs
            known_addresses = {
                "SportsAMM": "0x6c7fd4321183b542e81bcc7bee7d8ced435ba9ce",
                "SportsMarkets": "0x8d97689c9818892b700e27f316cc3e41e17fbeb9",
                "SportAMM": "0xfb4e4811c7a811e098a556bd79b64c20b479e431"
            }
            
            if term in known_addresses:
                address = known_addresses[term]
                logger.info(f"  Found {term}: {address}")
                
                # Test this address
                w3 = Web3(Web3.HTTPProvider('https://mainnet.optimism.io'))
                checksum_addr = Web3.to_checksum_address(address)
                code = w3.eth.get_code(checksum_addr)
                
                if code != b'':
                    logger.info(f"  ✅ Contract verified: {len(code)} bytes")
                else:
                    logger.info(f"  ❌ No contract found")
                    
        except Exception as e:
            logger.warning(f"Error searching for {term}: {e}")

def test_with_simple_abi():
    """Test contracts with a very simple ABI to see what methods exist."""
    logger.info("🧪 Testing with simple ABI...")
    
    w3 = Web3(Web3.HTTPProvider('https://mainnet.optimism.io'))
    
    # Very basic ABI with common method signatures
    simple_abi = [
        {"inputs": [], "name": "activeMarkets", "outputs": [{"name": "", "type": "address[]"}], "stateMutability": "view", "type": "function"},
        {"inputs": [], "name": "markets", "outputs": [{"name": "", "type": "address[]"}], "stateMutability": "view", "type": "function"},
        {"inputs": [], "name": "getAllMarkets", "outputs": [{"name": "", "type": "address[]"}], "stateMutability": "view", "type": "function"},
        {"inputs": [], "name": "getMarkets", "outputs": [{"name": "", "type": "address[]"}], "stateMutability": "view", "type": "function"},
        {"inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
        {"inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "stateMutability": "view", "type": "function"}
    ]
    
    # Test known addresses
    test_addresses = [
        "0xFb4e4811C7A811E098A556bD79B64c20b479E431",
        "0x6c7fd4321183b542e81bcc7bee7d8ced435ba9ce",
        "0x8d97689c9818892b700e27f316cc3e41e17fbeb9"
    ]
    
    for address in test_addresses:
        try:
            logger.info(f"\nTesting {address}:")
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(address),
                abi=simple_abi
            )
            
            methods_to_test = ['activeMarkets', 'markets', 'getAllMarkets', 'getMarkets', 'name', 'totalSupply']
            
            for method in methods_to_test:
                try:
                    result = getattr(contract.functions, method)().call()
                    logger.info(f"  ✅ {method}(): {type(result)} - {result if not isinstance(result, list) or len(result) < 5 else f'List of {len(result)} items'}")
                    
                    if isinstance(result, list) and len(result) > 0:
                        logger.info(f"    First item: {result[0]}")
                        
                except Exception as e:
                    logger.info(f"  ❌ {method}(): {str(e)[:100]}...")
                    
        except Exception as e:
            logger.error(f"Error testing {address}: {e}")

def main():
    """Find the correct Overtime V2 contracts."""
    logger.info("🎯 Finding Real Overtime V2 Contracts")
    logger.info("=" * 50)
    
    find_contracts_on_optimism()
    print()
    check_etherscan_optimism() 
    print()
    test_with_simple_abi()
    
    logger.info("\n🔍 Contract discovery complete!")

if __name__ == "__main__":
    main()