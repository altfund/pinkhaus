#!/usr/bin/env python3
"""
Check the discovered manager contract
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The manager we found
MANAGER_ADDRESS = '0xB155685132eEd3cD848d220e25a9607DD8871D38'

# Manager ABI methods
MANAGER_ABI = [
    {
        "inputs": [],
        "name": "numActiveMarkets",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "index", "type": "uint256"},
            {"internalType": "uint256", "name": "pageSize", "type": "uint256"}
        ],
        "name": "activeMarkets",
        "outputs": [{"internalType": "address[]", "name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "index", "type": "uint256"}],
        "name": "getActiveMarketAddress",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    }
]

def check_arbiscan(address):
    """Check contract info from Arbiscan."""
    try:
        url = f"https://api.arbiscan.io/api?module=contract&action=getsourcecode&address={address}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1' and data['result'][0]['ContractName']:
                return data['result'][0]['ContractName']
    except:
        pass
    return None

def main():
    """Main function."""
    logger.info("🔍 Checking Discovered Manager Contract")
    logger.info("=" * 60)
    
    try:
        w3 = Web3(Web3.HTTPProvider('https://arb1.arbitrum.io/rpc'))
        if not w3.is_connected():
            logger.error("Failed to connect")
            return
            
        logger.info(f"Manager address: {MANAGER_ADDRESS}")
        
        # Check if verified
        contract_name = check_arbiscan(MANAGER_ADDRESS)
        if contract_name:
            logger.info(f"✅ Verified as: {contract_name}")
        else:
            logger.info("❌ Not verified on Arbiscan")
            
        # Check code
        code = w3.eth.get_code(MANAGER_ADDRESS)
        logger.info(f"Code size: {len(code)} bytes")
        
        # Try to call manager methods
        contract = w3.eth.contract(address=Web3.to_checksum_address(MANAGER_ADDRESS), abi=MANAGER_ABI)
        
        try:
            num_markets = contract.functions.numActiveMarkets().call()
            logger.info(f"\n📊 Active markets: {num_markets}")
            
            if num_markets > 0:
                # Get first 5 markets
                logger.info("\nFirst 5 markets:")
                for i in range(min(5, num_markets)):
                    try:
                        market_addr = contract.functions.getActiveMarketAddress(i).call()
                        logger.info(f"  {i}: {market_addr}")
                        # Check if it's verified
                        market_name = check_arbiscan(market_addr)
                        if market_name:
                            logger.info(f"     Verified as: {market_name}")
                    except:
                        pass
        except Exception as e:
            logger.error(f"Error calling manager methods: {e}")
            
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()