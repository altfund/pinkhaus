#!/usr/bin/env python3
"""
Debug Overtime V2 smart contracts to find the correct methods
"""

import logging
from web3 import Web3
import json
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Contract addresses
CONTRACTS = {
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'sports_amm_v2': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        'name': 'Optimism'
    },
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'sports_amm_v2': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'name': 'Arbitrum'
    }
}

def debug_contract(network: str):
    """Debug a contract to find available methods."""
    config = CONTRACTS[network]
    logger.info(f"\n=== Debugging {config['name']} Contract ===")
    
    try:
        # Connect to blockchain
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {network}")
            return
            
        logger.info(f"Connected to {network} at block {w3.eth.block_number:,}")
        
        contract_address = Web3.to_checksum_address(config['sports_amm_v2'])
        logger.info(f"Contract address: {contract_address}")
        
        # Get contract code to verify it exists
        code = w3.eth.get_code(contract_address)
        if code == b'':
            logger.error("Contract not found at this address!")
            return
            
        logger.info(f"Contract code length: {len(code)} bytes")
        
        # Try to get the ABI from Etherscan-like APIs
        logger.info("Fetching ABI from blockchain explorers...")
        
        if network == 'optimism':
            # Try Optimism Etherscan
            api_url = f"https://api-optimistic.etherscan.io/api?module=contract&action=getabi&address={contract_address}"
        else:
            # Try Arbiscan
            api_url = f"https://api.arbiscan.io/api?module=contract&action=getabi&address={contract_address}"
            
        try:
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == '1':
                    abi = json.loads(data['result'])
                    logger.info(f"✅ Found ABI with {len(abi)} methods")
                    
                    # List all view/pure methods
                    view_methods = []
                    for item in abi:
                        if item.get('type') == 'function' and item.get('stateMutability') in ['view', 'pure']:
                            name = item['name']
                            inputs = [inp['type'] for inp in item.get('inputs', [])]
                            view_methods.append(f"{name}({', '.join(inputs)})")
                    
                    logger.info("Available view methods:")
                    for method in sorted(view_methods):
                        logger.info(f"  - {method}")
                    
                    # Try some common method names
                    contract = w3.eth.contract(address=contract_address, abi=abi)
                    
                    common_methods = [
                        'activeMarkets', 'getAllActiveMarkets', 'getAllActiveGameIds',
                        'getActiveGames', 'getMarkets', 'activeGames', 'marketsByDate',
                        'totalActiveMarkets', 'activeMarketsCount', 'getSportMarkets'
                    ]
                    
                    logger.info("\nTesting common methods:")
                    for method_name in common_methods:
                        try:
                            if hasattr(contract.functions, method_name):
                                logger.info(f"  ✅ {method_name} - method exists")
                                
                                # Try to call it
                                try:
                                    method = getattr(contract.functions, method_name)
                                    result = method().call()
                                    logger.info(f"    Result type: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                                    
                                    if isinstance(result, list) and len(result) > 0:
                                        logger.info(f"    First element: {result[0] if len(result) > 0 else 'None'}")
                                        
                                except Exception as e:
                                    logger.warning(f"    ❌ Call failed: {e}")
                            else:
                                logger.info(f"  ❌ {method_name} - not found")
                                
                        except Exception as e:
                            logger.error(f"  ❌ Error checking {method_name}: {e}")
                
                else:
                    logger.warning(f"API error: {data.get('message', 'Unknown error')}")
            else:
                logger.warning(f"API request failed: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Could not fetch ABI: {e}")
            
            # Try with minimal ABI and test basic methods
            logger.info("Trying with basic ERC20-like methods...")
            
            basic_abi = [
                {"inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
                {"inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "stateMutability": "view", "type": "function"}
            ]
            
            contract = w3.eth.contract(address=contract_address, abi=basic_abi)
            
            try:
                # Test if it responds to any standard calls
                for method in ['totalSupply', 'name']:
                    try:
                        result = getattr(contract.functions, method)().call()
                        logger.info(f"  ✅ {method}(): {result}")
                    except Exception as e:
                        logger.info(f"  ❌ {method}(): {e}")
            except Exception as e:
                logger.error(f"Basic method test failed: {e}")
        
    except Exception as e:
        logger.error(f"Error debugging {network}: {e}")

def main():
    """Debug all contracts."""
    logger.info("🔍 Debugging Overtime V2 Smart Contracts")
    logger.info("=" * 60)
    
    for network in ['optimism', 'arbitrum']:
        debug_contract(network)
        
    logger.info("\n🎯 Debug complete!")

if __name__ == "__main__":
    main()