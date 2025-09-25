#!/usr/bin/env python3
"""
Inspect Overtime contracts to understand their structure
"""

import os
os.environ['PG_PORT'] = '5999'

from web3 import Web3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Known addresses
CONTRACTS = {
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'sports_amm': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        'name': 'Optimism'
    },
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'sports_amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'name': 'Arbitrum'
    }
}

def inspect_contract(network):
    """Inspect what's deployed at the contract address."""
    config = CONTRACTS[network]
    logger.info(f"\n🔍 Inspecting {config['name']} contracts...")
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {network}")
            return
            
        # Check if contract exists
        code = w3.eth.get_code(config['sports_amm'])
        if code:
            logger.info(f"✅ Contract exists at {config['sports_amm']}")
            logger.info(f"   Code size: {len(code)} bytes")
            
            # Try to read storage slots to understand structure
            logger.info("   Reading storage slots...")
            for slot in range(0, 10):
                try:
                    data = w3.eth.get_storage_at(config['sports_amm'], slot)
                    if data != b'\x00' * 32:
                        logger.info(f"   Slot {slot}: {data.hex()}")
                except:
                    pass
                    
            # Check recent transactions to understand usage
            logger.info("   Checking recent transactions...")
            current_block = w3.eth.block_number
            
            tx_count = 0
            for block_num in range(current_block - 50, current_block):
                try:
                    block = w3.eth.get_block(block_num, full_transactions=True)
                    for tx in block['transactions']:
                        if tx['to'] and tx['to'].lower() == config['sports_amm'].lower():
                            logger.info(f"   Found tx: {tx['hash'].hex()}")
                            logger.info(f"     From: {tx['from']}")
                            logger.info(f"     Method: {tx['input'][:10]}")
                            tx_count += 1
                            if tx_count >= 5:
                                break
                    if tx_count >= 5:
                        break
                except:
                    continue
                    
        else:
            logger.error(f"❌ No contract at {config['sports_amm']}")
            
    except Exception as e:
        logger.error(f"Error inspecting {network}: {e}")

def main():
    """Main function."""
    logger.info("🔍 Overtime Contract Inspector")
    logger.info("=" * 60)
    
    for network in ['optimism', 'arbitrum']:
        inspect_contract(network)

if __name__ == "__main__":
    main()