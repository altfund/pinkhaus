#!/usr/bin/env python3
"""
Find real Overtime contracts by checking known patterns
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Addresses from transaction analysis and storage
CANDIDATES = {
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'sports_amm': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        # From storage slot 0
        'potential_manager': '0xef2b0a2ffcdcd3b53d81800d19a543136f26a6b8',
        'name': 'Optimism'
    },
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'sports_amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        # From storage slot 0
        'potential_manager': '0x0089282ac624bbbda7ec472d63ee99643d7edf4a',
        'name': 'Arbitrum'
    }
}

def check_arbiscan_api(address):
    """Check if contract is verified on Arbiscan."""
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

def check_optimistic_api(address):
    """Check if contract is verified on Optimistic Etherscan."""
    try:
        url = f"https://api-optimistic.etherscan.io/api?module=contract&action=getsourcecode&address={address}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1' and data['result'][0]['ContractName']:
                return data['result'][0]['ContractName']
    except:
        pass
    return None

def find_market_in_recent_txs(w3, amm_address):
    """Find market addresses from recent AMM transactions."""
    logger.info("Checking recent AMM transactions...")
    
    markets = set()
    
    try:
        current_block = w3.eth.block_number
        
        # Check last 50 blocks
        for block_num in range(current_block - 50, current_block):
            try:
                block = w3.eth.get_block(block_num, full_transactions=True)
                
                for tx in block['transactions']:
                    if tx['to'] and tx['to'].lower() == amm_address.lower():
                        # Decode transaction input
                        # Method 0x942b67dc is common - it likely involves a market address
                        if tx['input'].startswith('0x942b67dc'):
                            # Try to extract address parameter
                            try:
                                # Skip method signature (4 bytes = 8 hex chars + 0x = 10 chars)
                                data = tx['input'][10:]
                                # First parameter is often an address (32 bytes, last 20 are address)
                                if len(data) >= 64:
                                    addr_hex = '0x' + data[24:64]
                                    addr = Web3.to_checksum_address(addr_hex)
                                    
                                    # Verify it's a contract
                                    code = w3.eth.get_code(addr)
                                    if code and len(code) > 100:
                                        markets.add(addr)
                            except:
                                pass
            except:
                continue
                
    except Exception as e:
        logger.error(f"Error scanning transactions: {e}")
        
    return list(markets)

def main():
    """Main function."""
    logger.info("🔍 Finding Real Overtime Contracts")
    logger.info("=" * 60)
    
    # Check Arbitrum first (more activity)
    network = 'arbitrum'
    config = CANDIDATES[network]
    
    logger.info(f"\n📡 Checking {config['name']}...")
    
    # Check if AMM is verified
    contract_name = check_arbiscan_api(config['sports_amm'])
    if contract_name:
        logger.info(f"✅ AMM is verified as: {contract_name}")
    
    # Check potential manager
    manager_name = check_arbiscan_api(config['potential_manager'])
    if manager_name:
        logger.info(f"✅ Potential manager is: {manager_name}")
    
    # Connect and find markets
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if w3.is_connected():
            markets = find_market_in_recent_txs(w3, config['sports_amm'])
            
            if markets:
                logger.info(f"\n📊 Found {len(markets)} potential Game Markets:")
                for addr in markets[:5]:
                    name = check_arbiscan_api(addr)
                    logger.info(f"  • {addr}")
                    if name:
                        logger.info(f"    Contract type: {name}")
            else:
                logger.info("No markets found in recent transactions")
                
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()